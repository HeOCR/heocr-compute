import datetime as dt
import fcntl
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from heocr_compute.remote import budget, manifest, profiles, runner
from heocr_compute.remote.ledger import Ledger
from heocr_compute.remote.policy import Policy, Refused
from heocr_compute.remote.runner import Interrupted, Runner, Spec, sweep
from heocr_compute.remote.ssh import SSHError

NOW = dt.datetime(2026, 11, 18, 12, 0, tzinfo=dt.timezone.utc)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeAPI:
    def __init__(self, cost=0.06, balance=5.0, status="RUNNING"):
        self.cost, self._balance, self.status = cost, balance, status
        self.created, self.terminated, self.pods = [], [], {}
        self.terminate_works = True

    def balance(self):
        return self._balance

    def cpu_vcpu_range(self, flavor):
        return 2, 32

    def cpu_price_per_vcpu(self, flavor):
        return 0.03

    def gpu_price(self, gpu, cloud):
        return 0.89

    def create_pod(self, body):
        self.created.append(body)
        pid = f"pod{len(self.created)}"
        self.pods[pid] = {"id": pid, "name": body["name"], "cost": self.cost, "status": self.status,
                          "createdAt": NOW.isoformat(), "cpu": body.get("cpu"), "gpu": body.get("gpu")}
        return dict(self.pods[pid])

    def get_pod(self, pid):
        p = self.pods.get(pid)
        if p is None:
            return None
        return {**p, "ssh": {"direct": {"host": "h", "port": 22, "username": "root"}}}

    def list_pods(self, *, include_cluster_pods=False):
        return [dict(p) for p in self.pods.values()]

    def terminate(self, pid):
        self.terminated.append(pid)
        if self.terminate_works:
            self.pods.pop(pid, None)


class FakeRemote:
    """A pod's file system and shell, as far as the runner uses them."""

    def __init__(self, out_files, exit_code="0", fail_on=None, corrupt_once=()):
        self.out_files = out_files
        self.exit_code = exit_code
        self.fail_on = fail_on
        self.uploaded = None
        self.corrupt = set(corrupt_once)
        self.commands = []
        self.deadman = "armed\n"
        self.on_fire = None

    def run(self, command, input=None, timeout=600, check=True):
        self.commands.append(command)
        if "pod.json" in command:
            self.pod_json = json.loads(input)
        if self.fail_on and self.fail_on in command:
            exc = self.fail_on_exc if hasattr(self, "fail_on_exc") else SSHError("boom")
            raise exc
        ok = SimpleNamespace(returncode=0, stdout="", stderr="")
        if command.startswith("sh /heocr/deadman.sh now"):
            ok.stdout = "fired\n"
            if self.on_fire:
                self.on_fire()
        elif command.startswith("sh /heocr/deadman.sh"):
            ok.stdout = self.deadman
        elif "job.started" in command:
            self.starts = getattr(self, "starts", 0) + 1
            if self.starts <= getattr(self, "drop_starts", 0):
                raise SSHError("ssh timed out after 60s")
            ok.stdout = "started\n"
        elif command.startswith("cat /heocr/exit_code"):
            ok.stdout = self.exit_code
        elif "find . -type f" in command:
            ok.stdout = "".join(f"{len(b)}\t{rel}\n" for rel, b in self.out_files.items())
        elif "sha256sum --" in command:
            ok.stdout = "".join(f"{hashlib.sha256(b).hexdigest()}  {rel}\n" for rel, b in self.out_files.items()
                                if rel in command)
        return ok

    def upload(self, root, rels, dest, *, timeout=None):
        self.uploaded = list(rels)

    def download(self, src, rels, dest):
        for rel in rels:
            f = Path(dest) / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            body = self.out_files[rel]
            if rel in self.corrupt:
                self.corrupt.discard(rel)
                body = b"corrupted in transit"
            f.write_bytes(body)
        return list(rels)


OUT = {
    "job.log": b"invented log line\n",
    "scores/score.json": b'{"cer": 0.01}\n',
    "final_checkpoint": b"\x00" * 64,
    "heb.traineddata": b"\x01" * 32,
    "heb_1.234_10_100.checkpoint": b"\x02" * 16,   # intermediate: stays on the pod
    "page.png": b"\x89PNG",                         # not a checkpoint, log or score
}


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.inputs = d / "inputs"
        self.inputs.mkdir()
        (self.inputs / "lines.txt").write_text("invented training line\n")
        (self.inputs / "job.sh").write_text("echo invented\n")
        (d / "roots.toml").write_text(f'[roots]\nsynthetic = ["{self.inputs}"]\n')
        (d / "deny.txt").write_text("*/invented-private/*\n")
        self.policy = Policy.load(str(d / "roots.toml"), str(d / "deny.txt"))
        manifest.dump(manifest.build(self.inputs, self.policy), d / "inputs.json")
        self.manifest = d / "inputs.json"
        self.results = d / "results"
        self.clock = Clock()
        self.ledger = Ledger(d / "state" / "spend.jsonl", now=lambda: NOW)
        self.api = FakeAPI()
        self.patch = mock.patch.object(runner, "public_key", lambda: "ssh-ed25519 AAAAinvented test")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def spec(self, **kw):
        args = dict(profile=profiles.PROFILES["smoke"], manifest_path=self.manifest, root=self.inputs,
                    cmd="bash in/job.sh", hours=0.25, run_cap=0.5, results=self.results, yes=True,
                    policy=self.policy)
        args.update(kw)
        return Spec(**args)

    def make(self, remote):
        return Runner(self.api, self.ledger, out=lambda *a: None, sleep=self.clock.sleep, clock=self.clock,
                      now=lambda: NOW, poll_seconds=30, remote_factory=lambda pod: remote, keep_awake=False)

    def final(self):
        return [e for e in self.ledger.events() if e["event"] == "final"]

    def test_autonomous_archive_detaches_only_after_startup_and_preserves_budget(self):
        remote = FakeRemote(dict(OUT))
        settings = {"account": "a" * 32, "bucket": "invented-archive", "max_bytes": 2 << 30}
        credentials = self.inputs.parent / "invented-credential.env"
        remote_commands = remote.run

        def capture(command, input=None, **kwargs):
            if "cat > /heocr/.archive-secret" in command:
                remote.secret = json.loads(input)
            return remote_commands(command, input=input, **kwargs)

        remote.run = capture
        with mock.patch("heocr_compute.remote.r2_archive.ArchiveStore") as store:
            store.return_value.scope = settings
            store.return_value.urls.return_value = {"invented": "presigned-secret"}
            result = self.make(remote).run(self.spec(archive_credentials=credentials))
        self.assertEqual(result["status"], "archive_running", result)
        state = self.ledger.runs()[result["run_id"]]
        self.assertEqual(state["reserve"]["r2_archive"], settings)
        self.assertEqual(state["startup"]["status"], "running")
        self.assertNotIn("presigned-secret", json.dumps(state))
        self.assertEqual(remote.secret["urls"], {"invented": "presigned-secret"})
        self.assertFalse(any("presigned-secret" in cmd for cmd in remote.commands))
        self.assertEqual(self.api.terminated, [])
        self.assertEqual(self.final(), [])
        self.assertFalse(self.results.exists())
        self.assertEqual(sweep(self.api, self.ledger, now=lambda: NOW, pid_alive=lambda _: False,
                               sleep=self.clock.sleep, out=lambda *_: None), [])

    def test_autonomous_archive_ready_failure_terminates_the_pod(self):
        remote = FakeRemote(dict(OUT))
        remote_call = remote.run

        def not_ready(command, **kwargs):
            if command == "test -f /heocr/archive.ready":
                return SimpleNamespace(returncode=1)
            return remote_call(command, **kwargs)

        remote.run = not_ready
        with mock.patch("heocr_compute.remote.r2_archive.ArchiveStore") as store:
            store.return_value.scope = {"account": "a" * 32, "bucket": "invented-archive", "max_bytes": 2 << 30}
            store.return_value.urls.return_value = {}
            result = self.make(remote).run(self.spec(archive_credentials=self.inputs.parent / "credentials"))
        self.assertEqual(result["status"], "error")
        self.assertIn("did not acknowledge startup", result["error"])
        self.assertEqual(self.api.terminated, ["pod1"])
        self.assertEqual(self.final()[0]["status"], "error")

    def test_end_to_end(self):
        remote = FakeRemote(dict(OUT))
        res = self.make(remote).run(self.spec())
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(sorted(res["pulled"]), ["final_checkpoint", "heb.traineddata", "job.log", "scores/score.json"])
        dest = self.results / res["run_id"]
        self.assertFalse((dest / "page.png").exists())
        self.assertFalse((dest / "heb_1.234_10_100.checkpoint").exists())
        sums = manifest.parse_sha256sum((dest / "SHA256SUMS").read_text())
        for rel, digest in sums.items():
            self.assertEqual(manifest.sha256_file(dest / rel), digest)
        self.assertEqual(sorted(remote.uploaded), ["job.sh", "lines.txt"])
        self.assertTrue(any("sha256sum -c" in c for c in remote.commands))
        self.assertTrue(any("cat > /heocr/out/pod.json" in c for c in remote.commands))
        self.assertEqual(self.api.terminated, [res["pod_id"]])
        self.assertEqual(remote.pod_json["pod_type"], "cpu3cx2")
        self.assertEqual(remote.pod_json["pod_id"], res["pod_id"])
        body = self.api.created[0]
        self.assertTrue(body["name"].startswith("heocr-smoke-"))
        self.assertEqual(body["cpu"], {"id": "cpu3c", "vcpuCount": 2})
        self.assertTrue(body["startSsh"])
        [f] = self.final()
        self.assertEqual(f["status"], "ok")
        self.assertEqual(f["pod_type"], "cpu3cx2")
        self.assertGreater(f["usd"], 0)
        self.assertEqual(self.ledger.month_to_date()["open_estimate"], 0)

    def test_dry_run_and_explicit_fail_fast_do_not_join_the_queue(self):
        self.api.pods["existing"] = {"id": "existing", "name": "other-account-workload", "status": "PROVISIONING",
                                     "cpu": {"vcpuCount": 64}, "runtime": None}
        runner = self.make(None)
        self.assertEqual(runner.run(self.spec(dry_run=True))["status"], "dry_run")
        with self.assertRaisesRegex(Refused, "requested 2 vCPUs, 0 of 64 available"):
            runner.run(self.spec(capacity_wait_hours=None))
        self.assertEqual(self.api.created, [])
        self.assertEqual(self.api.terminated, [])
        self.assertEqual(self.ledger.events(), [])

    def test_unconfirmed_cpu_reservations_hold_slots_until_the_pod_is_listed(self):
        run_id = "20261118-120000-abcd"
        self.ledger.account = "example"
        self.ledger.reserve(run_id, profile="cpu", pod_type="cpu3cx32", usd_est=0.5, autonomous=True)
        self.ledger.reserve("other", account="other", profile="cpu", vcpus=32, usd_est=0.5)
        self.api.pods["busy"] = {"id": "busy", "name": "busy", "status": "RUNNING", "cpu": {"vcpuCount": 32}}
        self.api.pods["gpu"] = {"id": "gpu", "name": "gpu", "status": "RUNNING", "gpu": {"vcpuCount": 32},
                                "cpu": {"vcpuCount": 32}}
        self.api.pods["stopped"] = {"id": "stopped", "name": "stopped", "status": "EXITED"}
        runner = self.make(None)
        with self.ledger.locked(), self.assertRaisesRegex(Refused, "32 in pods, 32 in unconfirmed reservations"):
            runner.require_cpu_capacity(profiles.PROFILES["cpu"])
        self.api.pods["busy"]["name"] = "heocr-hf-" + run_id
        with self.ledger.locked():
            runner.require_cpu_capacity(profiles.PROFILES["cpu"])

    def test_missing_pod_is_rechecked_by_exact_id(self):
        self.ledger.reserve("existing", profile="cpu", vcpus=32, usd_est=0.5)
        self.ledger.pod("existing", pod_id="unlisted")
        self.api.pods["unlisted"] = {"id": "unlisted", "name": "unlisted", "status": "STARTING",
                                     "cpu": {"vcpuCount": 32}}
        self.api.list_pods = mock.Mock(return_value=[])
        runner = self.make(None)
        full_size = profiles.PROFILES["cpu"].sized(vcpus=64)
        with self.ledger.locked(), self.assertRaisesRegex(Refused, "32 of 64 available"):
            runner.require_cpu_capacity(full_size)
        self.api.pods.pop("unlisted")
        with self.ledger.locked():
            runner.require_cpu_capacity(full_size)
        self.api.list_pods.assert_called_with(include_cluster_pods=True)

    def test_unverifiable_pod_blocks_cpu_but_not_gpu_admission(self):
        self.api.pods["unknown"] = {"id": "unknown", "name": "unknown", "status": "RUNNING",
                                     "cpu": {"vcpuCount": True}}
        runner = self.make(None)
        with self.ledger.locked():
            runner.require_cpu_capacity(profiles.PROFILES["gpu"])
            with self.assertRaisesRegex(Refused, "verifiable RunPod compute type and vCPU count"):
                runner.require_cpu_capacity(profiles.PROFILES["cpu"])

    def test_capacity_wait_releases_ledger_lock_and_launches_when_cpu_is_free(self):
        for vcpus in (32, 16, 8, 4, 2):
            pod_id = f"pod-{vcpus}"
            self.api.pods[pod_id] = {"id": pod_id, "name": pod_id, "status": "RUNNING", "cpu": {"vcpuCount": vcpus}}
        run = self.make(FakeRemote(dict(OUT)))
        run.out, run.confirm = mock.Mock(), mock.Mock(return_value="y")

        def sleep(seconds):
            if self.api.created:
                self.clock.sleep(seconds)
                return
            self.assertEqual(seconds, 300)
            waiting, = self.ledger.waiting()
            self.assertEqual((waiting["event"], waiting["vcpus"], waiting["wait_seconds"]), ("queue", 8, 21_600))
            self.assertEqual(self.ledger.month_to_date()["total"], 0)
            with self.ledger.path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.api.pods.pop("pod-4")
            self.api.pods.pop("pod-2")
            self.clock.sleep(seconds)

        run.sleep = sleep
        request = profiles.PROFILES["cpu"].sized(vcpus=8)
        self.assertEqual(run.run(self.spec(profile=request, yes=False))["status"], "ok")
        self.assertEqual(len(self.api.created), 1)
        self.assertEqual(self.ledger.waiting(), [])
        self.assertEqual([e["event"] for e in self.ledger.events()], ["queue", "reserve", "pod", "final"])
        run.confirm.assert_called_once()
        self.assertTrue(any("Waiting 300s" in call.args[0] for call in run.out.call_args_list))

    def test_capacity_wait_expires_without_reserving_or_creating(self):
        self.api.pods["busy"] = {"id": "busy", "name": "busy", "status": "RUNNING", "cpu": {"vcpuCount": 64}}
        run = self.make(None)
        run.sleep = mock.Mock(side_effect=self.clock.sleep)
        with self.assertRaisesRegex(Refused, "capacity wait expired after 0.1 h"):
            run.run(self.spec(capacity_wait_hours=0.1))
        self.assertEqual(run.sleep.call_args_list, [mock.call(300), mock.call(60)])
        self.assertEqual([e["event"] for e in self.ledger.events()], ["queue", "queue_end"])
        self.assertEqual(self.ledger.events()[-1]["status"], "expired")
        self.assertEqual(self.ledger.waiting(), [])
        self.assertEqual(self.ledger.month_to_date()["total"], 0)
        self.assertEqual(self.api.created, [])

    def test_queued_run_checks_current_budget_before_reserving(self):
        self.api.pods["busy"] = {"id": "busy", "name": "busy", "status": "RUNNING", "cpu": {"vcpuCount": 64}}
        run = self.make(None)

        def sleep(seconds):
            self.api.pods.pop("busy")
            self.ledger.reserve("another-run", usd_est=150)
            self.clock.sleep(seconds)

        run.sleep = sleep
        with self.assertRaises(budget.BudgetRefused):
            run.run(self.spec(capacity_wait_hours=1))
        self.assertEqual([e["run_id"] for e in self.ledger.month_to_date()["open"]], ["another-run"])
        self.assertEqual(self.ledger.events()[-1]["status"], "failed")
        self.assertEqual(self.ledger.waiting(), [])
        self.assertEqual(self.api.created, [])

    def test_queued_run_rechecks_local_inputs_before_reserving(self):
        self.api.pods["busy"] = {"id": "busy", "name": "busy", "status": "RUNNING", "cpu": {"vcpuCount": 64}}
        run = self.make(None)

        def sleep(seconds):
            self.api.pods.pop("busy")
            (self.inputs / "lines.txt").write_text("changed invented input\n")
            self.clock.sleep(seconds)

        run.sleep = sleep
        with self.assertRaisesRegex(manifest.ManifestError, "changed since the manifest was written"):
            run.run(self.spec(capacity_wait_hours=1))
        self.assertEqual([e["event"] for e in self.ledger.events()], ["queue", "queue_end"])
        self.assertEqual(self.ledger.waiting(), [])
        self.assertEqual(self.api.created, [])

    def test_interrupted_wait_closes_local_queue_entry(self):
        self.api.pods["busy"] = {"id": "busy", "name": "busy", "status": "RUNNING", "cpu": {"vcpuCount": 64}}
        run = self.make(None)
        run.sleep = mock.Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            run.run(self.spec())
        self.assertEqual(self.ledger.events()[-1]["status"], "cancelled")
        self.assertEqual(self.ledger.waiting(), [])
        self.assertEqual(self.api.created, [])

    def test_deadman_armed_and_reported(self):
        remote = FakeRemote(dict(OUT))
        res = self.make(remote).run(self.spec())
        self.assertEqual(res["deadman"], "armed")
        self.assertIsNone(res["self_removal"])
        self.assertTrue(any("cat > /heocr/deadman.sh" in c for c in remote.commands))
        self.assertEqual(self.final()[0]["deadman"], "armed")

    def test_deadman_unavailable_is_reported(self):
        remote = FakeRemote(dict(OUT))
        remote.deadman = "unavailable: pod-key\n"
        res = self.make(remote).run(self.spec(prove_deadman=True))
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["deadman"], "unavailable: pod-key")
        self.assertIsNone(res["self_removal"])      # nothing to prove
        self.assertEqual(self.api.terminated, [res["pod_id"]])

    def test_smoke_proves_self_removal(self):
        remote = FakeRemote(dict(OUT))
        remote.on_fire = lambda: self.api.pods.clear()
        res = self.make(remote).run(self.spec(prove_deadman=True))
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["self_removal"], "works")

    def test_failed_self_removal_still_terminates(self):
        remote = FakeRemote(dict(OUT))
        res = self.make(remote).run(self.spec(prove_deadman=True))
        self.assertEqual(res["self_removal"], "failed")
        self.assertEqual(self.api.terminated, [res["pod_id"]])
        self.assertEqual(self.api.pods, {})

    def test_corrupted_transfer_is_fetched_again(self):
        remote = FakeRemote(dict(OUT), corrupt_once={"heb.traineddata"})
        res = self.make(remote).run(self.spec())
        self.assertEqual(res["status"], "ok", res)
        dest = self.results / res["run_id"]
        self.assertEqual((dest / "heb.traineddata").read_bytes(), OUT["heb.traineddata"])

    def test_dry_run_creates_nothing(self):
        res = self.make(FakeRemote(dict(OUT))).run(self.spec(dry_run=True))
        self.assertEqual(res["status"], "dry_run")
        self.assertEqual(self.api.created, [])
        self.assertEqual(self.ledger.events(), [])

    def test_budget_refusal_creates_nothing(self):
        with self.assertRaises(budget.BudgetRefused):
            self.make(FakeRemote(dict(OUT))).run(self.spec(profile=profiles.PROFILES["cpu"], hours=12))
        self.assertEqual(self.api.created, [])
        self.assertEqual(self.ledger.events(), [])

    def test_month_cap_counts_open_runs(self):
        self.ledger.reserve("x", pod_type="cpu3cx32", usd_est=149.99)
        with self.assertRaises(budget.BudgetRefused):
            self.make(FakeRemote(dict(OUT))).run(self.spec())
        self.assertEqual(self.api.created, [])

    def test_changed_input_refused_before_any_pod(self):
        (self.inputs / "lines.txt").write_text("changed after the manifest\n")
        with self.assertRaises(manifest.ManifestError):
            self.make(FakeRemote(dict(OUT))).run(self.spec())
        self.assertEqual(self.api.created, [])

    def test_error_terminates_pod(self):
        remote = FakeRemote(dict(OUT))
        remote.upload = mock.Mock(side_effect=SSHError("connection reset"))
        res = self.make(remote).run(self.spec())
        self.assertEqual(res["status"], "error")
        self.assertIn("connection reset", res["error"])
        self.assertEqual(self.api.terminated, [res["pod_id"]])
        self.assertEqual(self.final()[0]["status"], "error")

    def test_ctrl_c_terminates_pod(self):
        remote = FakeRemote(dict(OUT), fail_on="cat /heocr/exit_code")
        remote.fail_on_exc = Interrupted("SIGINT")
        res = self.make(remote).run(self.spec())
        self.assertEqual(res["status"], "interrupted")
        self.assertEqual(self.api.terminated, [res["pod_id"]])
        self.assertEqual(self.final()[0]["status"], "interrupted")

    def test_wall_time_limit(self):
        remote = FakeRemote(dict(OUT), exit_code="")  # the job never finishes
        res = self.make(remote).run(self.spec())
        self.assertEqual(res["status"], "timeout")
        self.assertEqual(self.api.terminated, [res["pod_id"]])
        self.assertIn("job.log", res["pulled"])  # logs still come back
        self.assertLessEqual(self.final()[0]["hours"], 0.25 + 0.01)

    def test_job_timeout_exit_code(self):
        res = self.make(FakeRemote(dict(OUT), exit_code="124")).run(self.spec())
        self.assertEqual(res["status"], "timeout")

    def test_job_failure(self):
        res = self.make(FakeRemote(dict(OUT), exit_code="3")).run(self.spec())
        self.assertEqual(res["status"], "job_failed")
        self.assertEqual(self.api.terminated, [res["pod_id"]])

    def test_price_above_quote_terminates_at_once(self):
        self.api.cost = 0.5
        res = self.make(FakeRemote(dict(OUT))).run(self.spec())
        self.assertEqual(res["status"], "error")
        self.assertIn("over the quote", res["error"])
        self.assertEqual(self.api.terminated, [res["pod_id"]])

    def test_failed_termination_keeps_reservation_open(self):
        self.api.terminate_works = False
        res = self.make(FakeRemote(dict(OUT))).run(self.spec())
        self.assertEqual(self.final(), [])
        self.assertGreater(self.ledger.month_to_date()["open_estimate"], 0)
        self.assertIn(res["pod_id"], self.api.pods)

    def test_lost_create_response_finds_the_pod(self):
        real = self.api.create_pod

        def create_then_fail(body):
            real(body)
            raise OSError("response lost")
        self.api.create_pod = create_then_fail
        res = self.make(FakeRemote(dict(OUT))).run(self.spec())
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(len(self.api.created), 1)      # found by name, never created twice
        self.assertEqual(self.api.terminated, ["pod1"])
        self.assertEqual(self.api.pods, {})

    def test_dropped_job_start_is_repeated(self):
        remote = FakeRemote(dict(OUT))
        remote.drop_starts = 1
        res = self.make(remote).run(self.spec())
        self.assertEqual(res["status"], "ok", res)
        starts = [c for c in remote.commands if "job.started" in c]
        self.assertEqual(len(starts), 2)
        self.assertIn("if [ ! -e /heocr/job.started ]", starts[0])   # never a second job

    def test_job_start_gives_up_after_tries(self):
        remote = FakeRemote(dict(OUT))
        remote.drop_starts = 99
        res = self.make(remote).run(self.spec())
        self.assertEqual(res["status"], "error")
        self.assertIn("could not be started", res["error"])
        self.assertEqual(self.api.terminated, [res["pod_id"]])

    def test_refused_create_is_tried_again(self):
        from heocr_compute.remote.api import RunPodError
        real, calls = self.api.create_pod, []

        def flaky(body):
            calls.append(1)
            if len(calls) <= 2:
                raise RunPodError("POST /v2/pods: HTTP 400 no capacity", 400)
            return real(body)
        self.api.create_pod = flaky
        res = self.make(FakeRemote(dict(OUT))).run(self.spec())
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(self.api.created), 1)

    def test_refused_create_that_made_a_pod_uses_it(self):
        from heocr_compute.remote.api import RunPodError
        real = self.api.create_pod

        def made_then_refused(body):
            real(body)
            raise RunPodError("POST /v2/pods: HTTP 500", 500)
        self.api.create_pod = made_then_refused
        res = self.make(FakeRemote(dict(OUT))).run(self.spec())
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(len(self.api.created), 1)
        self.assertEqual(res["pod_id"], "pod1")

    def test_auth_errors_are_not_retried(self):
        from heocr_compute.remote.api import RunPodError
        calls = []

        def forbidden(body):
            calls.append(1)
            raise RunPodError("POST /v2/pods: HTTP 403", 403)
        self.api.create_pod = forbidden
        res = self.make(FakeRemote(dict(OUT))).run(self.spec())
        self.assertEqual(res["status"], "error")
        self.assertEqual(len(calls), 1)

    def test_pull_size_cap(self):
        res = self.make(FakeRemote(dict(OUT))).run(self.spec(max_pull_gb=1e-9))
        self.assertEqual(res["status"], "error")
        self.assertIn("max-pull-gb", res["error"])
        self.assertTrue(self.api.terminated)


class SizedProfileTest(unittest.TestCase):
    def test_sized_cpu_profile(self):
        p = profiles.PROFILES["cpu"].sized("cpu5c", 16)
        self.assertEqual(p.pod_type, "cpu5cx16")
        self.assertEqual(p.body("heocr-cpu-x", {})["cpu"], {"id": "cpu5c", "vcpuCount": 16})
        self.assertEqual(profiles.PROFILES["cpu"].sized(vcpus=8).pod_type, "cpu3cx8")
        self.assertAlmostEqual(profiles.live_compute_price(FakeAPI(), p.sized("cpu3c", 16)), 0.03 * 16)

    def test_bad_sizes_refused(self):
        with self.assertRaises(ValueError):
            profiles.PROFILES["cpu"].sized(vcpus=12)
        with self.assertRaises(ValueError):
            profiles.PROFILES["gpu"].sized(vcpus=8)

    def test_cli_flags(self):
        from heocr_compute.remote import cli
        a = SimpleNamespace(profile="cpu", cpu_flavor="cpu5c", vcpus=16, community=False)
        self.assertEqual(cli._profile(a).pod_type, "cpu5cx16")
        a = SimpleNamespace(profile="cpu", cpu_flavor=None, vcpus=None, community=False)
        self.assertEqual(cli._profile(a).pod_type, "cpu3cx32")


class ReturnableTest(unittest.TestCase):
    def test_rules(self):
        r = runner.returnable
        self.assertIsNone(r("heb_checkpoint", ("**",), False))
        self.assertIsNone(r("best/heb.traineddata", ("**",), False))
        self.assertIsNone(r("eval/scores.tsv", ("**",), False))
        self.assertEqual(r("heb_0.512_120_9000.checkpoint", ("**",), False), "intermediate checkpoint")
        self.assertIsNone(r("heb_0.512_120_9000.checkpoint", ("**",), True))
        self.assertEqual(r("lines/page.png", ("**",), False), "not a checkpoint, log or score file")
        self.assertEqual(r("job.log", ("scores/*",), False), "not matched by --pull")


class SweepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.tmp.name) / "spend.jsonl", now=lambda: NOW)
        self.api = FakeAPI()

    def tearDown(self):
        self.tmp.cleanup()

    def pod(self, pid, name):
        self.api.pods[pid] = {"id": pid, "name": name, "cost": 0.96, "createdAt": "2026-11-18T10:00:00Z"}

    def test_sweep(self):
        import socket
        host = socket.gethostname()
        later = (NOW + dt.timedelta(hours=5)).isoformat()
        earlier = (NOW - dt.timedelta(minutes=1)).isoformat()
        rid = "20261118-100000-{}".format
        self.ledger.reserve(rid("aaaa"), pod_type="cpu3cx32", usd_est=12, host=host, pid=101, deadman_utc=later)
        self.ledger.final(rid("aaaa"), pod_type="cpu3cx32", hours=1, usd=1, status="ok")
        self.ledger.reserve(rid("bbbb"), pod_type="cpu3cx32", usd_est=12, host=host, pid=102, deadman_utc=later)
        self.ledger.reserve(rid("cccc"), pod_type="cpu3cx32", usd_est=12, host=host, pid=103, deadman_utc=later)
        self.ledger.reserve(rid("dddd"), pod_type="cpu3cx32", usd_est=12, host=host, pid=103, deadman_utc=earlier)
        self.ledger.reserve(rid("ffff"), pod_type="cpu3cx32", usd_est=12, host="other-mac", pid=102,
                            deadman_utc=later)
        self.pod("p1", "heocr-cpu-" + rid("aaaa"))   # its run finished: orphan
        self.pod("p2", "heocr-cpu-" + rid("bbbb"))   # its runner is gone
        self.pod("p3", "heocr-cpu-" + rid("cccc"))   # live run: kept
        self.pod("p4", "heocr-cpu-" + rid("dddd"))   # past its deadline
        self.pod("p5", "heocr-gpu-" + rid("eeee"))   # not in the ledger
        self.pod("p6", "heocr-cpu-" + rid("ffff"))   # another machine's live run: kept
        self.pod("p7", "someone-elses-pod")          # not ours: untouched
        killed = sweep(self.api, self.ledger, now=lambda: NOW, out=lambda *a: None, sleep=lambda s: None,
                       pid_alive=lambda pid: pid == 103)
        self.assertEqual(sorted(killed), ["p1", "p2", "p4", "p5"])
        self.assertEqual(sorted(self.api.pods), ["p3", "p6", "p7"])
        swept = {e["run_id"][-4:]: e for e in self.ledger.events() if e.get("status") == "swept"}
        self.assertEqual(sorted(swept), ["bbbb", "dddd"])   # finals written for open runs only
        self.assertAlmostEqual(swept["bbbb"]["hours"], 2.0)  # from the pod's createdAt
        self.assertAlmostEqual(swept["bbbb"]["usd"], 1.92)
        self.assertEqual(self.ledger.month_to_date()["open_estimate"], 24)  # cccc and ffff

    def test_sweep_all_and_dry_run(self):
        self.pod("p1", "heocr-cpu-20261118-100000-aaaa")
        self.pod("p2", "other")
        self.assertEqual(sweep(self.api, self.ledger, everything=True, dry_run=True, out=lambda *a: None), [])
        self.assertIn("p1", self.api.pods)
        self.assertEqual(sweep(self.api, self.ledger, everything=True, out=lambda *a: None,
                               sleep=lambda s: None), ["p1"])
        self.assertIn("p2", self.api.pods)


if __name__ == "__main__":
    unittest.main()
