"""Offline tests for autonomous training and its publication boundary."""
import contextlib
import datetime as dt
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode
from unittest.mock import MagicMock, Mock, patch

from heocr_compute.remote import cli, hf_job, hf_train, hf_train_pod, profiles, r2_input, r2_output
from heocr_compute.remote.api import RunPodError
from heocr_compute.remote.ledger import Ledger
from heocr_compute.remote.policy import Policy, Refused
from heocr_compute.remote.runner import Runner, sweep


class HfTrainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.write_bytes(b"printf invented > out/model.tmp\nmv out/model.tmp out/model.traineddata\nsleep 2\n")
        self.job = {
            "schema": hf_job.SCHEMA, "clean_sha256": "0" * 64, "image": "example/worker@sha256:" + "2" * 64,
            "inputs": [{"repo": "HeOCR/example-input", "type": "dataset", "revision": "1" * 40,
                        "path": "trainer.sh", "target": "train.sh",
                        "sha256": hf_job.digest(self.source.read_bytes()), "size": self.source.stat().st_size}],
            "entrypoint": "train.sh", "args": [],
            "snapshots": [{"name": "model.traineddata", "max_bytes": 100}],
            "output": {"repo": "HeOCR/example-output", "type": "model"},
        }
        self.jobfile = self.root / "spec.json"
        self.args = SimpleNamespace(job=self.jobfile, hf_secret="training_hub", hours=1, run_cap=1, month_cap=150,
                                    yes=True, dry_run=False)
        self.write_job()
        self.ledger = Ledger(self.root / "spend.jsonl", account="example")
        self.api = Mock()
        self.api.cpu_vcpu_range.return_value = (2, 32)
        self.api.cpu_price_per_vcpu.return_value = 0.05
        self.api.balance.return_value = 100
        self.api.list_pods.return_value = []
        self.api.create_pod.return_value = {"id": "pod-example", "cost": 0.1}
        self.api.pod_startup_state.return_value = "running"

    def write_job(self):
        raw = json.dumps(self.job).encode()
        self.jobfile.write_bytes(raw)
        self.args.approved_sha256 = hf_job.digest(raw)

    def test_refuses_revision_aliases_and_tampering(self):
        raw = self.jobfile.read_bytes()
        with self.assertRaisesRegex(ValueError, "approved SHA"):
            hf_job.parse(raw + b"\n", self.args.approved_sha256)
        self.job["inputs"][0]["revision"] = "main"
        self.write_job()
        with self.assertRaisesRegex(ValueError, "full revision"):
            hf_job.parse(self.jobfile.read_bytes(), self.args.approved_sha256)

    def test_only_spec_and_secret_reference_at_launch(self):
        with patch.object(Policy, "load", return_value=Policy()), contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger), 0)
        body, = self.api.create_pod.call_args.args
        self.assertEqual(body["env"]["HEOCR_HF_JOB"].encode(), self.jobfile.read_bytes())
        self.assertEqual(body["env"]["HEOCR_HF_TOKEN"], "{{ RUNPOD_SECRET_training_hub }}")
        self.assertFalse(body["startSsh"])
        self.assertEqual(body["ports"], [])
        self.assertNotIn("registry", body)
        self.assertNotIn("RUNPOD_SECRET", stdout.getvalue())
        self.assertIn("autonomous", self.ledger.events()[0])
        self.assertEqual(self.ledger.events()[0]["vcpus"], 2)
        self.assertEqual([e["event"] for e in self.ledger.events()], ["reserve", "pod", "startup"])
        self.api.pod_startup_state.assert_called_once_with("pod-example")
        self.api.terminate.assert_not_called()

    def r2_fixture(self):
        self.args.hf_repo_scope = "HeOCR/example-input"
        self.job["output"] = {"repo": self.args.hf_repo_scope, "type": "dataset"}
        self.job["schema"] = hf_job.TELEMETRY_SCHEMA
        self.job["telemetry"] = dict.fromkeys(
            ("metrics_max_bytes", "trainer_log_max_bytes", "run_log_max_bytes", "setup_log_max_bytes"), 100
        )
        pin = {"transport": "r2", "account": "1" * 32, "path": "train/exact.tar", "target": "exact.tar",
               "size": 3, "sha256": hf_job.digest(b"abc")}
        self.job["inputs"].append(pin)
        self.write_job()
        now = dt.datetime.now(dt.timezone.utc)
        qs = urlencode({
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": "fixture/" + now.strftime("%Y%m%d") + "/auto/s3/aws4_request",
            "X-Amz-Date": now.strftime("%Y%m%dT%H%M%SZ"),
            "X-Amz-Expires": "3600", "X-Amz-SignedHeaders": "host", "X-Amz-Signature": "2" * 64,
        })
        return pin, f"https://{pin['account']}.r2.cloudflarestorage.com/{r2_input.BUCKET}/{pin['path']}?{qs}"

    def test_r2_secrets_only_reach_pod_env_and_dry_run_creates_nothing(self):
        pin, url = self.r2_fixture()
        secret = json.dumps({pin["target"]: url})
        with patch.dict(os.environ, {r2_input.ENV: secret}), patch.object(
                Policy, "load", return_value=Policy()), contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.args.dry_run = True
            self.assertEqual(hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger), 0)
            self.api.create_pod.assert_not_called()
            self.args.dry_run = False
            self.assertEqual(hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger), 0)
        body, = self.api.create_pod.call_args.args
        self.assertEqual(json.loads(body["env"][r2_input.ENV]), {pin["target"]: url})
        for surface in (stdout.getvalue(), self.ledger.path.read_text(), self.jobfile.read_text()):
            self.assertNotIn("X-Amz-Signature", surface)

    def set_r2_output(self):
        self.job["output"] = {"transport": "r2", "account": "1" * 32}
        self.job["schema"] = hf_job.TELEMETRY_SCHEMA
        self.job["telemetry"] = dict.fromkeys(
            ("metrics_max_bytes", "trainer_log_max_bytes", "run_log_max_bytes", "setup_log_max_bytes"), 500)
        self.args.hf_repo_scope = self.job["inputs"][0]["repo"]
        self.args.require_worker_receipt = True
        self.write_job()

    def test_r2_output_launch_passes_puts_only_in_pod_env_and_uses_r2_startup(self):
        self.set_r2_output()
        envelope = "invented-put-secret"
        with (patch.object(Policy, "load", return_value=Policy()),
              patch.object(r2_output, "LocalR2") as local,
              patch.object(hf_train, "read_worker_startup", return_value="running") as startup,
              patch.object(hf_train, "load_hf_token", side_effect=AssertionError("using Hub output")),
              contextlib.redirect_stdout(io.StringIO()) as stdout):
            local.return_value.put_urls.return_value = envelope
            self.args.dry_run = True
            self.assertEqual(hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger), 0)
            self.api.create_pod.assert_not_called()
            self.args.dry_run = False
            self.assertEqual(hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger), 0)
        body, = self.api.create_pod.call_args.args
        self.assertEqual(body["env"][r2_output.ENV], envelope)
        self.assertEqual(startup.call_args.kwargs["r2"], local.return_value)
        self.assertEqual(local.return_value.put_urls.call_count, 2)
        for surface in (stdout.getvalue(), self.ledger.path.read_text(), self.jobfile.read_text()):
            self.assertNotIn(envelope, surface)

    def test_r2_worker_keeps_logs_after_checkpoint_put_failure(self):
        self.set_r2_output()
        self.source.write_text(
            'echo useful-trainer-output\n'
            'echo checkpoint > out/model.traineddata\n'
            'echo metrics > out/model.training-metrics.json\n'
            'echo lane-log > out/model.trainer.log\n'
        )
        self.job["inputs"][0].update(sha256=hf_job.digest(self.source.read_bytes()), size=self.source.stat().st_size)
        hub = Mock()
        hub.download.return_value = self.source
        publisher = Mock()
        commits = []

        def commit(receipt, artifacts=()):
            names = [path.name for path in artifacts]
            if "model.traineddata" in names:
                raise hf_job.PublishError("private R2 output PUT could not be confirmed")
            commits.append((receipt["status"], names, [path.read_bytes() for path in artifacts]))

        publisher.commit.side_effect = commit
        receipt = {"run_id": "20261123-001122-a123", "status": "running", "snapshots": []}
        work = self.root / "pod"
        work.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hf_train_pod.train(self.job, work, hub, receipt,
                                              publish_startup=True, r2_publisher=publisher), "failed")
        hub.commit.assert_not_called()
        self.assertEqual(commits[0], ("running", [], []))
        self.assertEqual(commits[1][:2], ("failed", ["training.log"]))
        self.assertIn(b"useful-trainer-output", commits[1][2][0])

    def add_public_archive(self):
        self.job["schema"] = hf_job.TELEMETRY_SCHEMA
        self.job["telemetry"] = {name: 100 for name in (
            "metrics_max_bytes", "trainer_log_max_bytes", "run_log_max_bytes", "setup_log_max_bytes")}
        self.args.hf_repo_scope = "HeOCR/example-input"
        self.job["output"] = {"repo": self.args.hf_repo_scope, "type": "dataset"}
        pin = {**self.job["inputs"][0], "repo": "HeOCR/example-public", "path": "open/train.tar",
               "target": "train.tar", "transport": hf_job.PUBLIC_TRAIN, "release_sha256": "a" * 64}
        self.job["inputs"].append(pin)
        self.write_job()
        return pin

    def test_public_train_needs_exact_allowlist_and_keeps_private_output(self):
        pin = self.add_public_archive()
        profile = profiles.PROFILES["smoke"]
        with self.assertRaisesRegex(ValueError, "public TRAIN dataset scope"):
            hf_train.launch(self.args, profile, self.api, self.ledger)
        self.api.create_pod.assert_not_called()
        self.args.hf_public_train_repo = [pin["repo"]]
        with patch.object(Policy, "load", return_value=Policy()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hf_train.launch(self.args, profile, self.api, self.ledger), 0)
        body, = self.api.create_pod.call_args.args
        self.assertEqual(body["env"]["HEOCR_HF_PUBLIC_TRAIN_REPOS"], pin["repo"])
        self.job["output"]["repo"] = pin["repo"]
        self.write_job()
        with self.assertRaisesRegex(ValueError, "approved Hub dataset scope"):
            hf_train.launch(self.args, profile, self.api, self.ledger)

    def test_public_train_can_only_be_a_release_bound_archive(self):
        pin = self.add_public_archive()
        for changes, message in [
            ({"path": "train.sh"}, "TRAIN archives"),
            ({"release_sha256": "unknown"}, "release receipt"),
            ({"type": "model"}, "release receipt"),
        ]:
            original = pin.copy()
            pin.update(changes)
            self.write_job()
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                hf_job.parse(self.jobfile.read_bytes(), self.args.approved_sha256)
            pin.update(original)

    def test_public_train_uses_anonymous_reads_and_rejects_gated_revision(self):
        pin = self.add_public_archive()
        hub = object.__new__(hf_job.Hub)
        hub.api = Mock()
        hub.token = "invented-private-token"
        hub.cache = self.root
        hub.get = Mock(return_value=self.source)
        hub.api.repo_info.return_value = SimpleNamespace(
            private=False, gated="manual", id=pin["repo"], sha=pin["revision"])
        with self.assertRaisesRegex(ValueError, "gated"):
            hub.public_train_revision(pin, pin["revision"])
        hub.api.repo_info.side_effect = [
            SimpleNamespace(private=True, id=self.args.hf_repo_scope, sha=pin["revision"]),
            SimpleNamespace(private=False, gated=False, id=pin["repo"], sha=pin["revision"]),
        ]
        hub.api.get_paths_info.side_effect = [
            [SimpleNamespace(path=item["path"], size=item["size"])] for item in self.job["inputs"]]
        out = self.root / "download"
        hf_train_pod.fetch_inputs(self.job, out, hub)
        self.assertEqual((out / pin["target"]).read_bytes(), self.source.read_bytes())
        self.assertIs(hub.api.repo_info.call_args.kwargs["token"], False)
        self.assertIs(hub.api.get_paths_info.call_args.kwargs["token"], False)
        self.assertIs(hub.get.call_args.kwargs["token"], False)
        self.assertEqual(hub.get.call_args_list[0].kwargs["token"], hub.token)

    def test_r2_wrong_object_expiry_and_job_secret_are_rejected(self):
        pin, url = self.r2_fixture()
        for changed in (url.replace("exact.tar", "other.tar"), url.replace("Expires=3600", "Expires=86401"),
                        url.replace(".r2.cloudflarestorage.com", ".example.com")):
            with self.assertRaisesRegex(ValueError, "mismatched presigned R2 GET") as exc:
                r2_input.read_urls(self.job, json.dumps({pin["target"]: changed}))
            self.assertNotIn("X-Amz-Signature", str(exc.exception))
        self.job["args"] = [url]
        self.write_job()
        with self.assertRaisesRegex(ValueError, "must not contain any presigned URLs"):
            hf_job.parse(self.jobfile.read_bytes(), self.args.approved_sha256)

    def test_r2_worker_verifies_bytes_and_consumes_secret(self):
        pin, url = self.r2_fixture()
        hub = Mock()
        hub.download.return_value = self.source
        opener = MagicMock()
        response = opener.open.return_value.__enter__.return_value
        response.status = 200
        response.headers = {"Content-Length": "3"}
        response.read.side_effect = [b"abc", b""]
        with patch.dict(os.environ, {r2_input.ENV: json.dumps({pin["target"]: url})}), patch.object(
                r2_input.urllib.request, "build_opener", return_value=opener):
            hf_train_pod.fetch_inputs(self.job, self.root, hub)
            self.assertNotIn(r2_input.ENV, os.environ)
        self.assertEqual((self.root / pin["target"]).read_bytes(), b"abc")
        hub.download.assert_called_once()
        response.read.side_effect = [b"def", b""]
        with patch.object(r2_input.urllib.request, "build_opener", return_value=opener), self.assertRaisesRegex(
                ValueError, "could not download and verify"):
            r2_input.download(pin, url, self.root / "changed.tar")
        self.assertFalse((self.root / "changed.tar").exists())

    def launch_with_clock(self, state):
        clock = dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc)

        def now():
            return clock

        def sleep(seconds):
            nonlocal clock
            clock += dt.timedelta(seconds=seconds)

        self.ledger.now = now
        self.args.startup_minutes = 0.25
        self.api.pod_startup_state.side_effect = state
        runner = Runner(self.api, self.ledger, now=now, sleep=sleep)
        with patch.object(hf_train, "Runner", return_value=runner), patch.object(
                Policy, "load", return_value=Policy()), contextlib.redirect_stdout(io.StringIO()):
            return hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger)

    def test_startup_timeout_terminates_exact_pod_and_waits_for_billing(self):
        self.api.get_pod.return_value = None
        self.api.billed_usd.return_value = None
        with self.assertRaisesRegex(ValueError, "terminated after its startup deadline"):
            self.launch_with_clock(["pending", "pending", "pending"])
        self.assertEqual(self.api.pod_startup_state.call_count, 3)
        self.api.terminate.assert_called_once_with("pod-example")
        run, = self.ledger.runs().values()
        self.assertEqual(run["reserve"]["startup_deadline_utc"], "2099-01-01T00:00:15+00:00")
        self.assertEqual(run["startup"]["date"], run["reserve"]["startup_deadline_utc"])
        self.assertEqual(run["startup"]["status"], "startup_timeout")
        self.assertNotIn("final", run)
        self.api.list_pods.return_value = []
        self.api.billed_usd.return_value = 0.013
        self.assertEqual(sweep(self.api, self.ledger, now=self.ledger.now, out=lambda _: None), [])
        final = self.ledger.runs()[run["run_id"]]["final"]
        self.assertEqual((final["status"], final["usd"]), ("startup_timeout", 0.013))

    def test_started_pod_is_kept_after_pending_and_unknown_observations(self):
        self.assertEqual(self.launch_with_clock(["pending", RunPodError("unavailable"), "running"]), 0)
        self.api.terminate.assert_not_called()
        run, = self.ledger.runs().values()
        self.assertEqual(run["startup"]["status"], "running")

    def test_startup_telemetry_error_is_not_evidence_to_delete(self):
        with self.assertRaisesRegex(ValueError, "startup could not be verified"):
            self.launch_with_clock(["pending", "pending", RunPodError("unavailable")])
        self.api.terminate.assert_not_called()
        run, = self.ledger.runs().values()
        self.assertNotIn("startup", run)

    def test_pod_that_exits_before_first_observation_can_still_be_collected(self):
        self.assertEqual(self.launch_with_clock(["gone"]), 0)
        self.api.terminate.assert_not_called()

    def require_worker_receipt(self):
        self.args.require_worker_receipt = True
        self.args.hf_repo_scope = "HeOCR/example-input"
        self.job["output"] = {"repo": self.args.hf_repo_scope, "type": "dataset"}
        self.job["schema"] = hf_job.TELEMETRY_SCHEMA
        self.job["telemetry"] = dict.fromkeys(
            ("metrics_max_bytes", "trainer_log_max_bytes", "run_log_max_bytes", "setup_log_max_bytes"), 500
        )
        self.write_job()

    def test_runtime_only_does_not_satisfy_worker_receipt_gate(self):
        self.require_worker_receipt()
        self.api.get_pod.return_value = None
        self.api.billed_usd.return_value = None
        with (patch.object(hf_train, "load_hf_token", return_value="invented"),
              patch.object(hf_job, "Hub"),
              patch.object(hf_train, "read_worker_startup", return_value=None) as receipt):
            with self.assertRaisesRegex(ValueError, "without a verified worker receipt"):
                self.launch_with_clock(["running", "running", "running"])
        self.assertEqual(receipt.call_count, 3)
        self.api.terminate.assert_called_once_with("pod-example")
        run, = self.ledger.runs().values()
        self.assertTrue(run["reserve"]["worker_receipt_required"])
        self.assertEqual(run["startup"]["status"], "startup_timeout")

    def test_launch_waits_for_matching_worker_receipt_and_passes_opt_in(self):
        self.require_worker_receipt()
        with (patch.object(hf_train, "load_hf_token", return_value="invented"),
              patch.object(hf_job, "Hub"),
              patch.object(hf_train, "read_worker_startup", side_effect=(None, "running")) as receipt):
            self.assertEqual(self.launch_with_clock(["running", "running"]), 0)
        self.assertEqual(receipt.call_count, 2)
        body, = self.api.create_pod.call_args.args
        self.assertEqual(body["env"]["HEOCR_HF_PUBLISH_STARTUP"], "1")
        self.assertNotIn("invented", json.dumps(body))
        self.api.terminate.assert_not_called()

    def test_worker_receipt_identity_is_bound_to_job_and_pod(self):
        self.require_worker_receipt()
        run_id = "20990101-000000-abcd"
        data = self.root / "receipt.json"
        receipt = {"schema": hf_job.TELEMETRY_RECEIPT, "run_id": run_id, "pod_id": "pod-example",
                   "job_sha256": self.args.approved_sha256, "clean_sha256": self.job["clean_sha256"],
                   "status": "running", "started_utc": "2099-01-01T00:00:00+00:00",
                   "finished_utc": None, "snapshots": []}
        data.write_text(json.dumps(receipt))
        with patch.object(hf_job, "Hub") as hub:
            hub.return_value.find_receipt.return_value = data
            reserve = {**self.job, "job_sha256": self.args.approved_sha256}
            self.assertEqual(hf_train.read_worker_startup("invented", run_id, "pod-example", reserve), "running")
            with self.assertRaisesRegex(ValueError, "does not match"):
                hf_train.read_worker_startup("invented", run_id, "different-pod", reserve)

    def test_worker_receipt_launch_rejects_r2_url_expiring_before_deadline(self):
        _, url = self.r2_fixture()
        self.args.require_worker_receipt = True
        self.args.hours = 2
        env = {r2_input.ENV: json.dumps({"exact.tar": url})}
        with (patch.dict(os.environ, env), patch.object(Policy, "load", return_value=Policy()),
              patch.object(hf_train, "load_hf_token", return_value="invented"), patch.object(hf_job, "Hub")):
            with self.assertRaisesRegex(ValueError, "expired or mismatched"):
                hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger)
        self.api.create_pod.assert_not_called()
        self.assertEqual(self.ledger.events(), [])

    def test_sweep_recovers_expired_startup_watch(self):
        now = dt.datetime(2099, 1, 1, 1, tzinfo=dt.timezone.utc)
        self.ledger.now = lambda: now
        run_id = "20990101-004000-abcd"
        self.ledger.reserve(run_id, autonomous=True, usd_est=0.5, pod_type="smoke",
                            deadman_utc="2099-01-01T02:00:00+00:00",
                            startup_deadline_utc="2099-01-01T00:55:00+00:00")
        self.ledger.pod(run_id, pod_id="pod-example", created_utc="2099-01-01T00:40:00+00:00")
        self.api.list_pods.return_value = [{"id": "pod-example", "name": "heocr-hf-" + run_id},
                                          {"id": "other-pod", "name": "heocr-hf-20990101-004000-abce"}]
        self.api.get_pod.return_value = None
        self.api.pod_startup_state.return_value = "pending"
        self.api.billed_usd.return_value = 0.02
        self.assertEqual(sweep(self.api, self.ledger, dry_run=True, now=lambda: now, out=lambda _: None), [])
        self.api.terminate.assert_not_called()
        self.assertEqual(sweep(self.api, self.ledger, now=lambda: now, out=lambda _: None), ["pod-example"])
        self.api.terminate.assert_called_once_with("pod-example")
        self.assertEqual(self.ledger.runs()[run_id]["final"]["usd"], 0.02)

    def test_sweep_rechecks_required_worker_receipt_after_launcher_exit(self):
        now = dt.datetime(2099, 1, 1, 1, tzinfo=dt.timezone.utc)
        self.ledger.now = lambda: now
        run_id = "20990101-004000-abcd"
        self.ledger.reserve(run_id, autonomous=True, usd_est=0.5, pod_type="smoke", worker_receipt_required=True,
                            deadman_utc="2099-01-01T02:00:00+00:00", startup_deadline_utc="2099-01-01T00:55:00+00:00")
        self.ledger.pod(run_id, pod_id="pod-example", created_utc="2099-01-01T00:40:00+00:00")
        self.api.list_pods.return_value = [{"id": "pod-example", "name": "heocr-hf-" + run_id}]
        self.api.pod_startup_state.return_value = "running"
        self.api.get_pod.return_value = None
        self.api.billed_usd.return_value = None
        with (patch("heocr_compute.remote.api.load_hf_token", return_value="invented"),
              patch.object(hf_train, "read_worker_startup", side_effect=[ValueError, None])):
            self.assertEqual(sweep(self.api, self.ledger, now=lambda: now, out=lambda _: None), [])
            self.api.terminate.assert_not_called()
            self.assertEqual(sweep(self.api, self.ledger, now=lambda: now, out=lambda _: None), ["pod-example"])
        self.api.terminate.assert_called_once_with("pod-example")

    def test_hub_publish_failure_exposes_http_status_without_response_body(self):
        hub = object.__new__(hf_job.Hub)
        hub.api = Mock()
        hub.private_revision = Mock(return_value="4" * 40)
        hub.add = Mock()
        error = RuntimeError("invented-sensitive-response")
        error.response = SimpleNamespace(status_code=403)
        hub.api.create_commit.side_effect = error
        with self.assertRaisesRegex(hf_job.PublishError, "HTTP 403") as caught:
            hub.commit(self.job["output"], "runs/example", {"run_id": "20990101-000000-abcd"})
        self.assertNotIn("invented-sensitive-response", str(caught.exception))

    def test_startup_minutes_cannot_exceed_quoted_allowance(self):
        self.args.startup_minutes = 16
        with self.assertRaisesRegex(ValueError, "startup allowance"):
            hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger)
        self.api.create_pod.assert_not_called()

    def test_registry_credential_is_passed_to_provider_for_private_image(self):
        self.args.registry_auth_id = "registry-fixture"
        with patch.object(Policy, "load", return_value=Policy()), contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger), 0)
        body, = self.api.create_pod.call_args.args
        self.assertEqual(body["registry"], "registry-fixture")
        self.assertNotIn("registry-fixture", stdout.getvalue())

    def test_kraken_launcher_requires_and_forwards_exact_dataset_scope(self):
        dataset = "HeOCR/example-train"
        self.job["image"] = hf_train.KRAKEN_TRAIN_IMAGE + "sha256:" + "2" * 64
        self.job["inputs"][0]["repo"] = dataset
        self.job["output"] = {"repo": dataset, "type": "dataset"}
        self.write_job()
        with self.assertRaisesRegex(ValueError, "--hf-repo-scope"):
            hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger)
        self.args.hf_repo_scope = "HeOCR/example-other"
        with self.assertRaisesRegex(ValueError, "outside its approved Hub dataset scope"):
            hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger)
        self.args.hf_repo_scope = dataset
        with self.assertRaisesRegex(ValueError, "CPU profile"):
            hf_train.launch(self.args, profiles.PROFILES["gpu"], self.api, self.ledger)
        self.assertEqual(self.api.mock_calls, [])
        self.assertEqual(self.ledger.events(), [])
        with patch.object(Policy, "load", return_value=Policy()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger), 0)
        body, = self.api.create_pod.call_args.args
        self.assertEqual(body["env"]["HEOCR_HF_REPO_SCOPE"], dataset)

    def test_worker_checks_scope_before_hub_access(self):
        dataset = "HeOCR/example-train"
        self.job["inputs"][0]["repo"] = dataset
        self.job["output"] = {"repo": dataset, "type": "dataset"}
        cases = (
            (None, "dataset", 1), ("HeOCR/example-other", "dataset", 1),
            (dataset, "model", 1), (dataset, "dataset", 0),
        )
        for scope, input_type, status in cases:
            with self.subTest(scope=scope, input_type=input_type):
                self.job["inputs"][0]["type"] = input_type
                self.write_job()
                env = {
                    "HEOCR_HF_REQUIRE_REPO_SCOPE": "1",
                    "HEOCR_HF_JOB": self.jobfile.read_text(),
                    "HEOCR_HF_JOB_SHA256": self.args.approved_sha256,
                    "HEOCR_HF_RUN_ID": "20990101-000000-abcd",
                    "RUNPOD_POD_ID": "pod-example",
                    "HEOCR_HF_TOKEN": "hf_fixture",
                }
                if scope is not None:
                    env["HEOCR_HF_REPO_SCOPE"] = scope
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.object(os, "umask"),
                    patch.object(hf_train_pod, "Path", return_value=self.root / "pod"),
                    patch.object(hf_train_pod, "train", return_value="ok"),
                    patch.object(hf_job, "Hub") as hub,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(hf_train_pod.main(), status)
                if status:
                    hub.assert_not_called()
                else:
                    hub.assert_called_once()

    def test_dry_run_creates_nothing_and_denylist_blocks(self):
        self.args.dry_run = True
        self.args.capacity_wait_hours = 1
        with patch.object(Policy, "load", return_value=Policy()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger), 0)
        self.api.create_pod.assert_not_called()
        self.api.list_pods.assert_not_called()
        self.assertEqual(self.ledger.events(), [])
        denied = Policy(deny_hashes=frozenset([self.job["inputs"][0]["sha256"]]))
        with patch.object(Policy, "load", return_value=denied), self.assertRaises(Refused):
            hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger)

    def test_cpu_admission_refuses_before_reserving_or_creating(self):
        self.api.list_pods.return_value = [
            {"id": "example-busy", "name": "other-workload", "status": "STARTING", "cpu": {"vcpuCount": 64}},
        ]
        with patch.object(cli, "_connect", return_value=(self.api, self.ledger)), \
                patch.object(Policy, "load", return_value=Policy()), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            rc = cli.main(["hf-train", "--profile", "smoke", "--job", str(self.jobfile), "--yes",
                           "--approved-sha256", self.args.approved_sha256, "--hf-secret", self.args.hf_secret,
                           "--hours", "1", "--run-cap", "1", "--no-wait-for-capacity"])
        self.assertEqual(rc, 2)
        self.assertIn("requested 2 vCPUs, 0 of 64 available", stderr.getvalue())
        self.api.create_pod.assert_not_called()
        self.api.terminate.assert_not_called()
        self.assertEqual(self.ledger.events(), [])

    def test_autonomous_cli_waits_then_starts_pod_and_wall_clock(self):
        before = dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc)
        after = before + dt.timedelta(minutes=5)
        elapsed = 0

        def now():
            return before + dt.timedelta(seconds=elapsed)

        self.ledger.now = now
        self.api.list_pods.side_effect = [
            [{"id": "busy", "name": "busy", "status": "PROVISIONING", "cpu": {"vcpuCount": 64}}], [],
        ]
        runner = Runner(self.api, self.ledger, now=now, clock=lambda: elapsed)

        def sleep(seconds):
            nonlocal elapsed
            self.assertEqual(seconds, 300)
            event, = self.ledger.waiting()
            self.assertEqual((event["vcpus"], event["wait_seconds"]), (2, 21_600))
            self.assertEqual(self.ledger.month_to_date()["total"], 0)
            self.api.create_pod.assert_not_called()
            elapsed += seconds

        runner.sleep = Mock(side_effect=sleep)
        with patch.object(cli, "_connect", return_value=(self.api, self.ledger)), \
                patch.object(hf_train, "Runner", return_value=runner), patch.object(Policy, "load", return_value=Policy()), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = cli.main(["hf-train", "--profile", "smoke", "--job", str(self.jobfile), "--yes",
                           "--approved-sha256", self.args.approved_sha256, "--hf-secret", self.args.hf_secret,
                           "--hours", "1", "--run-cap", "1"])
        self.assertEqual(rc, 0)
        runner.sleep.assert_called_once_with(300)
        self.api.create_pod.assert_called_once()
        self.assertEqual(self.ledger.waiting(), [])
        queued, reserve, _, _ = self.ledger.events()
        self.assertEqual((queued["event"], reserve["event"]), ("queue", "reserve"))
        self.assertEqual(reserve["startup_deadline_utc"], (after + dt.timedelta(minutes=15)).isoformat())
        self.assertEqual(reserve["deadman_utc"], (after + dt.timedelta(hours=1)).isoformat())

    def test_wait_flag_still_fails_immediately_on_bad_pod_identity(self):
        self.args.capacity_wait_hours = 1
        self.api.list_pods.return_value = [{"status": "RUNNING", "cpu": {"vcpuCount": 32}}]
        runner = Runner(self.api, self.ledger, sleep=Mock())
        with patch.object(hf_train, "Runner", return_value=runner), patch.object(Policy, "load", return_value=Policy()), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refused, "verifiable RunPod pod identity"):
            hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger)
        runner.sleep.assert_not_called()
        self.api.create_pod.assert_not_called()
        self.assertEqual(self.ledger.events(), [])

    def test_offline_runner_is_not_an_orphan(self):
        name = "heocr-hf-20990101-000000-abcd"
        self.api.list_pods.return_value = [{"id": "pod-example", "name": name}]
        self.assertEqual(sweep(self.api, self.ledger, pid_alive=lambda _: False, out=lambda _: None), [])
        self.ledger.reserve(name.removeprefix("heocr-hf-"), autonomous=True, usd_est=0.5,
                            deadman_utc="2099-01-01T00:00:00+00:00")
        self.assertEqual(sweep(self.api, self.ledger, pid_alive=lambda _: False, out=lambda _: None), [])
        self.api.terminate.assert_not_called()

    def test_publishes_snapshot_before_final_receipt(self):
        hub = Mock()
        hub.download.return_value = self.source
        work = self.root / "pod"
        work.mkdir()
        commits = []
        def commit(output, prefix, receipt, files=()):
            commits.append((prefix, json.loads(json.dumps(receipt)), [file.read_bytes() for file in files]))
        hub.commit.side_effect = commit
        receipt = {"run_id": "20990101-000000-abcd", "status": "running", "snapshots": []}
        self.assertEqual(hf_train_pod.train(self.job, work, hub, receipt), "ok")
        self.assertEqual([c[1]["status"] for c in commits], ["running", "ok"])
        self.assertEqual(commits[0][2], [b"invented"])
        self.assertEqual(commits[0][1]["snapshots"][0]["sha256"], hf_job.digest(b"invented"))
        hub.private_revision.assert_any_call({"repo": "HeOCR/example-input", "type": "dataset"}, "1" * 40)
        hub.download.assert_called_once_with(self.job["inputs"][0], "trainer.sh", "1" * 40, self.source.stat().st_size)

    def run_telemetry_job(self, script, *, use_r2=False):
        self.job["schema"] = hf_job.TELEMETRY_SCHEMA
        self.job["telemetry"] = {key: 500 for key in ("metrics_max_bytes", "trainer_log_max_bytes",
                                                    "run_log_max_bytes", "setup_log_max_bytes")}
        if use_r2:
            self.job["output"] = {"transport": "r2", "account": "a" * 32}
        self.source.write_text(script)
        self.job["inputs"][0].update(size=self.source.stat().st_size,
                                     sha256=hf_job.digest(self.source.read_bytes()))
        self.write_job()
        job = hf_job.parse(self.jobfile.read_bytes(), self.args.approved_sha256)
        work = self.root / "pod"
        work.mkdir()
        hub = Mock()
        hub.download.return_value = self.source
        commits = []

        def commit(output, prefix, receipt, files):
            commits.append((json.loads(json.dumps(receipt)), {file.name: file.read_bytes() for file in files}))

        hub.commit.side_effect = commit
        receipt = {"schema": hf_job.TELEMETRY_RECEIPT, "run_id": "20990101-000000-abcd",
                   "pod_id": "pod-example", "job_sha256": self.args.approved_sha256,
                   "clean_sha256": job["clean_sha256"], "started_utc": "2099-01-01T00:00:00+00:00",
                   "finished_utc": None, "status": "running", "snapshots": []}
        publisher = SimpleNamespace(commit=lambda receipt, files=(): commit(None, None, receipt, files)) if use_r2 else None
        status = hf_train_pod.train(job, work, hub, receipt, r2_publisher=publisher)
        hf_train.validate_receipt(receipt, receipt["run_id"], receipt["pod_id"], {
            **job, "job_sha256": self.args.approved_sha256,
        })
        return status, commits

    def test_telemetry_checkpoints_commit_with_companions_then_closed_logs(self):
        status, commits = self.run_telemetry_job(
            "echo 'entrypoint beginning'\n"
            "printf snapshot > out/model.traineddata\n"
            "printf '{}' > out/model.training-metrics.json\n"
            "printf phase > out/model.trainer.log\n"
            "printf setup > out/setup.log\n"
            "echo 'entrypoint finished'\n"
        )
        self.assertEqual(status, "ok")
        self.assertEqual([receipt["status"] for receipt, _ in commits], ["running", "ok"])
        self.assertEqual(set(commits[0][1]), {"model.traineddata", "model.training-metrics.json", "model.trainer.log"})
        final, logs = commits[1]
        self.assertEqual(set(logs), {"setup.log", "training.log"})
        self.assertTrue(logs["training.log"].endswith(b"entrypoint finished\n"))
        for entry in final["snapshots"]:
            data = (commits[0][1] | logs)[entry["name"]]
            self.assertEqual((entry["size"], entry["sha256"]), (len(data), hf_job.digest(data)))

    def test_r2_trainer_checkpoint_without_interval_progress_log_still_commits(self):
        self.job["snapshots"] = [{"name": name, "max_bytes": 500} for name in ("first.pt", "second.pt")]
        repo = Path(__file__).resolve().parents[1]
        rotate = f"PYTHONPATH={shlex.quote(str(repo))} {shlex.quote(sys.executable)} -B -m heocr_compute.remote.trainer_phase_log"
        status, commits = self.run_telemetry_job(
            "set -eu\n"
            "mkdir work\n"
            "printf 'TRAIN attempts 80000\\n' > work/phase.log\n"
            f"{rotate} work/phase.log out/first.trainer.log 0 85000\n"
            "printf '{}' > out/first.training-metrics.json\n"
            "printf first > out/first.pt\n"
            "test ! -e work/phase.log\n"
            f"{rotate} work/phase.log out/second.trainer.log 85000 170000\n"
            "printf '{}' > out/second.training-metrics.json\n"
            "printf second > out/second.pt\n",
            use_r2=True,
        )
        self.assertEqual(status, "ok")
        self.assertEqual([receipt["status"] for receipt, _ in commits], ["running", "running", "ok"])
        self.assertEqual(commits[0][1]["first.trainer.log"],
                         b"TRAIN attempts 80000\nCheckpoint visit window (0, 85000]\n")
        self.assertEqual(commits[1][1]["second.trainer.log"], b"Checkpoint visit window (85000, 170000]\n")
        self.assertEqual(set(commits[1][1]), {"second.pt", "second.training-metrics.json", "second.trainer.log"})
        self.assertEqual(set(commits[-1][1]), {"training.log"})

    def test_startup_receipt_precedes_inputs_and_failure_keeps_redacted_phase(self):
        self.require_worker_receipt()
        hub = Mock()
        commits = []

        def commit(output, prefix, receipt, files=()):
            commits.append((receipt["status"], {file.name: file.read_bytes() for file in files}))

        def refuse_input(*args):
            self.assertEqual(commits, [("running", {})])
            raise ValueError("invented-sensitive-response")

        hub.commit.side_effect = commit
        hub.download.side_effect = refuse_input
        root = self.root / "pod"
        root.mkdir()
        receipt = {"run_id": "20990101-000000-abcd", "status": "running", "snapshots": []}
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(hf_train_pod.train(self.job, root, hub, receipt, publish_startup=True), "failed")
        self.assertEqual([status for status, _ in commits], ["running", "failed"])
        log = commits[1][1]["training.log"]
        self.assertIn(b"Downloading a pinned private Hub input.", log)
        self.assertNotIn(b"invented-sensitive-response", log)
        self.assertNotIn("invented-sensitive-response", stdout.getvalue())

    def test_incomplete_checkpoint_is_not_uploaded_but_failure_logs_are(self):
        status, commits = self.run_telemetry_job(
            "printf snapshot > out/model.traineddata\n"
            "printf partial > out/partial.trainer.log\n"
            "echo 'entrypoint failed' >&2\nexit 3\n"
        )
        self.assertEqual(status, "failed")
        (receipt, files), = commits
        self.assertEqual(set(files), {"partial.trainer.log", "training.log"})
        self.assertEqual({entry["name"] for entry in receipt["snapshots"]}, set(files))

    def test_collector_rejects_incomplete_checkpoint_and_success_without_run_log(self):
        _, commits = self.run_telemetry_job(
            "printf snapshot > out/model.traineddata\n"
            "printf '{}' > out/model.training-metrics.json\n"
            "printf phase > out/model.trainer.log\n"
        )
        receipt = commits[1][0]
        reserve = {**self.job, "job_sha256": self.args.approved_sha256}
        for missing, error in [("model.trainer.log", "incomplete checkpoint group"),
                               ("training.log", "missing committed outputs")]:
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, error):
                broken = {**receipt, "snapshots": [e for e in receipt["snapshots"] if e["name"] != missing]}
                hf_train.validate_receipt(broken, receipt["run_id"], receipt["pod_id"], reserve)

    def test_mismatched_input_fails_before_training(self):
        hub = Mock()
        self.source.write_bytes(b"unapproved")
        hub.download.return_value = self.source
        work = self.root / "pod"
        work.mkdir()
        receipt = {"run_id": "20990101-000000-abcd", "status": "running", "snapshots": []}
        self.assertEqual(hf_train_pod.train(self.job, work, hub, receipt), "failed")
        self.assertFalse((work / "training.log").exists())
        self.assertEqual(receipt["snapshots"], [])

    def test_collect_rejects_modified_output(self):
        hub = Mock()
        hub.download.return_value = self.source
        result = self.root / "result-destination"
        receipt = {"snapshots": [{"name": "model.traineddata", "size": self.source.stat().st_size,
                                  "sha256": "e" * 64}]}
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            hf_train.materialize(hub, self.job["output"], "runs/example", "4" * 40, receipt, result, self.root)
        self.assertFalse(result.exists())

    def test_collect_recovers_run_when_create_response_was_lost(self):
        run_id = "20990101-000000-abcd"
        self.ledger.reserve(run_id, autonomous=True, usd_est=0.5, pod_type="smoke",
                            deadman_utc="2099-01-01T01:00:00+00:00", job_sha256=self.args.approved_sha256,
                            clean_sha256=self.job["clean_sha256"], snapshots=self.job["snapshots"],
                            output=self.job["output"])
        receipt = self.root / "hub-receipt.json"
        receipt.write_text(json.dumps({
            "schema": hf_job.RECEIPT, "run_id": run_id, "pod_id": "example",
            "job_sha256": self.args.approved_sha256, "clean_sha256": self.job["clean_sha256"],
            "started_utc": "2099-01-01T00:00:00+00:00", "finished_utc": "2099-01-01T00:01:00+00:00",
            "status": "ok", "snapshots": [{
                "name": "model.traineddata", "size": self.source.stat().st_size,
                "sha256": hf_job.digest(self.source.read_bytes()),
            }],
        }))
        hub = Mock()
        hub.private_revision.return_value = "4" * 40
        hub.find_receipt.return_value = receipt
        hub.download.return_value = self.source
        self.api.list_pods.return_value = []
        self.api.get_pod.return_value = None
        self.api.billed_usd.return_value = 0.01
        args = SimpleNamespace(run_id=run_id, results=self.root / "collected")
        with patch.object(hf_train, "load_hf_token", return_value="invented"), patch.object(
                hf_job, "Hub", return_value=hub), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hf_train.collect(args, self.api, self.ledger), 0)
        self.assertEqual((args.results / run_id / "model.traineddata").read_bytes(), self.source.read_bytes())
        self.assertEqual(self.ledger.runs()[run_id]["final"]["usd"], 0.01)

    def test_collect_reconciles_positive_bill_for_gone_pod_without_receipt(self):
        run_id = "20990101-000000-abcd"
        self.ledger.reserve(run_id, autonomous=True, usd_est=0.5, pod_type="smoke",
                            deadman_utc="2099-01-01T01:00:00+00:00", output=self.job["output"])
        self.ledger.pod(run_id, pod_id="pod-example", created_utc="2020-01-01T00:00:00Z")
        self.api.get_pod.return_value = None
        self.api.billed_usd.side_effect = (0.0, 0.11)
        hub = Mock()
        hub.private_revision.return_value = "4" * 40
        hub.find_receipt.return_value = None
        args = SimpleNamespace(run_id=run_id, results=self.root / "collected")
        with (patch.object(hf_train, "load_hf_token", return_value="invented"),
              patch.object(hf_job, "Hub", return_value=hub), contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(hf_train.collect(args, self.api, self.ledger), 1)
            self.assertNotIn("final", self.ledger.runs()[run_id])
            self.assertEqual(hf_train.collect(args, self.api, self.ledger), 1)
        final = self.ledger.runs()[run_id]["final"]
        self.assertEqual((final["status"], final["usd"]), ("missing_receipt", 0.11))
        self.assertEqual(final["hours_source"], "observation_or_deadline_bound")
        self.assertFalse((args.results / run_id).exists())

    def test_receipt_lookup_does_not_treat_network_error_as_missing(self):
        hub = object.__new__(hf_job.Hub)
        hub.api = Mock()
        hub.api.get_paths_info.side_effect = ([], OSError("invented-sensitive-response"))
        self.assertIsNone(hub.find_receipt(self.job["output"], "runs/example", "4" * 40))
        with self.assertRaisesRegex(ValueError, "could not check the private Hub receipt path") as caught:
            hub.find_receipt(self.job["output"], "runs/example", "4" * 40)
        self.assertNotIn("invented-sensitive-response", str(caught.exception))

    def test_expired_entrypoint_self_terminates_without_logging_provider_output(self):
        bindir = self.root / "bin"
        bindir.mkdir()
        for name, content in {
            "timeout": "shift 3\nexec \"$@\"\n",
            "runpodctl": "printf '%s\\n' \"$*\" > \"$MARKER\"\necho hidden-provider-output\n",
        }.items():
            script = bindir / name
            script.write_text("#!/bin/sh\n" + content)
            script.chmod(0o700)
        marker = self.root / "called"
        env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "MARKER": str(marker),
               "HEOCR_HF_DEADLINE": str(int(time.time()) - 1), "RUNPOD_POD_ID": "example",
               "RUNPOD_API_KEY": "invented"}
        entrypoint = Path(__file__).parents[1] / "docker" / "hf-train-entrypoint.sh"
        result = subprocess.run(["/bin/sh", str(entrypoint)], env=env, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(marker.read_text(), "pod delete example\n")
        self.assertNotIn(b"hidden-provider-output", result.stdout + result.stderr)

    def test_entrypoint_disables_inherited_hf_transfer(self):
        timeout = self.root / "timeout"
        timeout.write_text(
            '#!/bin/sh\nif [ "$1" = -s ]; then\n    printf "%s\\n" "$HF_HUB_ENABLE_HF_TRANSFER" > "$MARKER"\nfi\n'
        )
        timeout.chmod(0o700)
        marker = self.root / "transfer-mode"
        env = {**os.environ, "PATH": f"{self.root}:{os.environ['PATH']}", "MARKER": str(marker),
               "HEOCR_HF_DEADLINE": str(int(time.time()) + 30), "RUNPOD_POD_ID": "example",
               "RUNPOD_API_KEY": "invented", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
        entrypoint = Path(__file__).parents[1] / "docker" / "hf-train-entrypoint.sh"
        result = subprocess.run(["/bin/sh", str(entrypoint)], env=env, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "0\n")
