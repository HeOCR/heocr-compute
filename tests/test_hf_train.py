"""Offline tests for autonomous training and its publication boundary."""
import contextlib
import io
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from heocr_compute.remote import hf_job, hf_train, hf_train_pod, profiles
from heocr_compute.remote.ledger import Ledger
from heocr_compute.remote.policy import Policy, Refused
from heocr_compute.remote.runner import sweep


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
        self.api.create_pod.return_value = {"id": "pod-example", "cost": 0.1}

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
        self.assertEqual([e["event"] for e in self.ledger.events()], ["reserve", "pod"])

    def test_registry_credential_is_passed_to_provider_for_private_image(self):
        self.args.registry_auth_id = "registry-fixture"
        with patch.object(Policy, "load", return_value=Policy()), contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger), 0)
        body, = self.api.create_pod.call_args.args
        self.assertEqual(body["registry"], "registry-fixture")
        self.assertNotIn("registry-fixture", stdout.getvalue())

    def test_dry_run_creates_nothing_and_denylist_blocks(self):
        self.args.dry_run = True
        with patch.object(Policy, "load", return_value=Policy()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger), 0)
        self.api.create_pod.assert_not_called()
        self.assertEqual(self.ledger.events(), [])
        denied = Policy(deny_hashes=frozenset([self.job["inputs"][0]["sha256"]]))
        with patch.object(Policy, "load", return_value=denied), self.assertRaises(Refused):
            hf_train.launch(self.args, profiles.PROFILES["smoke"], self.api, self.ledger)

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
        def commit(output, prefix, receipt, snapshot=None):
            commits.append((prefix, json.loads(json.dumps(receipt)), snapshot.read_bytes() if snapshot else None))
        hub.commit.side_effect = commit
        receipt = {"run_id": "20990101-000000-abcd", "status": "running", "snapshots": []}
        self.assertEqual(hf_train_pod.train(self.job, work, hub, receipt), "ok")
        self.assertEqual([c[1]["status"] for c in commits], ["running", "ok"])
        self.assertEqual(commits[0][2], b"invented")
        self.assertEqual(commits[0][1]["snapshots"][0]["sha256"], hf_job.digest(b"invented"))
        hub.private_revision.assert_any_call({"repo": "HeOCR/example-input", "type": "dataset"}, "1" * 40)
        hub.download.assert_called_once_with(self.job["inputs"][0], "trainer.sh", "1" * 40, self.source.stat().st_size)

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
        hub.download.side_effect = [receipt, self.source]
        self.api.list_pods.return_value = []
        self.api.get_pod.return_value = None
        self.api.billed_usd.return_value = 0.01
        args = SimpleNamespace(run_id=run_id, results=self.root / "collected")
        with patch.object(hf_train, "load_hf_token", return_value="invented"), patch.object(
                hf_job, "Hub", return_value=hub), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hf_train.collect(args, self.api, self.ledger), 0)
        self.assertEqual((args.results / run_id / "model.traineddata").read_bytes(), self.source.read_bytes())
        self.assertEqual(self.ledger.runs()[run_id]["final"]["usd"], 0.01)

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
