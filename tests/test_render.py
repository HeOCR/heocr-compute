"""Render recipe and private publication tests. No test uses the network or a pod."""
import contextlib
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from heocr_compute.remote import api, cli, render, render_job
from heocr_compute.remote.ledger import Ledger
from heocr_compute.remote.policy import Policy
from heocr_compute.remote.runner import RunFailed
import test_runner as fixtures


class RecipeTest(unittest.TestCase):
    def test_selection_and_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "text"
            corpus.write_text("אבג דהו זחט יכל מנס עפצ\nשלום זהו משפט מומצא לבדיקה\nabc\n",
                              encoding="utf-8")
            selected = root / "selected"
            ids = render_job.select_lines(corpus, selected, 1, 9)
            before = selected.read_bytes()
            self.assertEqual(render_job.select_lines(corpus, selected, 1, 9), ids)
            self.assertEqual(selected.read_bytes(), before)
            self.assertEqual(len(ids), 1)
            with self.assertRaises(ValueError):
                render_job.select_lines(corpus, selected, 3, 9)
            args = render_job.render_args({"seed": 9, "fonts": ["font-a"], "mode": "future-mode"}, selected, root)
            self.assertEqual(args[-4:], ["--font", "font-a", "--mode", "future-mode"])

    def test_pack_hashes_every_output_and_rejects_missing_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dest = root / "rendered", root / "out"
            (src / "lines").mkdir(parents=True)
            dest.mkdir()
            (src / "summary.json").write_text(json.dumps({
                "lines_in": 1, "rendered": 1, "refused": [], "font_manifest_sha256": "f" * 64}))
            (src / "manifest.jsonl").write_text('{"id":"pby-1"}\n')
            (src / "lines/pby-1.png").write_bytes(b"invented png")
            with self.assertRaisesRegex(ValueError, "invalid render output"):
                render_job.pack(src, dest, ["pby-1"], {"font_manifest_sha256": "f" * 64})
            (src / "lines/pby-1.gt.txt").write_text("invented label")
            render_job.pack(src, dest, ["pby-1"], {"font_manifest_sha256": "f" * 64})
            sidecar = json.loads((dest / render_job.ARTIFACT_MANIFEST).read_text())
            archive = dest / render_job.ARCHIVE
            self.assertEqual(sidecar["archive"]["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            with tarfile.open(archive) as tar:
                self.assertEqual(set(tar.getnames()), {f["path"] for f in sidecar["files"]})
                for row in sidecar["files"]:
                    self.assertEqual(row["sha256"], hashlib.sha256(tar.extractfile(row["path"]).read()).hexdigest())


class PublisherTest(unittest.TestCase):
    dataset = "HeOCR/invented-render-tests"

    def setUp(self):
        self.publisher = render.HubPublisher.__new__(render.HubPublisher)
        self.publisher.api = mock.Mock()
        self.publisher.add = SimpleNamespace
        self.parent, self.commit = "a" * 40, "b" * 40
        self.publisher.api.dataset_info.side_effect = lambda dataset, *, revision, timeout: SimpleNamespace(
            private=True, id=dataset, sha=revision or self.parent)
        self.publisher.api.create_commit.return_value = SimpleNamespace(oid=self.commit)

    def test_private_commit_with_two_files_and_hash_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / render_job.ARCHIVE
            archive.write_bytes(b"invented archive")
            manifest = root / render_job.ARTIFACT_MANIFEST
            sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            job = {"example": "fixture"}
            manifest.write_text(json.dumps({
                "schema": render_job.ARTIFACT_SCHEMA, "job": job,
                "archive": {"path": archive.name, "size": archive.stat().st_size, "sha256": sha}}))
            prefix = "renders/run-test/"
            blob = f"blob {manifest.stat().st_size}\0".encode() + manifest.read_bytes()
            sidecar_entry = SimpleNamespace(path=prefix + manifest.name, size=manifest.stat().st_size,
                                            lfs=None, blob_id=hashlib.sha1(blob).hexdigest())
            archive_entry = SimpleNamespace(path=prefix + archive.name, size=archive.stat().st_size,
                                            lfs=SimpleNamespace(sha256=sha))
            self.publisher.api.get_paths_info.return_value = [archive_entry, sidecar_entry]
            url = self.publisher.publish(self.dataset, "run-test", root, job)
            self.assertTrue(url.endswith(f"/{self.commit}/renders/run-test"))
            kwargs = self.publisher.api.create_commit.call_args.kwargs
            self.assertEqual(kwargs["parent_commit"], self.parent)
            self.assertEqual({op.path_in_repo for op in kwargs["operations"]},
                             {prefix + archive.name, prefix + manifest.name})
            archive_entry.lfs.sha256 = "0" * 64
            with self.assertRaisesRegex(RunFailed, "readback failed"):
                self.publisher.publish(self.dataset, "run-test", root, job)

    def test_public_and_non_organization_destinations_fail_closed(self):
        self.publisher.api.dataset_info.side_effect = None
        self.publisher.api.dataset_info.return_value = SimpleNamespace(private=False, sha=self.parent, id=self.dataset)
        with self.assertRaises(RunFailed):
            self.publisher.check(self.dataset)
        with self.assertRaises(RunFailed):
            self.publisher.check("Other/invented-render-tests")
        self.publisher.api.create_commit.assert_not_called()

    def test_local_credential_ignores_generic_env_and_refuses_insecure_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secrets.toml"
            path.write_text('hf_token = "hf_invented"\n')
            env = {"HEOCR_RUNPOD_CONFIG": str(path), "HF_TOKEN": "hf_other"}
            path.chmod(0o644)
            with self.assertRaises(api.RunPodError):
                api.load_hf_token(env)
            path.chmod(0o600)
            self.assertEqual(api.load_hf_token(env), "hf_invented")


class RenderRunnerTest(unittest.TestCase):
    def test_public_dry_run_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.check_dry_run(Path(tmp))

    def check_dry_run(self, root: Path, *, show=False):
        fixture = fixtures.FakeAPI(balance=5.0)
        fixture.pods["orphan"] = {"id": "orphan", "name": "heocr-cpu-orphan"}
        ledger = Ledger(root / "spend.jsonl", now=lambda: fixtures.NOW)
        policy = Policy(roots={"synthetic": [root]})
        publisher = mock.Mock()
        font_manifest = b"invented font manifest"
        args = ["render-lines", "--count", "20000", "--seed", "19", "--dataset", PublisherTest.dataset,
                "--font-manifest-sha256", hashlib.sha256(font_manifest).hexdigest(),
                "--results", str(root), "--dry-run"]
        with mock.patch.object(cli, "_connect", return_value=(fixture, ledger)), \
                mock.patch.object(render.Policy, "load", return_value=policy), \
                mock.patch.object(render, "_read_public", side_effect=[
                    json.dumps({"private": False, "gated": False, "sha": render.REVISION,
                                "cardData": {"license": "cc0-1.0"}}).encode(),
                    json.dumps([{"path": render.FILE, "size": render.SOURCE["size"],
                                 "lfs": {"oid": render.SOURCE["sha256"]}}]).encode(),
                    font_manifest,
                ]), mock.patch.object(render, "HubPublisher", return_value=publisher), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(cli.main(args), 0)
        self.assertEqual(fixture.terminated, [])
        self.assertEqual(fixture.created, [])
        self.assertEqual(ledger.events(), [])
        publisher.preflight.assert_called_once_with(PublisherTest.dataset)
        publisher.publish.assert_not_called()
        self.assertIn("would create", stdout.getvalue())
        if show:
            print(stdout.getvalue())

    def test_rejects_mode_on_old_revision_and_excessive_count(self):
        args = ["render-lines", "--count", "20000", "--seed", "19", "--dataset", PublisherTest.dataset, "--dry-run"]
        with mock.patch("sys.stderr"):
            self.assertEqual(cli.main([*args, "--mode", "future-mode"]), 2)
            self.assertEqual(cli.main([*args, "--hours", "nan"]), 2)
            self.assertEqual(cli.main([*args, "--run-cap", "1.01"]), 2)

    def test_render_runner_teardown_and_bundle_handling(self):
        for exit_code, deadman, expected, pull in (
            ("0", "armed", "ok", True), ("0", "unavailable: pod-key", "error", False),
            ("124", "armed", "timeout", False), ("3", "armed", "job_failed", False),
            ("", "armed", "timeout", False),
        ):
            with self.subTest(exit_code=exit_code, deadman=deadman):
                case = fixtures.RunnerTest()
                case.setUp()
                try:
                    remote = fixtures.FakeRemote({render_job.ARCHIVE: b"invented bytes",
                                         render_job.ARTIFACT_MANIFEST: b"{}", "job.log": b"log"},
                                        exit_code=exit_code)
                    remote.deadman = deadman
                    remote.upload = lambda *a, **k: None
                    download = remote.download
                    remote.download = lambda *a, **k: download(*a)
                    result = case.make(remote).run(case.spec(hours=0.25, render_bundle=True))
                    self.assertEqual(result["status"], expected)
                    self.assertEqual(render_job.ARCHIVE in result["pulled"], pull)
                    self.assertEqual(case.api.terminated, [result["pod_id"]])
                finally:
                    case.tearDown()
