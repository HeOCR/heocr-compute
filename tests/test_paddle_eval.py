from contextlib import contextmanager
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from heocr_compute.remote import paddle_eval


def sha(data):
    return hashlib.sha256(data).hexdigest()


def append(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


def fixture(split="DEV"):
    return {"schema": "heocr.paddle_gpu_predictions.v1", "split": split,
            "image": "ghcr.io/heocr/heocr-compute-comparator-paddle-cuda@sha256:" + "a" * 64,
            "authority_sha256": "b" * 64, "dev_receipt_sha256": None, "manifest_sha256": "c" * 64, "count": 2,
            "input": {"account": "d" * 32, "bucket": "example-private", "path": "inference-runs/toy/input.tar",
                      "size": 10240, "sha256": "e" * 64},
            "output": {"account": "d" * 32, "bucket": "example-private", "path": "inference-runs/toy/predictions.jsonl",
                       "max_bytes": 1000}}


class PaddleEvalTest(unittest.TestCase):
    def test_sealed_scope_requires_dev_receipt_and_exact_approval(self):
        job = fixture("SEALED")
        raw = json.dumps(job).encode()
        with self.assertRaisesRegex(ValueError, "CUDA DEV receipt"):
            paddle_eval.parse(raw, sha(raw))
        job["dev_receipt_sha256"] = "f" * 64
        raw = json.dumps(job).encode()
        self.assertEqual(paddle_eval.parse(raw, sha(raw)), job)
        with self.assertRaisesRegex(ValueError, "approval pin"):
            paddle_eval.parse(raw, "a" * 64)

    def test_archive_accepts_only_the_exact_ordered_raster_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = [{"id": ident, "path": f"rasters/{ident}.png", "sha256": sha(payload)}
                        for ident, payload in (("r002", b"two"), ("r001", b"one"))]
            raw = json.dumps(manifest).encode()
            path = root / "fixture.tar"
            for extra in (None, "reference.txt", "rasters/r002.png"):
                with self.subTest(extra=extra):
                    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as out:
                        append(out, "manifest.json", raw)
                        append(out, "rasters/r001.png", b"one")
                        append(out, "rasters/r002.png", b"two")
                        if extra:
                            append(out, extra, b"invented extra")
                    if extra:
                        with self.assertRaises(ValueError):
                            paddle_eval.materialize(path, root, sha(raw), 2)
                    else:
                        self.assertEqual(paddle_eval.materialize(path, root, sha(raw), 2), ["r002", "r001"])
                        self.assertEqual((root / "rasters/r001.png").read_bytes(), b"one")

    def test_only_complete_predictions_are_uploaded(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "predictions.jsonl"
            with patch.object(paddle_eval.r2_output, "Publisher") as publisher:
                path.write_text('{"id":"r001","text":""}\n')
                with self.assertRaises(ValueError):
                    paddle_eval.publish(fixture(), root, ["r001", "r002"], "opaque PUT")
                publisher.assert_not_called()
                path.write_text('{"id":"r001","text":""}\n{"id":"r002","text":"fixture"}\n')
                paddle_eval.publish(fixture(), root, ["r001", "r002"], "opaque PUT")
                self.assertEqual(publisher.return_value.put.call_args.args[0], "predictions.jsonl")

    def test_dry_run_does_not_reserve_or_create(self):
        with TemporaryDirectory() as tmp, patch.object(paddle_eval, "_connect") as connect, \
                patch.object(paddle_eval, "Runner") as runner, patch.dict(paddle_eval.os.environ, {}, clear=True):
            path = Path(tmp) / "job.json"
            raw = json.dumps(fixture()).encode()
            path.write_bytes(raw)
            api, ledger = Mock(), Mock()
            connect.return_value = (api, ledger)

            @contextmanager
            def slot(spec, run_id):
                self.assertTrue(spec.dry_run)
                yield SimpleNamespace(hours=0.35), False

            runner.return_value.now.return_value = dt.datetime(2026, 11, 23, tzinfo=dt.timezone.utc)
            runner.return_value.launch_slot.side_effect = slot
            args = SimpleNamespace(job=path, approved_sha256=sha(raw), run_cap=0.32, launch=False)
            self.assertEqual(paddle_eval.launch(args), 0)
            ledger.reserve.assert_not_called()
            api.create_pod.assert_not_called()
