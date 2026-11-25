"""Offline tests for the handwriting experiment's staged data contract."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


TRAIN = Path(__file__).resolve().parents[1] / "experiments" / "handwriting" / "train.py"
loader = importlib.util.spec_from_file_location("handwriting_pilot_train", TRAIN)
trainer = importlib.util.module_from_spec(loader)
loader.loader.exec_module(trainer)


class HandwritingPilotTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.archive = self.root / "lines.tar"
        self.manifest = self.root / "manifest.json"
        self.destination = self.root / "extracted"
        self.destination.mkdir()
        self.entries = {}
        self.rows = []
        for role in ("fit", "monitor"):
            self.entries[f"lines/{role}.png"] = role.encode()
            self.entries[f"lines/{role}.gt.txt"] = b"invented line\n"
            self.rows.append({"id": role, "group": f"writer-{role}", "role": role, "split": "train", "source": "source",
                              "image_sha256": hashlib.sha256(role.encode()).hexdigest(),
                              "label_sha256": hashlib.sha256(b"invented line\n").hexdigest()})
        self.source = {"id": "source", "clean_sha256": "a" * 64, "runpod_allowed": True, "partner": False}

    def stage(self):
        with tarfile.open(self.archive, "w") as archive:
            for name, body in self.entries.items():
                record = tarfile.TarInfo(name)
                record.size = len(body)
                archive.addfile(record, io.BytesIO(body))
        self.manifest.write_text(json.dumps({"schema": "heocr.handwriting_train.v1", "split": "train",
                                            "archive_sha256": trainer.digest(self.archive),
                                            "sources": [self.source], "rows": self.rows}), encoding="utf-8")

    def prepare(self):
        return trainer.prepare(self.manifest, self.archive, self.destination)

    def test_exact_fit_and_monitor(self):
        self.stage()
        self.assertEqual(self.prepare(), {"fit": 1, "monitor": 1})
        for role in ("fit", "monitor"):
            image = self.destination / "lines" / f"{role}.png"
            self.assertEqual(image.read_bytes(), role.encode())
            self.assertEqual((self.destination / f"{role}.lst").read_text(), f"{image}\n")

    def test_wrong_split_fails_before_unpacking(self):
        self.rows[1]["split"] = "dev"
        self.stage()
        with self.assertRaisesRegex(ValueError, "invalid or duplicate TRAIN"):
            self.prepare()
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_shared_writer_fails_before_unpacking(self):
        self.rows[1]["group"] = self.rows[0]["group"]
        self.stage()
        with self.assertRaisesRegex(ValueError, "share a writer group"):
            self.prepare()

    def test_partner_or_local_only_source_is_refused(self):
        for flags in ({"partner": True, "runpod_allowed": True}, {"partner": False, "runpod_allowed": False}):
            with self.subTest(flags=flags):
                self.source.update(flags)
                self.stage()
                with self.assertRaisesRegex(ValueError, "separate CLEAN and non-partner RunPod clearance"):
                    self.prepare()

    def test_archive_rejects_extra_paths(self):
        self.entries["../escape.png"] = b"unexpected"
        self.stage()
        with self.assertRaisesRegex(ValueError, "undeclared"):
            self.prepare()
        self.assertFalse((self.root / "escape.png").exists())

    def test_archive_checks_declared_file_hash(self):
        self.entries["lines/fit.png"] = b"changed crop"
        self.stage()
        with self.assertRaisesRegex(ValueError, "line SHA-256"):
            self.prepare()
