"""Offline integrity checks for comparator assets."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("comparator_runner", Path(__file__).resolve().parents[1] / "comparators" / "runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ComparatorIntegrityTests(unittest.TestCase):
    def test_integrity_checks_reject_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weight"
            path.write_bytes(b"original")
            digest = hashlib.sha256(b"original").hexdigest()
            runner.verify(path, digest)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                runner.verify(path, digest)

    def test_each_model_has_pinned_assets(self):
        for model in runner.MODELS:
            assets = runner.selected(model)
            self.assertTrue(assets, model)
            for asset in assets:
                self.assertEqual(len(asset["sha256"]), 64)
                self.assertTrue(asset["url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
