"""Offline checks of the raw scorer transport with invented readers and IDs."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from comparators.raw_output import write_batch


class RawComparatorOutputTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.args = SimpleNamespace(
            manifest=self.root / "manifest.json",
            output_dir=self.root / "output",
            runtime_sha256="1" * 64,
            config_sha256="2" * 64,
            mode="page",
        )
        self.provenance = {"model": "invented", "settings": {"segmentation": "invented"}}

    def item(self, name, input_id=None):
        (self.root / name).write_bytes(name.encode("utf-8"))
        return {"id": input_id or name, "path": name, "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest()}

    def run_batch(self, items, reader):
        self.args.manifest.write_text(json.dumps(items), encoding="utf-8")
        write_batch(self.args, reader, self.provenance)
        return json.loads((self.args.output_dir / "run.json").read_bytes())

    def test_page_bytes_keep_marks_spacing_order_controls_and_breaks(self):
        entries = [self.item("first", "z/7"), self.item("second", "a/7")]
        emitted = "  א\u05b8\u05bc!\u200f\r\n\r\nfoo\t  \x0cbar\n"
        expected = "  א\u05b8\u05bc!\u200f\n\nfoo\t  \x0cbar\n".encode("utf-8")
        sidecar = self.run_batch(entries, lambda path: emitted if path.name == "first" else " \n")
        rows = sidecar["results"]
        self.assertEqual([(r["id"], r["status"]) for r in rows], [("z/7", "ok"), ("a/7", "blank")])
        self.assertEqual((self.args.output_dir / rows[0]["file"]).read_bytes(), expected)
        self.assertEqual((self.args.output_dir / rows[1]["file"]).read_bytes(), b" \n")
        self.assertEqual(rows[0]["text_sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(sidecar["runtime_sha256"], self.args.runtime_sha256)
        self.assertEqual(sidecar["config_sha256"], self.args.config_sha256)

    def test_line_errors_keep_expected_ids_and_extras(self):
        self.args.mode = "line"
        names = ("extra", "blank", "missing", "wrong-hash", "bad-text", "crash")
        entries = [self.item(name) for name in names]
        (self.root / "missing").unlink()
        entries[3]["sha256"] = "0" * 64

        def reader(path):
            if path.name == "crash":
                raise RuntimeError("scratch path or stderr are not output")
            return {"extra": " א\r\nב \n", "blank": "", "bad-text": "\ud801"}[path.name]

        sidecar = self.run_batch(entries, reader)
        with (self.args.output_dir / "lines.jsonl").open("rb") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual([r["id"] for r in rows], list(names))
        self.assertEqual(rows[0], {"id": "extra", "text": " א\nב \n"})
        self.assertTrue(all(r == {"id": name, "text": ""} for name, r in zip(names[1:], rows[1:])))
        status = sidecar["results"]
        self.assertTrue(status[0]["unexpected_line_break"])
        self.assertEqual([r["status"] for r in status], ["ok", "blank", "error", "error", "error", "error"])
        self.assertEqual(status[3]["error"], "input_sha256_mismatch")
        self.assertNotIn("scratch path", json.dumps(sidecar))

    def test_duplicate_ids_fail_before_publishing_any_results(self):
        entries = [self.item("a", "same"), self.item("b", "same")]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.run_batch(entries, lambda _: self.fail("reader was called"))
        self.assertFalse(self.args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
