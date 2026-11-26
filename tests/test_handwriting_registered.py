"""Exercise the registered one-shot runner using invented metadata and line counts."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


scripts = Path(__file__).resolve().parents[1] / "experiments" / "handwriting"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"handwriting_{name}", scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules, {"train": load_script("train")}):
    compare = load_script("compare")
    local_batch = load_script("local_batch")
    with patch.dict(sys.modules, {"local_batch": local_batch}):
        compose_local = load_script("compose_local")


def pin(body):
    return hashlib.sha256(body).hexdigest()


def put(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return compare.digest(path)


def roster(writers, retain):
    return {"schema": "heocr.handwriting_dev.v1", "split": "dev", "rows": [
        {"id": f"invented-{group}-{n}", "group": f"writer-{group}", "page": f"page-{group}",
         "layout": compare.LAYOUTS[group % 2], "line": n, "split": "dev", "retained": n <= retain,
         "image_sha256": "1" * 64, "label_sha256": "2" * 64}
        for group in range(writers) for n in range(1, 30)
    ]}


class RegisteredHandwritingTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.args = SimpleNamespace(actual_use=self.root / "permit.json", manifest=self.root / "dev.json",
                                    registration=42, candidate_sha256="a" * 64, output=self.root / "receipt.json")

    def stage(self, writers, retained, signed="signed"):
        self.args.manifest_sha256 = put(self.args.manifest, roster(writers, retained))
        self.args.actual_use_sha256 = put(self.args.actual_use, {
            "registration": 42, "actual_use": signed, "manifest_sha256": self.args.manifest_sha256,
            "candidate_sha256": self.args.candidate_sha256, "train_clean_sha256": "b" * 64,
            "terminal_sha256": "c" * 64,
        })

    def test_no_manifest_read_until_actual_use_binds(self):
        self.stage(4, 29, signed="pending")
        self.args.manifest.unlink()
        with self.assertRaisesRegex(ValueError, "actual-use receipt"):
            compare.score(self.args)
        self.stage(4, 29)
        self.args.candidate_sha256 = "d" * 64
        with self.assertRaisesRegex(ValueError, "actual-use receipt"):
            compare.score(self.args)

    def test_masks_set_support_and_low_support_never_opens_pairs(self):
        for writers, retained in ((3, 29), (4, 24)):
            with self.subTest(writers=writers, retained=retained):
                self.stage(writers, retained)
                with patch.object(compare, "read_cases", side_effect=AssertionError):
                    result = compare.score(self.args)
                self.assertEqual(result["verdict"], "hold")
                self.assertFalse(result["attempted"])
                self.assertEqual(result["lines"], writers * retained)
                self.assertEqual(result["masked_lines"], writers * (29 - retained))
                self.assertEqual(result["writers"], writers)

    def test_entire_registered_page_is_required_even_when_a_line_is_masked(self):
        manifest = roster(4, 28)
        manifest["rows"].pop()
        with self.assertRaisesRegex(ValueError, "all 29 lines"):
            compare.admitted_rows(manifest)

    def test_registered_fold_and_paired_blank_guard(self):
        translations = str.maketrans({"׳": "'", "־": "-", "”": '"'})
        self.assertEqual(compare.normalize("\u200f   קָם־א׳  ”א”\t", translations), "קָם-א' \"א\"")
        rows = [{"group": f"writer-{i}", "layout": compare.LAYOUTS[int(i == 3)], "chars": 10,
                 "edits": {"candidate": 0, "stock": 3}, "empty": {"candidate": False, "stock": False}}
                for i in range(4) for _ in range(25)]
        result = compare.summarize(rows)
        self.assertEqual(result["paired_95pct_pp"], [-30.0, -30.0])
        self.assertEqual(result["verdict"], "improved")
        self.assertEqual(result["by_layout"]["separate_page"], {"lines": 25, "writers": 1})
        self.assertNotIn("writer-", json.dumps(result))
        rows[0]["empty"]["candidate"] = True
        rows[0]["edits"]["candidate"] = 10
        guarded = compare.summarize(rows)
        self.assertLess(guarded["paired_95pct_pp"][1], 0)
        self.assertEqual(guarded["verdict"], "not improved")

    def test_bootstrap_resamples_whole_writer_clusters(self):
        rows = [{"group": f"writer-{writer}", "chars": 10,
                 "edits": {"candidate": 10 if writer < 2 else 0,
                           "stock": 0 if writer < 2 else 10}}
                for writer in range(4) for _ in range(25)]
        self.assertEqual(compare.paired(rows), [-100, 100])
        self.assertEqual(compare.paired(list(reversed(rows))), [-100, 100])

    def test_claim_blocks_retry_after_reader_failure(self):
        self.stage(4, 29)
        self.args.candidate = self.root / "candidate"
        self.args.tesseract = self.root / "tesseract"
        self.args.tesseract_sha256 = "f" * 64
        self.args.tessdata = self.root / "tessdata"
        self.args.fold = self.root / "fold.json"
        self.args.fold.write_text('{"folds": {}}', encoding="utf-8")
        with (patch.object(compare, "version", return_value="7.2"),
              patch.object(compare, "check", side_effect=lambda path, _: path),
              patch.object(compare, "read_cases", return_value=[]),
              patch.object(compare, "read_both", side_effect=RuntimeError("failed")) as reader):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                compare.score(self.args)
            with self.assertRaises(FileExistsError):
                compare.score(self.args)
            reader.assert_called_once()

    def test_local_batch_row_and_bytes_join(self):
        label, image = b"invented line\n", b"invented image"
        rows = self.root / "rows.jsonl"
        rows.write_text(json.dumps({"id": "line", "group": "writer", "role": "fit", "split": "train",
                                    "image_sha256": pin(image), "label_sha256": pin(label)}) + "\n")
        archive = self.root / "lines.tar"
        output = self.root / "staged.tar"
        with tarfile.open(archive, "w") as raw:
            for name, data in (("lines/line.png", image), ("lines/line.gt.txt", label)):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                raw.addfile(member, io.BytesIO(data))
        with tarfile.open(output, "w"):
            pass
        receipt = self.root / "clean.json"
        value = {"schema": "heocr.handwriting_train_batch.v1", "split": "train", "batch": "batch",
                 "verdict": "TRAIN CLEAN", "runpod_allowed": False, "partner": False,
                 "rows_sha256": compare.digest(rows), "archive_sha256": compare.digest(archive)}
        manifest = {"sources": [], "rows": []}
        expected = put(receipt, value)
        local_batch.append_batch(output, manifest, receipt, expected)
        self.assertEqual(manifest["rows"][0]["group"], "writer")
        self.assertFalse(manifest["sources"][0]["runpod_allowed"])
        with tarfile.open(output) as staged:
            self.assertEqual(staged.extractfile("lines/batch-line.gt.txt").read(), label)
        with self.assertRaisesRegex(ValueError, "distinct"):
            local_batch.append_batch(output, manifest, receipt, expected)
        fresh = {"sources": [], "rows": []}
        with self.assertRaisesRegex(ValueError, "roster or archive"):
            local_batch.append_batch(output, fresh, receipt, put(receipt, value | {"rows_sha256": "0" * 64}))

    def test_compose_pinned_index_preserves_tags_and_publishes_all_or_nothing(self):
        base = self.root / "base"
        base.mkdir()
        indexed = self.root / "approved"
        indexed.mkdir()
        unindexed = self.root / "unindexed"
        unindexed.mkdir()
        (unindexed / "clean.json").write_text("Unread", encoding="utf-8")
        label = b"invented label\n"

        def make_archive(directory, names):
            with tarfile.open(directory / "lines.tar", "w") as archive:
                for name, image in names.items():
                    for suffix, body in ((".png", image), (".gt.txt", label)):
                        item = tarfile.TarInfo(f"lines/{name}{suffix}")
                        item.size = len(body)
                        archive.addfile(item, io.BytesIO(body))
            return compare.digest(directory / "lines.tar")

        base_archive = make_archive(base, {"a": b"fit", "b": b"monitor"})
        base_sha = put(base / "manifest.json", {
            "schema": "heocr.handwriting_train.v1", "split": "train", "archive_sha256": base_archive,
            "sources": [{"id": "base", "clean_sha256": "a" * 64, "runpod_allowed": False, "partner": False}],
            "rows": [{"id": key, "group": f"group-{role}", "role": role, "split": "train", "source": "base",
                      "image_sha256": pin(role.encode()), "label_sha256": pin(label)}
                     for key, role in (("a", "fit"), ("b", "monitor"))],
        })
        archive_sha = make_archive(indexed, {"c": b"indexed"})
        rows = indexed / "rows.jsonl"
        rows.write_text(json.dumps({"id": "c", "group": "writer", "role": "fit", "split": "train",
                                   "image_sha256": pin(b"indexed"), "label_sha256": pin(label),
                                   "tags": ["label_unreviewed"]}) + "\n", encoding="utf-8")
        receipt = {"schema": "heocr.handwriting_train_batch.v1", "split": "train", "batch": "approved",
                   "verdict": "TRAIN CLEAN", "runpod_allowed": False, "partner": False,
                   "rows_sha256": compare.digest(rows), "archive_sha256": archive_sha}
        clean = indexed / "clean.json"
        index = self.root / "index.json"

        def stage_index(approved):
            return put(index, {"schema": "heocr.handwriting_train_index.v1", "split": "train", "batches": [
                {"id": "approved", "clean_sha256": put(clean, approved)},
            ]})

        output = self.root / "out"
        stage_index(receipt)
        with self.assertRaisesRegex(ValueError, "index SHA-256"):
            compose_local.compose(base, base_sha, index, "f" * 64, output)
        with self.assertRaisesRegex(ValueError, "distinct, Mac-only"):
            compose_local.compose(base, base_sha, index, stage_index(receipt | {"batch": "unindexed"}), output)
        self.assertFalse(output.exists())
        index_sha = stage_index(receipt)
        pins = compose_local.compose(base, base_sha, index, index_sha, output)
        manifest = json.loads((output / "manifest.json").read_bytes())
        self.assertEqual(pins["indexed_batches"], 1)
        self.assertEqual(pins["train_pairs"], 3)
        self.assertEqual(manifest["local_train_index_sha256"], index_sha)
        self.assertEqual(manifest["rows"][-1]["tags"], ["label_unreviewed"])
        self.assertEqual(compare.digest(base / "lines.tar"), base_archive)
        with tarfile.open(output / "lines.tar") as archive:
            self.assertEqual(archive.extractfile("lines/approved-c.gt.txt").read(), label)
        symlinked = self.root / "symlinked"
        symlinked.symlink_to(indexed, target_is_directory=True)
        bad_index_sha = put(index, {"schema": "heocr.handwriting_train_index.v1", "split": "train", "batches": [
            {"id": "symlinked", "clean_sha256": compare.digest(clean)},
        ]})
        with self.assertRaisesRegex(ValueError, "within its named directory"):
            local_batch.index_batches(index, bad_index_sha)
