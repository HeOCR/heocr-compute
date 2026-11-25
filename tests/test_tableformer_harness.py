import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

HARNESS = Path(__file__).resolve().parents[1] / "experiments/tableformer"
sys.path.insert(0, str(HARNESS))
try:
    from baseline import docling_tables, read_asset, structure_only
    from targets import otsl
    from train import load_manifest
finally:
    sys.path.remove(str(HARNESS))


def gold():
    return {"schema": "heocr_table.v1", "rows": 2, "cols": 3, "direction": "rtl", "header_rows": 0,
            "cells": [{"r": 0, "c": 0, "rowspan": 2, "colspan": 2, "text": "field"},
                      {"r": 0, "c": 2, "rowspan": 1, "colspan": 1, "text": ""},
                      {"r": 1, "c": 2, "rowspan": 1, "colspan": 1, "text": "value"}]}


class Targets(unittest.TestCase):
    def test_visual_coordinates_and_both_spans(self):
        self.assertEqual(otsl(gold()), ["ecel", "fcel", "lcel", "nl", "fcel", "ucel", "xcel", "nl"])
        self.assertEqual(otsl({**gold(), "header_rows": 1})[:4], ["ched", "ched", "lcel", "nl"])

    def test_fragment_scope_and_coverage_are_explicit(self):
        for overrides in ({"claude_drafted": True}, {"dev_only": True}, {"pages": [1, 2]},
                          {"blocks_meta": {"snake": True}}, {"cells": gold()["cells"][:-1]}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                otsl({**gold(), **overrides})
        for key, value in (("mask", True), ("c", 1)):
            example = gold()
            example["cells"][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                otsl(example)
        with self.assertRaisesRegex(ValueError, "position limit"):
            otsl(gold(), max_tokens=8)

    def test_docling_headers_and_span_mirror(self):
        cells = [SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=0,
                                 end_row_offset_idx=1, end_col_offset_idx=2, column_header=True),
                 SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=2,
                                 end_row_offset_idx=1, end_col_offset_idx=3, column_header=True)]
        document = SimpleNamespace(tables=[SimpleNamespace(
            data=SimpleNamespace(num_rows=1, num_cols=3, table_cells=cells))])
        result, = docling_tables(document, rtl=True)
        self.assertEqual(result["header_rows"], 1)
        self.assertEqual({(x["c"], x["colspan"]) for x in result["cells"]}, {(1, 2), (0, 1)})
        self.assertEqual(structure_only(gold())["cells"][0]["text"], "")


class TrainIntake(unittest.TestCase):
    def asset(self, root, name, content):
        path = root / name
        path.write_bytes(content)
        return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}

    def test_only_exact_signed_train_manifest(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            img = self.asset(root, "i.png", b"fixture image bytes")
            table = self.asset(root, "g.json", json.dumps(gold()).encode())
            entry = {"id": "fixture", "source_id": "source", "split": "train", "image": img, "gold": table}
            manifest = {"schema": "heocr.tableformer_train.v1", "split": "train", "examples": [entry]}

            def bind(doc):
                signed = self.asset(root, "manifest.json", json.dumps(doc).encode())
                clean = {"split": "train", "verdict": "TRAIN CLEAN", "files": {"manifest.json": signed}}
                receipt = self.asset(root, "clean.json", json.dumps(clean).encode())
                return SimpleNamespace(root=root, manifest=root / signed["path"], manifest_sha256=signed["sha256"],
                                       clean=root / receipt["path"], clean_sha256=receipt["sha256"])

            self.assertEqual(load_manifest(bind(manifest)), [(b"fixture image bytes", otsl(gold()))])
            with self.assertRaisesRegex(ValueError, "outside TRAIN"):
                load_manifest(bind({**manifest, "examples": [{**entry, "split": "dev"}]}))
            args = bind(manifest)
            args.manifest_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest SHA"):
                load_manifest(args)

    def test_assets_cannot_leave_the_declared_scope(self):
        with TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            root = base / "packet"
            root.mkdir()
            outside = self.asset(base, "source", b"fixture asset")
            pointer = root / "link"
            pointer.symlink_to(base / outside["path"])
            with self.assertRaisesRegex(ValueError, "packet root"):
                read_asset(root, {"path": "link", "sha256": outside["sha256"]})
