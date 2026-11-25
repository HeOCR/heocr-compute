import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comparators/tables"))
from grids import Cell, paddle_table, table
from worker import load_manifest


class TestTableComparators(unittest.TestCase):
    def test_mirror_merged_cell_once(self):
        found = table([Cell(0, 0, columns=2, header=True, text="כותרת"), Cell(0, 2, header=True)],
                      1, 3, 2, [0, 0, 100, 100])
        self.assertEqual(found["header_rows"], 1)
        self.assertEqual(found["pages"], [2, 2])
        merged = found["cells"][1]
        self.assertEqual((merged["c"], merged["colspan"], merged["text"]), (1, 2, "כותרת"))

    def test_reject_grid_with_hole(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            table([Cell(0, 0)], 2, 2, 1, None)

    def test_map_paddle_rowspans(self):
        html = "<table><tr><td rowspan='2'></td><td></td></tr><tr><td></td></tr></table>"
        found, paired = paddle_table(html, [0, 1, 2], None, 1, None, lambda _, box: str(box))
        self.assertTrue(paired)
        self.assertEqual([(cell["r"], cell["c"], cell["rowspan"], cell["text"])
                          for cell in found["cells"]], [(0, 0, 1, "1"), (0, 1, 2, "0"), (1, 0, 1, "2")])

    def test_report_geometry_mismatch(self):
        html = "<table><tr><td></td><td></td></tr></table>"
        found, paired = paddle_table(html, [0], None, 1, None, lambda *_: self.fail("guessed cell"))
        self.assertFalse(paired)
        self.assertEqual([cell["text"] for cell in found["cells"]], ["", ""])

    def test_require_pinned_unique_page_roster(self):
        page = {"id": "doc", "page": 1, "path": "page.png", "sha256": "0" * 64}
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "pages.json"
            data = {"schema": "heocr.table_comparator_pages.v1", "dpi": 200, "pages": [page]}
            raw = json.dumps(data).encode()
            manifest.write_bytes(raw)
            self.assertEqual(load_manifest(manifest, hashlib.sha256(raw).hexdigest()), [page])
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                load_manifest(manifest, "1" * 64)
            data["pages"].append(page)
            raw = json.dumps(data).encode()
            manifest.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_manifest(manifest, hashlib.sha256(raw).hexdigest())
