"""Convert native visual-order cells to the Hebrew logical column convention."""

from dataclasses import dataclass, replace
from html.parser import HTMLParser
from math import ceil, floor
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path


@dataclass(frozen=True)
class Cell:
    row: int
    column: int
    rows: int = 1
    columns: int = 1
    header: bool = False
    text: str = ""


def table(cells, rows, columns, page, bbox):
    occupied = set()
    for cell in cells:
        r, c, rs, cs = cell.row, cell.column, cell.rows, cell.columns
        if not (0 <= r < r + rs <= rows and 0 <= c < c + cs <= columns):
            raise ValueError("cell span outside grid")
        covered = {(i, j) for i in range(r, r + rs) for j in range(c, c + cs)}
        if occupied & covered:
            raise ValueError("overlapping cells")
        occupied.update(covered)
    if not occupied or len(occupied) != rows * columns:
        raise ValueError("incomplete predicted grid")
    header_rows = 0
    for row in range(rows):
        if any(not c.header for c in cells if c.row <= row < c.row + c.rows):
            break
        header_rows += 1
    gold = [{"r": c.row, "c": columns - c.column - c.columns, "rowspan": c.rows,
             "colspan": c.columns, "text": c.text} for c in cells]
    return {"schema": "heocr_table.v1", "direction": "rtl", "rows": rows, "cols": columns,
            "header_rows": header_rows, "cells": sorted(gold, key=lambda c: (c["r"], c["c"])),
            "pages": [page, page], "bbox": bbox}


class PaddleHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.occupied = set()
        self.row = -1
        self.column = 0
        self.in_header = False

    def handle_starttag(self, tag, attrs):
        if tag == "thead":
            self.in_header = True
        elif tag == "tr":
            self.row += 1
            self.column = 0
        elif tag in {"td", "th"}:
            attrs = dict(attrs)
            rs = int(attrs.get("rowspan", 1))
            cs = int(attrs.get("colspan", 1))
            if self.row < 0 or not (1 <= rs <= 512 and 1 <= cs <= 512):
                raise ValueError("invalid Paddle HTML cell span")
            while (self.row, self.column) in self.occupied:
                self.column += 1
            self.cells.append(Cell(self.row, self.column, rs, cs, self.in_header or tag == "th"))
            self.occupied.update((r, c) for r in range(self.row, self.row + rs)
                                 for c in range(self.column, self.column + cs))
            self.column += cs

    def handle_endtag(self, tag):
        if tag == "thead":
            self.in_header = False


def paddle_table(html, boxes, image, page, bbox, read_cell):
    parser = PaddleHTML()
    parser.feed(html)
    parser.close()
    cells = parser.cells
    if not cells:
        raise ValueError("PP-Structure produced no structured cells")
    rows = parser.row + 1
    cols = max(cell.column + cell.columns for cell in cells)
    # The upstream table-cell geometry and token serialization can disagree.
    # Retain the structural grid when complete and flag the failed OCR join explicitly.
    text_join = len(boxes) == len(cells)
    if text_join:
        cells = [replace(cell, text=read_cell(image, box)) for cell, box in zip(cells, boxes, strict=True)]
    result = table(cells, rows, cols, page, bbox)
    return result, text_join


def pixel_box(box, size):
    l, t, r, b = box
    w, h = size
    return [max(0, floor(l)), max(0, floor(t)), min(w, ceil(r)), min(h, ceil(b))]


def tesseract_reader(tessdata):
    def recognize(image, box):
        l, t, r, b = pixel_box(box, image.size)
        if l >= r or t >= b:
            return ""
        with TemporaryDirectory(prefix="heb-cell-") as temp:
            cell = Path(temp) / "cell.png"
            image.crop((l, t, r, b)).save(cell, dpi=(200, 200))
            result = subprocess.run(["tesseract", str(cell), "stdout", "--tessdata-dir", str(tessdata),
                                     "-l", "heb", "--psm", "6"], check=True, capture_output=True)
        return result.stdout.decode("utf-8").rstrip("\r\n\f")

    return recognize
