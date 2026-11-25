"""Run a pinned table comparator on an ordered raster-only manifest."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fetch_models import verify


VERSIONS = {
    "paddle": {"paddleocr": "3.7.0", "paddlepaddle": "3.3.1", "paddlex": "3.7.0"},
    "tatr": {"transformers": "5.24.1", "timm": "1.0.31"},
}


def load_manifest(path, digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError("page manifest SHA-256 differs")
    data = json.loads(raw)
    if data["schema"] != "heocr.table_comparator_pages.v1" or data["dpi"] != 200:
        raise ValueError("expected the registered 200 DPI raster manifest")
    rows = data["pages"]
    if not rows or len({(r["id"], r["page"]) for r in rows}) != len(rows):
        raise ValueError("expected a nonempty, unique page roster")
    return rows


def open_page(row, root):
    from PIL import Image

    relative = Path(row["path"])
    path = (root / relative).resolve(strict=True)
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("image path escapes the raster root")
    if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("page raster SHA-256 differs")
    with Image.open(path) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("expected PNG or JPEG input")
        return path, image.convert("RGB")


def run(args):
    from grids import tesseract_reader

    expected = VERSIONS[args.lane]
    if any(version(name) != pin for name, pin in expected.items()):
        raise ValueError("comparator package versions differ from registration")
    names = json.loads((Path(__file__).parent / f"{args.lane}-models.json").read_bytes())
    verify(args.models, names)
    if args.lane == "tatr":
        from tatr import predictor
    else:
        from ppstructure import predictor
    predict = predictor(args.models, tesseract_reader(args.tessdata))
    records = {}
    issues = []
    for row in load_manifest(args.manifest, args.manifest_sha256):
        source, pixels = open_page(row, args.manifest.parent.resolve())
        tables, errs = predict(pixels, row["page"], source)
        records.setdefault(row["id"], []).extend(tables)
        issues.extend({"id": row["id"], "page": row["page"], "issue": issue} for issue in errs)
    if args.output.exists() or args.issues.exists():
        raise FileExistsError("use new output paths")
    with NamedTemporaryFile(dir=args.output.parent, mode="w", encoding="utf-8") as temp:
        for ident, tables in records.items():
            temp.write(json.dumps({"id": ident, "tables": tables}, ensure_ascii=False,
                                  separators=(",", ":"), sort_keys=True) + "\n")
        temp.flush()
        os.link(temp.name, args.output)
    with args.issues.open("x", encoding="utf-8") as f:
        f.write(json.dumps(issues, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=sorted(VERSIONS))
    for key in ("manifest", "models", "tessdata", "output", "issues"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
