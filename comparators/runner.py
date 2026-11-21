#!/usr/bin/env python3
"""Fetch verified public assets and produce one UTF-8 text file per image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

ASSETS = json.loads((Path(__file__).parent / "assets.json").read_text())
MODELS = ("best", "fast", "rivok-full", "rivok-lite", "paddle")


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def selected(model):
    return [a for name, a in ASSETS.items() if name == model or (model == "paddle" and name.startswith("paddle-"))]


def fetch(root, model):
    for asset in selected(model):
        path = root / asset["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            verify(path, asset["sha256"])
            continue
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".download-")
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(asset["url"], timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            verify(Path(temp), asset["sha256"])
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def run(root, model, image, mode):
    for asset in selected(model):
        verify(root / asset["path"], asset["sha256"])
    if model == "paddle":
        from paddleocr_hebrew import HebrewOCR
        result = HebrewOCR.word(str(root / "paddle")).read(str(image), rec_mode="cascade")
        return "\n".join(line["text"] for line in result["lines"]) + "\n"
    version = subprocess.run(["tesseract", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    if version != "tesseract 5.5.3":
        raise RuntimeError(f"expected tesseract 5.5.3, got {version}")
    language = {"best": "heb", "fast": "heb", "rivok-full": "heb_rivok", "rivok-lite": "heb_rivok_lite"}[model]
    cmd = ["tesseract", str(image), "stdout", "--tessdata-dir", str(root / model), "-l", language, "--oem", "1", "--psm", "3" if mode == "page" else "7"]
    return subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8").stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "run"])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--mode", choices=["page", "line"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch(args.assets, args.model)
    else:
        if not all((args.input, args.mode, args.output)):
            parser.error("run requires --input, --mode and --output")
        text = run(args.assets, args.model, args.input, args.mode)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
