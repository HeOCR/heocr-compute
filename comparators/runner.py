#!/usr/bin/env python3
"""Fetch verified public assets and produce one UTF-8 text file per image."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

ASSETS = json.loads((Path(__file__).parent / "assets.json").read_text())
PADDLE_REVISION = "3baae91ad6479ef0473f901b4ed96366ae6c1670"
MODELS = ("best", "fast", "rivok-full", "rivok-lite", "paddle", "trocr-s44")


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def selected(model):
    return [asset for name, asset in ASSETS.items()
            if name == model or asset["path"].startswith(f"{model}/")]


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


def prepare(root, model, mode):
    if model == "trocr-s44" and mode != "line":
        raise ValueError("trocr-s44 requires a pre-cropped line image")
    assets = {asset["path"]: asset["sha256"] for asset in selected(model)}
    for path, digest in assets.items():
        verify(root / path, digest)

    provenance = {"model": model, "asset_sha256": assets}
    if model == "trocr-s44":
        from trocr import line_recognizer

        provenance["tool"] = {"name": "transformers", "version": metadata.version("transformers"),
                              "torch_version": metadata.version("torch")}
        provenance["settings"] = {
            "layout": None, "input_color": "RGB", "resize": "bilinear", "size": [384, 384],
            "scaling": "x/127.5-1", "threads": 1, "device": "cpu",
            "generation": "generation_config.json", "skip_special_tokens": True,
        }
        return line_recognizer(root / model), provenance
    if model == "paddle":
        from paddleocr_hebrew import HebrewOCR

        dist = metadata.distribution("paddleocr-hebrew")
        source = json.loads(dist.read_text("direct_url.json") or "{}")
        if source.get("vcs_info", {}).get("commit_id") != PADDLE_REVISION:
            raise RuntimeError("expected the pinned Rivok Paddle wrapper commit")

        options = {
            "det_thresh": 0.40, "det_unclip": 1.3, "det_max_side": 1280,
            "dilate_w": 0, "dilate_h": 0, "conf_threshold": 0.50,
            "split_long": 80, "gershayim": True, "geresh_yod": True,
        }
        engine = HebrewOCR.word(str(root / "paddle"), providers=["CPUExecutionProvider"], **options)
        provenance["tool"] = {"name": "paddleocr-hebrew", "version": dist.version,
                              "revision": PADDLE_REVISION, "onnxruntime_version": metadata.version("onnxruntime")}
        provenance["settings"] = {
            "pipeline": "word", "rec_mode": "cascade", "provider": "CPUExecutionProvider",
            "detector": "word-det/det.onnx", "order": "upstream RTL rows", "options": options,
        }

        def recognize(image):
            lines = engine.read(str(image), rec_mode="cascade")["lines"]
            return "\n".join(line["text"] for line in lines)

        return recognize, provenance

    executable = shutil.which("tesseract")
    if executable is None:
        raise FileNotFoundError("Tesseract executable was not found")
    version = subprocess.run([executable, "--version"], check=True, capture_output=True).stdout.decode("utf-8")
    if version.splitlines()[0] != "tesseract 5.5.3":
        raise RuntimeError("expected tesseract 5.5.3")
    language = {"best": "heb", "fast": "heb", "rivok-full": "heb_rivok", "rivok-lite": "heb_rivok_lite"}[model]
    psm = 3 if mode == "page" else 7
    settings = ["-l", language, "--oem", "1", "--psm", str(psm)]
    with Path(executable).open("rb") as binary:
        binary_sha = hashlib.file_digest(binary, "sha256").hexdigest()
    provenance["tool"] = {"name": "tesseract", "version": "5.5.3", "build": version,
                          "executable_sha256": binary_sha}
    provenance["settings"] = {"flags": settings, "tessdata": model}

    def recognize(image):
        cmd = [executable, str(image), "stdout", "--tessdata-dir", str(root / model), *settings]
        return subprocess.run(cmd, check=True, capture_output=True).stdout.decode("utf-8")

    return recognize, provenance


def run(root, model, image, mode):
    recognize, _ = prepare(root, model, mode)
    return recognize(image)


def main():
    from raw_output import add_batch_arguments, run_cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "run"])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--mode", choices=["page", "line"])
    parser.add_argument("--output", type=Path)
    add_batch_arguments(parser)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch(args.assets, args.model)
    else:
        adapter = Path(__file__)
        files = [adapter]
        if args.model == "trocr-s44":
            files.append(adapter.with_name("trocr.py"))
        suffix = "\n" if args.model in ("paddle", "trocr-s44") else ""
        run_cli(args, parser, prepare, files, legacy_suffix=suffix)


if __name__ == "__main__":
    main()
