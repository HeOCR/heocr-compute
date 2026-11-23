"""Emit ordered Paddle CUDA page predictions from a pinned raster manifest."""
import argparse
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from raw_output import json_bytes, read_manifest, recognize_item, valid_sha256
from runner import prepare


def predict(manifest, manifest_sha256, assets, output):
    raw = manifest.read_bytes()
    if not valid_sha256(manifest_sha256) or hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("raster manifest differs from its pin")
    items = read_manifest(raw)
    root = manifest.parent.resolve()
    for item in items:
        path = Path(item["path"])
        if path.is_absolute() or not (root / path).resolve().is_relative_to(root):
            raise ValueError("raster path escapes the manifest directory")
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("expected a normalized PNG or JPEG raster")
    recognize, _ = prepare(assets, "paddle", "page", paddle_provider="CUDAExecutionProvider")
    with TemporaryDirectory(prefix=".paddle-cuda-", dir=output.parent) as tmp:
        staged = Path(tmp) / "predictions.jsonl"
        with staged.open("xb") as stream:
            for item in items:
                data, status, _ = recognize_item(item, root, recognize)
                if status == "error" or (status == "blank" and data.strip()):
                    raise ValueError("Paddle failed to produce a valid prediction")
                text = "" if status == "blank" else data.decode("utf-8")
                stream.write(json_bytes({"id": item["id"], "text": text}))
        # Publish only the complete ordered batch and refuse an existing output.
        os.link(staged, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    predict(args.manifest, args.manifest_sha256, args.assets, args.output)


if __name__ == "__main__":
    main()
