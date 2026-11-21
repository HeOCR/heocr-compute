#!/usr/bin/env python3
"""Run a hash-verified Kraken recognizer on one page or line image."""
import argparse
import hashlib
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import tempfile

MODELS = {
    "ppocrv6-small": ("small.safetensors", "3108259779f5abfed3908a1c9eafba64d17e4a2a169368c539ecfeff04c018f3"),
    "biblia-general": ("BiblIA_01.mlmodel", "bb48c481d8c46b41465920482e253dc990163405c584d24700cfb0d9b4ca9147"),
    "biblia-ashkenazi": ("Ashkenazi_01.mlmodel", "38bd8cce2480089eb5d1ae3b37e7930d6565858249185626f69a13f6ec85d16e"),
    "biblia-italian": ("Italian_01.mlmodel", "7a396f6363503285cf7ec02c543a08ee9faf8bc591851efbaf04217089e76e14"),
    "biblia-sephardi": ("Sephardi_01.mlmodel", "1ab5ef193001f336e886f3335c624a2454c1ef0a1dfbe87341dcba4bd59c5108"),
}
SEGMENTER_SHA = "a3ccc7309fe8df62d1cbd372706edc73db6a2a32806e3758522ed8528a734073"


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def prepare(root, model, mode):
    if metadata.version("kraken") != "7.2":
        raise RuntimeError("kraken 7.2 is required")
    import kraken
    model_name, digest = MODELS[model]
    weights = root / model_name
    verify(weights, digest)
    assets = {model_name: digest}
    segmenter = Path(kraken.__file__).parent / "blla.mlmodel"
    segment = []
    if mode == "page":
        verify(segmenter, SEGMENTER_SHA)
        assets["blla.mlmodel"] = SEGMENTER_SHA
        segment = ["segment", "--baseline", "--model", str(segmenter), "--text-direction", "horizontal-rl"]
    ocr = ["ocr", "--model", str(weights), "--num-line-workers", "0", "--reorder", "--base-dir", "R"]
    if mode == "line":
        ocr += ["--no-segmentation"]
    global_flags = ["--device", "cpu", "--threads", "1", "--raise-on-error"]
    executable = Path(sys.executable).parent / "kraken"

    def recognize(image):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output.txt"
            cmd = [str(executable), *global_flags, "-i", str(image), str(output), *segment, *ocr]
            subprocess.run(cmd, check=True, capture_output=True)
            return output.read_bytes().decode("utf-8")

    provenance = {
        "model": model,
        "tool": {"name": "kraken", "version": "7.2", "torch_version": metadata.version("torch")},
        "asset_sha256": assets,
        "settings": {
            "global_flags": global_flags,
            "segmentation": ["segment", "--baseline", "--model", "blla.mlmodel", "--text-direction",
                             "horizontal-rl"] if segment else None,
            "recognition": ["ocr", "--model", model_name, *ocr[3:]],
        },
    }
    return recognize, provenance


def main():
    from raw_output import add_batch_arguments, run_cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("page", "line"), required=True)
    add_batch_arguments(parser)
    args = parser.parse_args()
    run_cli(args, parser, prepare, [Path(__file__)])


if __name__ == "__main__":
    main()
