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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("page", "line"), required=True)
    args = parser.parse_args()
    if metadata.version("kraken") != "7.2":
        raise RuntimeError("kraken 7.2 is required")
    import kraken
    model_name, digest = MODELS[args.model]
    model = args.assets / model_name
    verify(model, digest)
    segmenter = Path(kraken.__file__).parent / "blla.mlmodel"
    if args.mode == "page":
        verify(segmenter, SEGMENTER_SHA)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "output.txt"
        cmd = [str(Path(sys.executable).parent / "kraken"), "--device", "cpu", "--threads", "1", "--raise-on-error", "-i", str(args.input), str(output)]
        if args.mode == "page":
            cmd += ["segment", "--baseline", "--model", str(segmenter), "--text-direction", "horizontal-rl"]
        cmd += ["ocr", "--model", str(model), "--num-line-workers", "0", "--reorder", "--base-dir", "R"]
        if args.mode == "line":
            cmd += ["--no-segmentation"]
        subprocess.run(cmd, check=True)
        args.output.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
