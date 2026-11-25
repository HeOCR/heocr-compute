"""Locally compare one admitted DEV line roster with stock Hebrew and Kraken."""

import argparse
from collections import defaultdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unicodedata

from train import ID, SHA256, STARTER_SHA256, digest, matches


STOCK_SHA256 = "dbaa827aea6bc21215638447f17783a1004987c2d0bf5573d111fee397abdae5"


def check(path, expected):
    if not matches(SHA256, expected) or digest(path) != expected:
        raise ValueError("input SHA-256 mismatch")
    return path


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def paired(rows, baseline):
    groups = defaultdict(lambda: [0, 0])
    for line in rows:
        counts = groups[line["group"]]
        counts[0] += line["chars"]
        counts[1] += line["edits"]["candidate"] - line["edits"][baseline]
    population = list(groups.values())
    rng = random.Random(20261125)
    trials = []
    for _ in range(20000):
        draw = rng.choices(population, k=len(population))
        trials.append(sum(d for _, d in draw) / sum(n for n, _ in draw))
    trials.sort()
    delta = sum(d for _, d in population) / sum(n for n, _ in population)
    return {"groups": len(groups), "line_cer_delta": delta, "paired_95pct": [trials[500], trials[19499]],
            "verdict": "improved" if len(groups) > 1 and trials[19499] < 0 else "not improved"}


def read_output(command, env):
    result = subprocess.run(command, env=env, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"DEV reader failed (exit {result.returncode})")
    return result.stdout.decode("utf-8")


def score(args):
    if version("kraken") != "7.2":
        raise RuntimeError("the pilot comparison requires Kraken 7.2")
    manifest = json.loads(check(args.manifest, args.manifest_sha256).read_bytes())
    if manifest.get("schema") != "heocr.handwriting_dev.v1" or manifest.get("split") != "dev":
        raise ValueError("expected the independently cleared DEV line roster")
    candidate = check(args.candidate, args.candidate_sha256)
    readers = {"candidate": candidate}
    if args.starter is not None:
        readers["starter"] = check(args.starter, STARTER_SHA256)
    tess = check(args.tesseract, args.tesseract_sha256)
    check(args.tessdata / "heb.traineddata", STOCK_SHA256)
    kraken = Path(sys.executable).with_name("kraken")
    env = os.environ | {"OMP_THREAD_LIMIT": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    root = args.manifest.parent.resolve()
    rows, ids = [], set()
    with tempfile.TemporaryDirectory(prefix="handwriting-dev-") as directory:
        output = Path(directory) / "ocr.txt"
        for line in manifest["rows"]:
            key, group = line["id"], line["group"]
            if not matches(ID, key) or key in ids or not matches(ID, group) or line.get("split") != "dev":
                raise ValueError("invalid or duplicate DEV line")
            ids.add(key)
            image = (root / "lines" / f"{key}.png").resolve()
            label = (root / "lines" / f"{key}.gt.txt").resolve()
            if not image.is_relative_to(root) or not label.is_relative_to(root):
                raise ValueError("DEV pair is outside the staged directory")
            check(image, line["image_sha256"])
            expected = normalize(check(label, line["label_sha256"]).read_text(encoding="utf-8"))
            if not expected:
                raise ValueError("empty DEV reference")
            hypotheses = {"stock": read_output([str(tess), str(image), "stdout", "--tessdata-dir", str(args.tessdata),
                                                 "-l", "heb", "--oem", "1", "--psm", "13"], env)}
            for name, weights in readers.items():
                read_output([str(kraken), "--device", "cpu", "--threads", "1", "--raise-on-error",
                             "-i", str(image), str(output), "ocr", "--model", str(weights),
                             "--num-line-workers", "0", "--no-segmentation", "--reorder", "--base-dir", "R"], env)
                hypotheses[name] = output.read_text(encoding="utf-8")
                output.unlink()
            rows.append({"id": key, "group": group, "chars": len(expected),
                         "edits": {name: distance(expected, normalize(text)) for name, text in hypotheses.items()}})
    if not rows:
        raise ValueError("DEV roster is empty")
    return {"split": "dev", "manifest_sha256": args.manifest_sha256, "candidate_sha256": args.candidate_sha256,
            "stock_sha256": STOCK_SHA256, "tesseract_sha256": args.tesseract_sha256, "kraken": version("kraken"),
            "vs_stock": paired(rows, "stock"), **({"vs_starter": paired(rows, "starter")} if args.starter else {}),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "candidate", "tesseract", "tessdata", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("manifest", "candidate", "tesseract"):
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--starter", type=Path)
    args = parser.parse_args()
    result = score(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
