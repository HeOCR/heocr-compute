"""Run the registered handwriting DEV comparison locally."""

import argparse
from collections import defaultdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unicodedata

from train import ID, SHA256, digest, matches


STOCK_SHA256 = "dbaa827aea6bc21215638447f17783a1004987c2d0bf5573d111fee397abdae5"
FOLD_SHA256 = "2e2627104be4f27dc3204787dd3ff4f8591d95d1d96294c65833d2773041d317"
BIDI = re.compile(r"[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
LAYOUTS = ("printed_sheet", "separate_page")
READERS = ("candidate", "stock")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261125


def check(path, expected):
    if not matches(SHA256, expected) or digest(path) != expected:
        raise ValueError("input SHA-256 mismatch")
    return path


def normalize(text, translations):
    nfc = unicodedata.normalize("NFC", BIDI.sub("", text))
    return " ".join(nfc.translate(translations).split())


def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def paired(rows):
    groups = defaultdict(lambda: [0, 0])
    for line in rows:
        counts = groups[line["group"]]
        counts[0] += line["chars"]
        counts[1] += line["edits"]["candidate"] - line["edits"]["stock"]
    population = [groups[group] for group in sorted(groups)]
    rng = random.Random(BOOTSTRAP_SEED)
    trials = []
    for _ in range(BOOTSTRAP_DRAWS):
        draw = rng.choices(population, k=len(population))
        trials.append(100 * sum(d for _, d in draw) / sum(n for n, _ in draw))
    trials.sort()
    return [trials[249], trials[9749]]  # Nearest-rank 2.5th and 97.5th percentiles.


def aggregate(rows):
    chars = sum(row["chars"] for row in rows)
    return {
        "lines": len(rows), "writers": len({row["group"] for row in rows}), "reference_chars": chars,
        "readers": {
            name: {
                "cer": sum(row["edits"][name] for row in rows) / chars,
                "empty_lines": sum(row["empty"][name] for row in rows),
                "empty_share": sum(row["empty"][name] for row in rows) / len(rows),
            }
            for name in READERS
        },
    }


def summarize(rows):
    overall = aggregate(rows)
    candidate = overall["readers"]["candidate"]
    stock = overall["readers"]["stock"]
    bounds = paired(rows)
    guard = candidate["empty_lines"] <= stock["empty_lines"]
    by_layout = {}
    for layout in LAYOUTS:
        subset = [row for row in rows if row["layout"] == layout]
        writers = len({row["group"] for row in subset})
        by_layout[layout] = aggregate(subset) if writers > 1 else {"lines": len(subset), "writers": writers}
    return {"verdict": "improved" if bounds[1] < 0 and guard else "not improved",
            "overall": overall, "by_layout": by_layout, "empty_share_guard": guard,
            "cer_delta_pp": 100 * (candidate["cer"] - stock["cer"]), "paired_95pct_pp": bounds}


def admitted_rows(manifest):
    if manifest.get("schema") != "heocr.handwriting_dev.v1" or manifest.get("split") != "dev":
        raise ValueError("expected the admitted handwriting DEV roster")
    ids, pages = set(), {}
    admitted = []
    for row in manifest["rows"]:
        key, group, page, layout = (row.get(name) for name in ("id", "group", "page", "layout"))
        ordinal, retained = row.get("line"), row.get("retained")
        if (not all(matches(ID, item) for item in (key, group, page)) or key in ids
                or row.get("split") != "dev" or layout not in LAYOUTS
                or type(ordinal) is not int or not 1 <= ordinal <= 29 or type(retained) is not bool):
            raise ValueError("invalid or duplicate DEV line metadata")
        writer, page_layout, ordinals = pages.setdefault(page, (group, layout, set()))
        if writer != group or page_layout != layout or ordinal in ordinals:
            raise ValueError("inconsistent DEV page metadata")
        ordinals.add(ordinal)
        ids.add(key)
        if retained:
            if not all(matches(SHA256, row.get(name)) for name in ("image_sha256", "label_sha256")):
                raise ValueError("retained line is missing a crop or label pin")
            admitted.append(row)
    if any(len(ordinals) != 29 for _, _, ordinals in pages.values()):
        raise ValueError("DEV pages must contain metadata for all 29 lines, including masks")
    return admitted


def read_output(command, env):
    result = subprocess.run(command, env=env, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"DEV reader failed (exit {result.returncode})")
    return result.stdout.decode("utf-8")


def read_cases(root, rows, translations):
    cases = []
    for row in rows:
        key = row["id"]
        image = (root / "lines" / f"{key}.png").resolve()
        label = (root / "lines" / f"{key}.gt.txt").resolve()
        if not image.is_relative_to(root) or not label.is_relative_to(root):
            raise ValueError("DEV pair is outside the staged directory")
        check(image, row["image_sha256"])
        expected = normalize(check(label, row["label_sha256"]).read_text(encoding="utf-8"), translations)
        if not expected:
            raise ValueError("empty DEV reference")
        cases.append((row, image, expected))
    return cases


def read_both(cases, candidate, tess, tessdata, translations):
    kraken = Path(sys.executable).with_name("kraken")
    env = os.environ | {"OMP_THREAD_LIMIT": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    rows = []
    with tempfile.TemporaryDirectory(prefix="handwriting-dev-") as directory:
        for start in range(0, len(cases), 32):
            batch = cases[start:start + 32]
            command = [str(kraken), "--device", "cpu", "--threads", "1", "--raise-on-error"]
            outputs = []
            for index, (_, image, _) in enumerate(batch):
                output = Path(directory) / f"{start + index}.txt"
                command.extend(("-i", str(image), str(output)))
                outputs.append(output)
            command.extend(("ocr", "--model", str(candidate), "--num-line-workers", "0",
                            "--no-segmentation", "--reorder", "--base-dir", "R"))
            read_output(command, env)
            for (row, image, expected), output in zip(batch, outputs, strict=True):
                stock = read_output([str(tess), str(image), "stdout", "--tessdata-dir", str(tessdata),
                                     "-l", "heb", "--oem", "1", "--psm", "7", "--dpi", "300"], env)
                predictions = {"candidate": normalize(output.read_text(encoding="utf-8"), translations),
                               "stock": normalize(stock, translations)}
                rows.append({"group": row["group"], "layout": row["layout"], "chars": len(expected),
                             "edits": {name: distance(expected, text) for name, text in predictions.items()},
                             "empty": {name: not text for name, text in predictions.items()}})
    return rows


def actual_use(args):
    receipt = json.loads(check(args.actual_use, args.actual_use_sha256).read_bytes())
    if (receipt.get("registration") != args.registration or receipt.get("actual_use") != "signed"
            or receipt.get("manifest_sha256") != args.manifest_sha256
            or receipt.get("candidate_sha256") != args.candidate_sha256
            or not all(matches(SHA256, receipt.get(key)) for key in ("train_clean_sha256", "terminal_sha256"))):
        raise ValueError("actual-use receipt does not bind this registration, manifest and candidate")
    return receipt


def write_json(path, result):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


def score(args):
    permit = actual_use(args)
    manifest = json.loads(check(args.manifest, args.manifest_sha256).read_bytes())
    admitted = admitted_rows(manifest)
    groups = {row["group"] for row in admitted}
    result = {"registration": args.registration, "split": "dev", "actual_use_sha256": args.actual_use_sha256,
              "manifest_sha256": args.manifest_sha256, "candidate_sha256": args.candidate_sha256,
              "train_clean_sha256": permit["train_clean_sha256"], "terminal_sha256": permit["terminal_sha256"],
              "masked_lines": len(manifest["rows"]) - len(admitted)}
    if len(groups) < 4 or len(admitted) < 100:
        return result | {"verdict": "hold", "reason": "insufficient support",
                         "writers": len(groups), "lines": len(admitted), "attempted": False}
    if version("kraken") != "7.2":
        raise RuntimeError("the comparison requires Kraken 7.2")
    candidate = check(args.candidate, args.candidate_sha256)
    tess = check(args.tesseract, args.tesseract_sha256)
    check(args.tessdata / "heb.traineddata", STOCK_SHA256)
    translations = str.maketrans(json.loads(check(args.fold, FOLD_SHA256).read_bytes())["folds"])
    claim = args.output.with_name(args.output.name + ".started")
    if args.output.exists() or claim.exists():
        raise FileExistsError("receipt or registered-run claim already exists")
    cases = read_cases(args.manifest.parent.resolve(), admitted, translations)
    write_json(claim, result)
    rows = read_both(cases, candidate, tess, args.tessdata, translations)
    return result | summarize(rows) | {
        "attempted": True, "line_fold_v1_sha256": FOLD_SHA256, "stock_sha256": STOCK_SHA256,
        "tesseract_sha256": args.tesseract_sha256, "kraken": "7.2",
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "actual-use", "candidate", "fold", "tesseract", "tessdata", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("manifest", "actual-use", "candidate", "tesseract"):
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--registration", type=int, required=True)
    args = parser.parse_args()
    write_json(args.output, score(args))


if __name__ == "__main__":
    main()
