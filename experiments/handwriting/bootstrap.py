"""Render a CLEAN source text pool as Mac-local handwriting TRAIN pairs."""

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import random
import subprocess
import tarfile
import tempfile
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageFilter

from local_batch import combine
from train import ID, SHA256, digest, matches


FONT_REVISION = "6ebe737e44fd64ee736bbc6feed6136a6dbfd222"
FONTS = (
    ("gveretlevin", "Gveret Levin", "GveretLevin-Regular.ttf",
     "05c443395d48b97f43469da4fb08772a4c612fda306348577dbc40a1267912b1",
     "d9b228718f3f81e69ce614c6b2f4c382fd34396ca8149b4f28d07e85b8781eba"),
    ("playpensanshebrew", "Playpen Sans Hebrew", "PlaypenSansHebrew[wght].ttf",
     "0ec8a95033c5a01e1799c60b49ab76d9922a68f715a081fc1f5587e4d2bb5b69",
     "59440310cc2c539b53c835306f74ac8c01f1ece58bf4c65e556fca275b80a730"),
    ("amaticsc", "Amatic SC", "AmaticSC-Regular.ttf",
     "513a76f96d39a5366bd27c258674aaea35ee51c42b8f3060387f74649ee8c8aa",
     "301f77908df2b00141b4dcea50cc1a586c3afcee78843bd811088e8263623f9d"),
)
SEED = 20261125


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def clean_rows(root, expected):
    clean = root / "out" / "train" / "clean.json"
    keep_path = clean.with_name("keep.jsonl")
    units_path = root / "units.jsonl"
    if not matches(SHA256, expected) or digest(clean) != expected:
        raise ValueError("CLEAN pin differs")
    receipt = json.loads(clean.read_bytes())
    if receipt.get("split") != "train" or receipt.get("verdict") != "TRAIN CLEAN":
        raise ValueError("expected a TRAIN CLEAN source pool")
    if digest(keep_path) != receipt["ordered_ledgers"]["keep"] or digest(units_path) != receipt["units_sha256"]:
        raise ValueError("TRAIN text or keep ledger differs")
    units = {unit["id"]: unit for unit in jsonl(units_path)}
    result = []
    for row in jsonl(keep_path):
        unit = units[row["id"]]
        line, = unit["lines"]
        text = line["text"]
        if text != text.strip() or not text or len(text.splitlines()) != 1:
            raise ValueError("TRAIN text pool contains an invalid line")
        result.append((unit["id"], unit["source"]["pby_work"], text))
    return result


def font_chars(font):
    output = subprocess.check_output(["fc-query", "--index=0", "--format=%{charset}", str(font)], text=True)
    codes = set()
    for part in output.split():
        ends = part.split("-")
        codes.update(range(int(ends[0], 16), int(ends[-1], 16) + 1))
    return codes


def read_fonts(root):
    for slug, family, name, font_sha, license_sha in FONTS:
        path = root / name
        license_path = root / f"{slug}-OFL.txt"
        if digest(path) != font_sha or digest(license_path) != license_sha:
            raise ValueError("font or OFL-1.1 license pin differs")
        yield slug, family, path, font_chars(path)


def render(text, family, raw, seed, env):
    rng = random.Random(seed)
    markup = raw.with_suffix(".pango")
    markup.write_text(f'<span fallback="false">{escape(text)}</span>', encoding="utf-8")
    subprocess.run(["pango-view", "--no-display", "--pixels", "--rtl", "--language=he", "--single-par",
                    "--margin=12", f"--font={family} {rng.randint(34, 52)}", "--markup",
                    f"--output={raw}", str(markup)],
                   env=env, check=True, capture_output=True)
    with Image.open(raw) as source:
        image = source.convert("L")
    image = image.rotate(rng.uniform(-2.8, 2.8), Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.85)))
    ink = np.asarray(image, dtype=np.float32) / 255
    background = np.linspace(rng.uniform(232, 254), rng.uniform(232, 254), image.width, dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0, rng.uniform(0.8, 4), ink.shape)
    pixels = (rng.uniform(13, 49) + (background - 35) * ink + noise).clip(0, 255)
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buffer, format="PNG", dpi=(300, 300))
    return buffer.getvalue()


def add_member(archive, name, content):
    entry = tarfile.TarInfo(name)
    entry.size = len(content)
    archive.addfile(entry, io.BytesIO(content))
    return sha256(content).hexdigest()


def stage(args):
    fonts = list(read_fonts(args.fonts))
    if {path.name for path in args.fonts.iterdir() if path.suffix.lower() in {".ttf", ".otf", ".ttc"}} != {
            path.name for _, _, path, _ in fonts}:
        raise ValueError("font directory must contain exactly the pinned font faces")
    if not matches(ID, args.source_id):
        raise ValueError("invalid source id")
    units = clean_rows(args.clean_root, args.clean_sha256)
    work_hash = {work: sha256(f"{SEED}:{work}".encode()).hexdigest() for _, work, _ in units}
    if len(work_hash) < 3:
        raise ValueError("need at least three source works for fit and the TRAIN monitor")
    monitor = set(sorted(work_hash, key=work_hash.__getitem__)[:2])
    args.output.mkdir(mode=0o700)
    archive_path = args.output / "lines.tar"
    counts, omitted = Counter(), Counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="handwriting-render-") as directory:
        scratch = Path(directory)
        config = scratch / "fonts.conf"
        cache = scratch / "cache"
        config.write_text(f"<fontconfig><dir>{escape(str(args.fonts.resolve()))}</dir>"
                          f"<cachedir>{cache}</cachedir></fontconfig>", encoding="utf-8")
        env = os.environ | {"FONTCONFIG_FILE": str(config), "FONTCONFIG_PATH": str(scratch)}
        for _, family, path, _ in fonts:
            resolved = subprocess.check_output(["fc-match", "--format=%{file}", family], env=env, text=True)
            if Path(resolved).resolve() != path.resolve():
                raise ValueError("the renderer resolved an unexpected font")
        with tarfile.open(archive_path, "w") as archive:
            for unit, work, text in units:
                role = "monitor" if work in monitor else "fit"
                for slug, family, _, chars in fonts:
                    if any(ord(char) not in chars for char in text if not char.isspace()):
                        omitted[slug] += 1
                        continue
                    for variant in range(2):
                        key = f"{unit.replace(':', '-')}-{slug}-{variant}"
                        seed = int(sha256(f"{SEED}:{key}".encode()).hexdigest()[:16], 16)
                        png = render(text, family, scratch / "line.png", seed, env)
                        rows.append({"id": key, "source": args.source_id, "group": work_hash[work],
                                     "split": "train", "role": role,
                                     "image_sha256": add_member(archive, f"lines/{key}.png", png),
                                     "label_sha256": add_member(archive, f"lines/{key}.gt.txt", (text + "\n").encode())})
                        counts[role, slug] += 1
    manifest = {"schema": "heocr.handwriting_train.v1", "split": "train", "archive_sha256": digest(archive_path),
                "sources": [{"id": args.source_id, "clean_sha256": args.clean_sha256,
                             "runpod_allowed": False, "partner": False}], "rows": rows}
    batches = [(filename, expected, None) for filename, expected in args.local_train_batch]
    combine(archive_path, manifest, batches)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": {f"{role}/{slug}": count for (role, slug), count in sorted(counts.items())},
                      "missing_glyph_line_font_pairs": dict(omitted), "archive_sha256": manifest["archive_sha256"],
                      "manifest_sha256": digest(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("clean-root", "fonts", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--clean-sha256", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--local-train-batch", nargs=2, metavar=("CLEAN_RECEIPT", "SHA256"),
                        action="append", default=[])
    stage(parser.parse_args())
