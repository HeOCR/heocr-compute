"""Pod-side line render recipe. Shipped as a manifest-checked input, with no credentials."""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

RENDERER = "https://github.com/HeOCR/hocrsyngen"
FONT_MANIFEST = "src/hocrsyngen/data/synthetic/fonts/manifest.yaml"
ARCHIVE = "render.tar"
ARTIFACT_MANIFEST = "artifact-manifest.json"
ARTIFACT_SCHEMA = "heocr_remote_render_artifacts.v1"
MAX_ARCHIVE = (2 << 30) - (16 << 20)  # leave room for metadata within the runner's 2 GiB pull cap
# Keep the pilot's labels short and inside the governed Hebrew fonts' common repertoire.
TEXT = re.compile(r"[א-ת 0-9.,:;!?־׳״'()\"-]{20,48}")


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def select_lines(corpus: Path, output: Path, count: int, seed: int) -> list[str]:
    """Reservoir-sample eligible physical lines with stable source-line IDs."""
    rng = random.Random(seed)
    rows = []
    eligible = 0
    with corpus.open(encoding="utf-8-sig") as f:
        for number, raw in enumerate(f, 1):
            line = " ".join(unicodedata.normalize("NFC", raw).split())
            if not TEXT.fullmatch(line) or not re.search("[א-ת]", line):
                continue
            eligible += 1
            row = {"id": f"pby-{number}", "text": line}
            if len(rows) < count:
                rows.append(row)
            elif (slot := rng.randrange(eligible)) < count:
                rows[slot] = row
    if len(rows) != count:
        raise ValueError(f"only {eligible} eligible lines for a count of {count}")
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return [r["id"] for r in rows]


def render_args(settings: dict, source: Path, dest: Path) -> list[str]:
    args = ["render-lines", "--input", str(source), "--output", str(dest), "--seed", str(settings["seed"])]
    for font in settings["fonts"]:
        args.extend(["--font", font])
    if settings["mode"] is not None:
        args.extend(["--mode", settings["mode"]])
    return args


def fetch_corpus(source: dict, path: Path) -> None:
    with urllib.request.urlopen(source["url"], timeout=60) as reply, path.open("wb") as out:
        remaining = source["size"]
        while chunk := reply.read(min(1 << 20, remaining + 1)):
            remaining -= len(chunk)
            if remaining < 0:
                raise ValueError("pinned corpus is larger than expected")
            out.write(chunk)
    if remaining or sha256(path) != source["sha256"]:
        raise ValueError("pinned corpus failed size or SHA-256 verification")


def pack(rendered: Path, out: Path, ids: list[str], settings: dict) -> None:
    """Require one image/label pair per selected line and hash every bundled file."""
    summary = json.loads((rendered / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("lines_in") != len(ids) or summary.get("rendered") != len(ids)
            or summary.get("refused") != [] or summary.get("font_manifest_sha256") != settings["font_manifest_sha256"]):
        raise ValueError("renderer did not complete the requested batch with the pinned font manifest")
    paths = ["manifest.jsonl", "summary.json"]
    paths.extend(f"lines/{id_}.{suffix}" for id_ in ids for suffix in ("png", "gt.txt"))
    if rendered.is_symlink() or (rendered / "lines").is_symlink():
        raise ValueError("render output directories must not be symlinks")
    entries = []
    for rel in sorted(paths):
        p = rendered / rel
        if p.is_symlink() or not p.is_file() or not p.stat().st_size:
            raise ValueError(f"missing or invalid render output: {rel}")
        entries.append({"path": rel, "sha256": sha256(p), "size": p.stat().st_size})

    # USTAR uses one 512-byte header and pads each file to a 512-byte boundary.
    data_bytes = sum(512 + ((entry["size"] + 511) // 512) * 512 for entry in entries)
    if data_bytes + tarfile.RECORDSIZE + 1024 > MAX_ARCHIVE:
        raise ValueError("render archive would exceed the 2 GiB pull limit")
    archive = out / ARCHIVE
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        for entry in entries:
            tar.add(rendered / entry["path"], arcname=entry["path"], recursive=False)
    info = {"schema": ARTIFACT_SCHEMA, "job": settings, "files": entries,
            "archive": {"path": ARCHIVE, "size": archive.stat().st_size, "sha256": sha256(archive)}}
    (out / ARTIFACT_MANIFEST).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    settings = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    work = Path("work").resolve()
    repo = work / "hocrsyngen"
    revision = settings["code_revision"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "fetch", "-q", "--depth=1", RENDERER + ".git", revision], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=repo, check=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != revision:
        raise ValueError("renderer checkout did not match the pinned commit")
    if sha256(repo / FONT_MANIFEST) != settings["font_manifest_sha256"]:
        raise ValueError("font manifest did not match the pinned SHA-256")
    venv = work / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    subprocess.run([str(venv / "bin/python"), "-m", "pip", "--quiet", "--disable-pip-version-check",
                    "install", str(repo)], check=True)
    # This command verifies the packaged font files against their governed SHA-256 values.
    renderer = str(venv / "bin/hocrsyngen")
    governed = json.loads(subprocess.check_output([renderer, "render-lines", "--list-fonts"], text=True))
    if any(governed.get(font, {}).get("license") != "OFL-1.1" for font in settings["fonts"]):
        raise ValueError("the requested fonts must be governed OFL-1.1 faces")
    corpus = work / "corpus.txt"
    selected = work / "selected.jsonl"
    dest = work / "rendered"
    fetch_corpus(settings["source"], corpus)
    ids = select_lines(corpus, selected, settings["count"], settings["seed"])
    subprocess.run([renderer, *render_args(settings, selected, dest)], check=True)
    pack(dest, Path("out"), ids, settings)


if __name__ == "__main__":
    main()
