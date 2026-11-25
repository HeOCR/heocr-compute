"""Prepare exact, cloud-cleared TRAIN pairs and continue Kraken PP-OCRv6."""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


STARTER_SHA256 = "15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9"
WEIGHTS_NAME = "handwriting-medium-pilot.safetensors"
ROLES = ("fit", "monitor")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows_from_manifest(manifest):
    if (not isinstance(manifest, dict) or manifest.get("schema") != "heocr.handwriting_train.v1"
            or manifest.get("split") != "train" or not isinstance(manifest.get("sources"), list)
            or not isinstance(manifest.get("rows"), list)
            or not matches(SHA256, manifest.get("archive_sha256"))):
        raise ValueError("expected a TRAIN-only handwriting manifest with a pinned archive")
    sources = {}
    for source in manifest["sources"]:
        if not isinstance(source, dict):
            raise ValueError("invalid TRAIN source record")
        key = source.get("id")
        if (not matches(ID, key) or key in sources or not matches(SHA256, source.get("clean_sha256"))
                or source.get("runpod_allowed") is not True or source.get("partner") is not False):
            raise ValueError("every source needs separate CLEAN and non-partner RunPod clearance")
        sources[key] = source

    ids, groups, crops, roles, used = set(), {}, {}, set(), set()
    rows = manifest["rows"]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid TRAIN line row")
        key, group, role, source = (row.get(k) for k in ("id", "group", "role", "source"))
        image_sha, label_sha = row.get("image_sha256"), row.get("label_sha256")
        if (not matches(ID, key) or key in ids or not matches(ID, group) or role not in ROLES
                or row.get("split") != "train" or not matches(ID, source) or source not in sources
                or not matches(SHA256, image_sha) or not matches(SHA256, label_sha)):
            raise ValueError("invalid or duplicate TRAIN line row")
        if groups.setdefault(group, role) != role or crops.setdefault(image_sha, role) != role:
            raise ValueError("TRAIN fit and monitor share a writer group or image")
        ids.add(key)
        roles.add(role)
        used.add(source)
    if roles != set(ROLES) or used != set(sources):
        raise ValueError("manifest must bind both TRAIN roles and exactly their source roster")
    return rows


def prepare(manifest_path, archive_path, destination):
    """Extract only the exact manifest members and return Kraken's two path lists."""
    manifest = json.loads(manifest_path.read_bytes())
    rows = rows_from_manifest(manifest)
    if digest(archive_path) != manifest["archive_sha256"]:
        raise ValueError("TRAIN archive SHA-256 mismatch")
    lists = {role: [] for role in ROLES}
    members_by_name = {}
    for row in rows:
        root = row["id"]
        members_by_name[f"lines/{root}.png"] = row["image_sha256"]
        members_by_name[f"lines/{root}.gt.txt"] = row["label_sha256"]
        lists[row["role"]].append(destination / "lines" / f"{root}.png")
    (destination / "lines").mkdir()
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            expected = members_by_name.pop(member.name, None)
            if expected is None or not member.isfile():
                raise ValueError("TRAIN archive has an undeclared, duplicate or non-regular member")
            target = destination / member.name
            with archive.extractfile(member) as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            if digest(target) != expected:
                raise ValueError("TRAIN line SHA-256 mismatch")
            if target.suffix == ".txt":
                label = target.read_bytes().decode("utf-8")
                line = label.removesuffix("\n")
                if not line.strip() or "\r" in line or "\n" in line or "\0" in line:
                    raise ValueError("TRAIN labels must contain exactly one nonempty UTF-8 line")
    if members_by_name:
        raise ValueError("TRAIN archive is missing declared members")
    for role, paths in lists.items():
        (destination / f"{role}.lst").write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    return {role: len(paths) for role, paths in lists.items()}


def run(root):
    if version("kraken") != "7.2":
        raise RuntimeError("the pilot requires Kraken 7.2")
    inputs = root / "in"
    starter = inputs / "starter.safetensors"
    if digest(starter) != STARTER_SHA256:
        raise ValueError("unexpected starter checkpoint")
    with tempfile.TemporaryDirectory(prefix="handwriting-pilot-", dir=root) as scratch:
        work = Path(scratch)
        counts = prepare(inputs / "manifest.json", inputs / "lines.tar", work)
        print(f"TRAIN fit {counts['fit']}, TRAIN monitor {counts['monitor']}", flush=True)
        cmd = [str(Path(sys.executable).with_name("ketos")), "--device", "cpu", "--seed", "20261125",
               "--deterministic", "--workers", "0", "--threads", "8", "train", "--arch", "ppocrv6",
               "--load", str(starter), "--resize", "union", "--format-type", "path",
               "--training-data", "fit.lst", "--evaluation-data", "monitor.lst", "--normalization", "NFD",
               "--normalize-whitespace", "--reorder", "--base-dir", "R", "--batch-size", "8",
               "--optimizer", "AdamW", "--schedule", "constant", "--lrate", "0.0001", "--quit", "fixed", "--epochs", "3",
               "--output", "checkpoints", "--weights-format", "safetensors"]
        subprocess.run(cmd, check=True, cwd=work)
        best = list((work / "checkpoints").glob("best_*.safetensors"))
        if len(best) != 1 or best[0].stat().st_size == 0:
            raise RuntimeError("expected one nonempty TRAIN-monitor checkpoint")
        out = root / "out"
        with tempfile.NamedTemporaryFile(prefix=f".{WEIGHTS_NAME}.", suffix=".tmp", dir=out, delete=False) as output:
            pending = Path(output.name)
            try:
                with best[0].open("rb") as source:
                    shutil.copyfileobj(source, output)
                output.close()
                os.replace(pending, out / WEIGHTS_NAME)
            finally:
                pending.unlink(missing_ok=True)


if __name__ == "__main__":
    run(Path.cwd())
