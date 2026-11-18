"""Manifests: the exact files a run sends, each with its sha256.

A manifest lists paths relative to a root, never absolute ones, so it can be shared without exposing
the machine's layout. Before an upload every file is hashed again, checked against the policy (its root
must be an open root; the local denylist by path and by content) and compared with its manifest entry; a
file that changed since the manifest was written stops the run. On the pod, ``sha256sum -c`` checks the same hashes after unpacking.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .policy import Policy

SCHEMA = "heocr_remote_manifest.v1"


class ManifestError(Exception):
    pass


@dataclass(frozen=True)
class Entry:
    path: str     # relative, POSIX separators
    sha256: str
    size: int


def sha256_file(path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(rel: str) -> str:
    p = PurePosixPath(rel)
    if p.is_absolute() or not p.parts or any(part in ("..", "") for part in p.parts) or "\\" in rel:
        raise ManifestError(f"unsafe manifest path: {rel!r}")
    return str(p)


def build(root, policy: Policy, includes=("**",), excludes=()) -> list[Entry]:
    """Walk ``root`` (an open root, or a folder under one) and list the files matching ``includes``."""
    root = Path(root).expanduser()
    category = policy.check_root(root)
    entries = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            if not any(fnmatch.fnmatch(rel, g) for g in includes):
                continue
            if any(fnmatch.fnmatch(rel, g) for g in excludes):
                continue
            policy.check_file(full, category)
            digest = sha256_file(full)
            policy.check_hash(rel, digest)
            entries.append(Entry(rel, digest, full.stat().st_size))
    if not entries:
        raise ManifestError(f"no files matched under {root}")
    return entries


def dump(entries, path) -> None:
    data = {"schema": SCHEMA, "files": [e.__dict__ for e in entries]}
    Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load(path) -> list[Entry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ManifestError(f"{path}: not a {SCHEMA} manifest")
    out, seen = [], set()
    for f in data["files"]:
        rel = safe_rel(f["path"])
        if rel in seen:
            raise ManifestError(f"{path}: {rel} listed twice")
        seen.add(rel)
        sha = str(f["sha256"]).lower()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ManifestError(f"{path}: {rel} has no valid sha256")
        out.append(Entry(rel, sha, int(f["size"])))
    if not out:
        raise ManifestError(f"{path}: empty manifest")
    return out


def verify_local(entries, root, policy: Policy) -> int:
    """Re-check every file before upload: the policy by path and content, then its hash. Returns bytes."""
    root = Path(root).expanduser()
    category = policy.check_root(root)
    total = 0
    for e in entries:
        full = root / e.path
        policy.check_file(full, category)
        if full.is_symlink() or not full.is_file():
            raise ManifestError(f"{e.path}: missing, or not a regular file")
        digest = sha256_file(full)
        policy.check_hash(e.path, digest)
        if digest != e.sha256:
            raise ManifestError(f"{e.path}: changed since the manifest was written")
        total += full.stat().st_size
    return total


def sha256sum_text(entries) -> str:
    """The manifest in ``sha256sum -c`` format."""
    return "".join(f"{e.sha256}  {e.path}\n" for e in entries)


def parse_sha256sum(text: str) -> dict[str, str]:
    """Parse ``sha256sum`` output into {relative path: hash}."""
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        rel = rel[2:] if rel.startswith("./") else rel
        out[safe_rel(rel)] = digest.strip().lower()
    return out
