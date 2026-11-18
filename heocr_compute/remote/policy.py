"""What may be sent to a remote pod: an allowlist of open roots, then a local denylist.

Remote pods get open training inputs only. Two checks, both on every file, by its path as given and
after resolving symlinks:

1. **Open roots (allowlist).** A file is sent only if it lies under a root listed for one of the open
   categories below. Everything else is refused by default. The categories are fixed in code; where
   their roots sit on a machine is that machine's business, in a local file that is never committed:

       ~/.config/heocr/remote_open_roots.toml
       [roots]
       synthetic = ["~/data/synthetic-lists"]
       htext = ["~/data/htext"]
       pd_scans = ["~/data/pd-scans"]

2. **Local denylist.** ``~/.config/heocr/remote_denylist.txt``, also never committed, names what must
   never leave the machine even inside an open root. It is required: without it nothing is sent. One rule
   per line; ``#`` starts a comment:

       ~/data/private-set              a path: it and everything under it
       */reference-texts/*             a glob on the full resolved path, ignoring case
       sha256-names: ~/data/private    every file in that folder is named by its sha256; a file with one
                                       of those hashes is refused wherever it sits and whatever it is called

The rules hold for the whole run: no flag loosens them.
"""
from __future__ import annotations

import fnmatch
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

OPEN_CATEGORIES = {
    "synthetic": "synthetic training lists and pages",
    "htext": "public-domain Hebrew text (HeOCR/htext)",
    "pd_scans": "public-domain scans",
}
OPEN_ROOTS_FILE = "~/.config/heocr/remote_open_roots.toml"
DENYLIST_FILE = "~/.config/heocr/remote_denylist.txt"
OPEN_ROOTS_ENV = "HEOCR_REMOTE_OPEN_ROOTS"
DENYLIST_ENV = "HEOCR_REMOTE_DENYLIST"

_SHA256 = re.compile(r"[0-9a-f]{64}")


class Refused(Exception):
    """A path or file that must not be sent to a remote pod."""


def _expand(p) -> Path:
    return Path(os.path.expanduser(str(p)))


def _real(p) -> Path:
    return _expand(p).resolve(strict=False)


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


@dataclass
class Policy:
    roots: dict[str, list[Path]] = field(default_factory=dict)   # category -> resolved roots
    deny_paths: list[Path] = field(default_factory=list)
    deny_globs: list[str] = field(default_factory=list)
    deny_hashes: frozenset[str] = frozenset()

    # -- loading -----------------------------------------------------------------------------------
    @classmethod
    def load(cls, open_roots: str | None = None, denylist: str | None = None) -> "Policy":
        roots_file = _expand(open_roots or os.environ.get(OPEN_ROOTS_ENV) or OPEN_ROOTS_FILE)
        deny_file = _expand(denylist or os.environ.get(DENYLIST_ENV) or DENYLIST_FILE)
        if not roots_file.is_file():
            raise Refused(f"no open roots configured ({roots_file}): nothing may be sent")
        if not deny_file.is_file():
            raise Refused(f"no local denylist ({deny_file}): nothing may be sent")
        p = cls()
        data = tomllib.loads(roots_file.read_text(encoding="utf-8")).get("roots", {})
        for cat, paths in data.items():
            if cat not in OPEN_CATEGORIES:
                raise Refused(f"{roots_file}: {cat!r} is not an open category ({', '.join(OPEN_CATEGORIES)})")
            if isinstance(paths, str):
                paths = [paths]
            p.roots[cat] = [_real(x) for x in paths]
        if not any(p.roots.values()):
            raise Refused(f"{roots_file}: no open roots listed")
        p._read_denylist(deny_file)
        return p

    def _read_denylist(self, path: Path) -> None:
        hashes, rules = set(), 0
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            rules += 1
            if line.startswith("sha256-names:"):
                folder = _expand(line.split(":", 1)[1].strip())
                if folder.is_dir():
                    hashes |= {f.name.split(".", 1)[0].lower() for f in folder.iterdir()
                               if _SHA256.fullmatch(f.name.split(".", 1)[0].lower())}
            elif line.startswith(("/", "~")):
                self.deny_paths.append(_real(line))
            else:
                self.deny_globs.append(line.lower())
        if not rules:
            raise Refused(f"{path}: the local denylist is empty")
        self.deny_hashes = frozenset(hashes)

    # -- checks ------------------------------------------------------------------------------------
    def category_of(self, path) -> str | None:
        real = _real(path)
        for cat, roots in self.roots.items():
            if any(_under(real, r) for r in roots):
                return cat
        return None

    def deny_reason(self, path) -> str | None:
        given, real = _expand(path).absolute(), _real(path)
        for cand in (given, real):
            for d in self.deny_paths:
                if _under(cand, d):
                    return "listed in the local denylist"
            low = cand.as_posix().lower()
            for g in self.deny_globs:
                if fnmatch.fnmatchcase(low, g):
                    return f"matches the local denylist rule {g!r}"
        return None

    def check_root(self, root) -> str:
        """The open category of a manifest root, or Refused."""
        reason = self.deny_reason(root)
        if reason:
            raise Refused(f"{root}: {reason}")
        cat = self.category_of(root)
        if cat is None:
            raise Refused(f"{root}: not under an open root ({', '.join(OPEN_CATEGORIES)}); nothing else is sent")
        return cat

    def check_file(self, path, root_category: str) -> None:
        reason = self.deny_reason(path)
        if reason:
            raise Refused(f"{path}: {reason}")
        if self.category_of(path) != root_category:
            raise Refused(f"{path}: leaves its open root (a symlink out of it?)")

    def check_hash(self, path, sha256: str) -> None:
        if sha256.lower() in self.deny_hashes:
            raise Refused(f"{path}: its content is listed in the local denylist")

    def with_root(self, category: str, root) -> "Policy":
        """A copy with one more open root: used for the smoke run's own invented inputs only."""
        roots = {k: list(v) for k, v in self.roots.items()}
        roots.setdefault(category, []).append(_real(root))
        return Policy(roots, list(self.deny_paths), list(self.deny_globs), self.deny_hashes)
