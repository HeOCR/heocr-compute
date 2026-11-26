"""Stage every Mac-local batch in a pinned TRAIN CLEAN index."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
import tempfile

from local_batch import combine, index_batches
from train import SHA256, digest, matches, rows_from_manifest


def compose(base, base_sha, index, index_sha, output):
    batches = index_batches(index, index_sha)
    base_manifest = base / "manifest.json"
    if not matches(SHA256, base_sha) or digest(base_manifest) != base_sha:
        raise ValueError("base TRAIN manifest SHA-256 mismatch")
    manifest = json.loads(base_manifest.read_bytes())
    rows_from_manifest(manifest, local=True)
    if digest(base / "lines.tar") != manifest["archive_sha256"]:
        raise ValueError("base TRAIN archive SHA-256 mismatch")

    with tempfile.TemporaryDirectory(prefix="handwriting-compose-", dir=output.parent) as scratch:
        draft = Path(scratch) / "bundle"
        draft.mkdir(mode=0o700)
        copyfile(base / "lines.tar", draft / "lines.tar")
        combine(draft / "lines.tar", manifest, batches)
        manifest["local_train_index_sha256"] = index_sha
        manifest["base_manifest_sha256"] = base_sha
        dest_manifest = draft / "manifest.json"
        dest_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pins = {"manifest_sha256": digest(dest_manifest), "archive_sha256": manifest["archive_sha256"],
                "indexed_batches": len(batches), "train_pairs": len(manifest["rows"])}
        output.mkdir(mode=0o700)
        draft.rename(output)
    return pins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "index", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("base", "index"):
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(compose(args.base, args.base_sha256, args.index, args.index_sha256, args.output), sort_keys=True))
