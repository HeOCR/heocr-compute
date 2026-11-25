"""Consume a SHA-pinned heocr-clean TRAIN batch on the Mac."""

from hashlib import file_digest
import json
import os
from pathlib import Path
from shutil import copyfile
import tarfile
import tempfile

from train import ID, SHA256, digest, matches, rows_from_manifest


def append_batch(output, manifest, receipt_path, receipt_sha):
    if not matches(SHA256, receipt_sha) or digest(receipt_path) != receipt_sha:
        raise ValueError("local TRAIN batch CLEAN receipt SHA-256 mismatch")
    signed = json.loads(receipt_path.read_bytes())
    source = signed.get("batch")
    if (signed.get("schema") != "heocr.handwriting_train_batch.v1" or signed.get("split") != "train"
            or signed.get("verdict") != "TRAIN CLEAN" or signed.get("runpod_allowed") is not False
            or signed.get("partner") is not False or not matches(ID, source)
            or source in {row["id"] for row in manifest["sources"]}):
        raise ValueError("expected a distinct, Mac-only, CLEAN local TRAIN batch")
    batch_rows = receipt_path.with_name("rows.jsonl")
    batch_tar = receipt_path.with_name("lines.tar")
    if digest(batch_rows) != signed.get("rows_sha256") or digest(batch_tar) != signed.get("archive_sha256"):
        raise ValueError("local TRAIN batch roster or archive SHA-256 mismatch")

    rows, members = [], {}
    for line in batch_rows.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key, writer = row.get("id"), row.get("group")
        role = row.get("role")
        if (not matches(ID, key) or not matches(ID, writer) or role not in ("fit", "monitor")
                or row.get("split") != "train"
                or not all(matches(SHA256, row.get(name)) for name in ("image_sha256", "label_sha256"))):
            raise ValueError("invalid local TRAIN pair")
        uid = f"{source}-{key}"
        if not matches(ID, uid) or f"lines/{key}.png" in members:
            raise ValueError("invalid or duplicate local TRAIN pair id")
        for suffix, field in ((".png", "image_sha256"), (".gt.txt", "label_sha256")):
            members[f"lines/{key}{suffix}"] = f"lines/{uid}{suffix}", row[field]
        rows.append({"id": uid, "source": source, "split": "train", "group": writer, "role": role,
                     "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"]})
    if not rows:
        raise ValueError("local TRAIN batch has no pairs")

    with tarfile.open(batch_tar) as original, tarfile.open(output, mode="a") as archive:
        for member in original:
            pair = members.pop(member.name, None)
            if pair is None or not member.isfile():
                raise ValueError("local TRAIN batch has an undeclared or invalid tar member")
            name, expected = pair
            with original.extractfile(member) as stream:
                if file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("local TRAIN batch member SHA-256 mismatch")
                stream.seek(0)
                info = tarfile.TarInfo(name)
                info.size = member.size
                archive.addfile(info, stream)
    if members:
        raise ValueError("local TRAIN batch is missing a declared tar member")
    manifest["sources"].append({"id": source, "clean_sha256": receipt_sha,
                                "runpod_allowed": False, "partner": False})
    manifest["rows"].extend(rows)


def combine(destination, manifest, batches):
    """Publish a combined archive after all additional signed batches validate."""
    if not batches:
        return
    with tempfile.TemporaryDirectory(prefix="handwriting-batches-", dir=destination.parent) as directory:
        draft = Path(directory) / destination.name
        copyfile(destination, draft)
        for filename, expected in batches:
            append_batch(draft, manifest, Path(filename), expected)
        manifest["archive_sha256"] = digest(draft)
        rows_from_manifest(manifest, local=True)
        os.replace(draft, destination)
