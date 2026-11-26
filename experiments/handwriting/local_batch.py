"""Consume a SHA-pinned heocr-clean TRAIN batch on the Mac."""

from hashlib import file_digest
import json
import os
from pathlib import Path
from shutil import copyfile
import tarfile
import tempfile

from train import ID, SHA256, digest, matches, rows_from_manifest


def index_batches(index_path, index_sha):
    """Resolve only the batches named in the pinned CLEAN index."""
    if not matches(SHA256, index_sha) or digest(index_path) != index_sha:
        raise ValueError("local TRAIN index SHA-256 mismatch")
    index = json.loads(index_path.read_bytes())
    if (not isinstance(index, dict) or index.get("schema") != "heocr.handwriting_train_index.v1"
            or index.get("split") != "train" or not isinstance(index.get("batches"), list)):
        raise ValueError("expected a Mac-local TRAIN batch index")
    root = index_path.parent.resolve()
    seen, batches = set(), []
    for item in index["batches"]:
        if not isinstance(item, dict):
            raise ValueError("invalid CLEAN index entry")
        key, clean_sha = item.get("id"), item.get("clean_sha256")
        if not matches(ID, key) or key in seen or not matches(SHA256, clean_sha):
            raise ValueError("invalid or duplicate CLEAN index entry")
        receipt = root / key / "clean.json"
        for path in (receipt, receipt.with_name("rows.jsonl"), receipt.with_name("lines.tar"),
                     receipt.with_name("row-tags.jsonl")):
            if path.resolve() != path:
                raise ValueError("indexed batch must use files within its named directory")
        seen.add(key)
        batches.append((receipt, clean_sha, key))
    return batches


def valid_tags(tags):
    return isinstance(tags, list) and all(matches(ID, tag) for tag in tags) and len(tags) == len(set(tags))


def read_row_tags(receipt_path, expected):
    if expected is None:
        return None
    sidecar = receipt_path.with_name("row-tags.jsonl")
    if not matches(SHA256, expected) or digest(sidecar) != expected:
        raise ValueError("local TRAIN row tags SHA-256 mismatch")
    result = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key, writer, tags = row.get("id"), row.get("group"), row.get("tags")
        if not matches(ID, key) or not matches(ID, writer) or key in result or not valid_tags(tags):
            raise ValueError("invalid or duplicate CLEAN row tags")
        result[key] = (writer, tags)
    return result


def append_batch(output, manifest, receipt_path, receipt_sha, expected_source=None):
    if not matches(SHA256, receipt_sha) or digest(receipt_path) != receipt_sha:
        raise ValueError("local TRAIN batch CLEAN receipt SHA-256 mismatch")
    signed = json.loads(receipt_path.read_bytes())
    source = signed.get("batch")
    if (signed.get("schema") != "heocr.handwriting_train_batch.v1" or signed.get("split") != "train"
            or signed.get("verdict") != "TRAIN CLEAN" or signed.get("runpod_allowed") is not False
            or signed.get("partner") is not False or not matches(ID, source)
            or (expected_source is not None and source != expected_source)
            or source in {row["id"] for row in manifest["sources"]}):
        raise ValueError("expected a distinct, Mac-only, CLEAN local TRAIN batch")
    batch_rows = receipt_path.with_name("rows.jsonl")
    batch_tar = receipt_path.with_name("lines.tar")
    if digest(batch_rows) != signed.get("rows_sha256") or digest(batch_tar) != signed.get("archive_sha256"):
        raise ValueError("local TRAIN batch roster or archive SHA-256 mismatch")

    approved_tags = read_row_tags(receipt_path, signed.get("row_tags_sha256"))
    rows, members = [], {}
    for line in batch_rows.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key, writer = row.get("id"), row.get("group")
        role = row.get("role")
        tags = row.get("tags", [])
        if (not matches(ID, key) or not matches(ID, writer) or role not in ("fit", "monitor")
                or row.get("split") != "train"
                or not valid_tags(tags)
                or not all(matches(SHA256, row.get(name)) for name in ("image_sha256", "label_sha256"))):
            raise ValueError("invalid local TRAIN pair")
        if approved_tags is not None:
            tag_writer, additional = approved_tags.pop(key, (None, []))
            if tag_writer != writer:
                raise ValueError("CLEAN row tags do not match the TRAIN pair and writer")
            tags = list(dict.fromkeys(tags + additional))
        uid = f"{source}-{key}"
        if not matches(ID, uid) or f"lines/{key}.png" in members:
            raise ValueError("invalid or duplicate local TRAIN pair id")
        for suffix, field in ((".png", "image_sha256"), (".gt.txt", "label_sha256")):
            members[f"lines/{key}{suffix}"] = f"lines/{uid}{suffix}", row[field]
        rows.append({"id": uid, "source": source, "split": "train", "group": writer, "role": role,
                     "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
                     **({"tags": tags} if tags else {})})
    if not rows:
        raise ValueError("local TRAIN batch has no pairs")
    if approved_tags:
        raise ValueError("CLEAN row tags include undeclared TRAIN pairs")

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
        for filename, expected, batch_id in batches:
            append_batch(draft, manifest, Path(filename), expected, batch_id)
        manifest["archive_sha256"] = digest(draft)
        rows_from_manifest(manifest, local=True)
        os.replace(draft, destination)
