"""Transport raw comparator output and its separate run metadata."""

import hashlib
import json
from pathlib import Path
import re
import tempfile


SCHEMA = "heocr.raw-ocr.v1"


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def transport_text(text):
    # Use LF for line endings. Preserve form feeds and Unicode separators.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def add_batch_arguments(parser):
    parser.add_argument("--manifest", type=Path, help="JSON manifest of pinned raster inputs and IDs")
    parser.add_argument("--output-dir", type=Path, help="new output directory for batch mode")
    parser.add_argument("--runtime-sha256", help="digest of the runtime image or pinned local runtime receipt")
    parser.add_argument("--config-sha256", help="digest of the evaluator's frozen run configuration")


def batch_requested(args, parser):
    batch = (args.manifest, args.output_dir, args.runtime_sha256, args.config_sha256)
    if not any(value is not None for value in batch):
        return False
    if not all(batch) or args.input is not None or args.output is not None:
        parser.error("batch requires --manifest, --output-dir, --runtime-sha256 and --config-sha256 "
                     "in place of --input and --output")
    return True


def valid_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_manifest(manifest_bytes):
    items = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("manifest must be a nonempty JSON array")
    ids = set()
    for item in items:
        if not isinstance(item, dict) or item.keys() != {"id", "path", "sha256"}:
            raise ValueError("manifest items require exactly id, path and sha256")
        input_id = item["id"]
        if not isinstance(input_id, str) or not input_id or input_id in ids:
            raise ValueError("manifest IDs must be nonempty, unique strings")
        input_id.encode("utf-8")
        if not isinstance(item["path"], str) or not item["path"] or not valid_sha256(item["sha256"]):
            raise ValueError("manifest paths and SHA-256 pins are required")
        ids.add(input_id)
    return items


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def recognize_item(item, parent, recognize):
    path = parent / item["path"]
    try:
        if sha256_file(path) != item["sha256"]:
            return b"", "error", "input_sha256_mismatch"
        text = transport_text(recognize(path))
        data = text.encode("utf-8")
    except Exception as exc:
        # Backends use their own exception types for per-image inference failures.
        # Keep messages and stderr out of the output and preserve the ID.
        return b"", "error", type(exc).__name__
    return data, "blank" if not data or text.isspace() else "ok", None


def write_batch(args, recognize, provenance):
    if not (valid_sha256(args.runtime_sha256) and valid_sha256(args.config_sha256)):
        raise ValueError("runtime and configuration SHA-256 values must have 64 lowercase hex digits")
    manifest_bytes = args.manifest.read_bytes()
    items = read_manifest(manifest_bytes)
    pins = {
        "schema": SCHEMA,
        "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "runtime_sha256": args.runtime_sha256,
        "config_sha256": args.config_sha256,
        "mode": args.mode,
        **provenance,
    }
    sidecar = {**pins, "run_sha256": hashlib.sha256(json_bytes(pins)).hexdigest(), "results": []}
    output_dir = args.output_dir
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)

    with tempfile.TemporaryDirectory(prefix=".ocr-batch-", dir=output_dir.parent) as tmp:
        stage = Path(tmp) / "output"
        stage.mkdir(mode=0o700)
        if args.mode == "page":
            (stage / "pages").mkdir()
            for number, item in enumerate(items):
                data, status, error = recognize_item(item, args.manifest.parent, recognize)
                relative = f"pages/{number:08d}.txt"
                (stage / relative).write_bytes(data)
                sidecar["results"].append(result_metadata(item, data, status, error, file=relative))
        else:
            with (stage / "lines.jsonl").open("wb") as lines:
                for item in items:
                    data, status, error = recognize_item(item, args.manifest.parent, recognize)
                    text = data.decode("utf-8")
                    lines.write(json_bytes({"id": item["id"], "text": text}))
                    sidecar["results"].append(result_metadata(
                        item, data, status, error, unexpected_line_break="\n" in text.removesuffix("\n"),
                    ))
        (stage / "run.json").write_bytes(json_bytes(sidecar))
        stage.rename(output_dir)


def result_metadata(item, data, status, error, **fields):
    result = {"id": item["id"], "status": status, "text_sha256": hashlib.sha256(data).hexdigest(), **fields}
    if error is not None:
        result["error"] = error
    return result



def run_cli(args, parser, prepare, adapter_files, *, legacy_suffix=""):
    batch = batch_requested(args, parser)
    if args.mode is None or (not batch and not all((args.input, args.output))):
        parser.error("run requires --mode and either batch arguments or --input and --output")
    recognize, provenance = prepare(args.assets, args.model, args.mode)
    if batch:
        provenance["adapter_sha256"] = {path.name: sha256_file(path)
                                        for path in (*adapter_files, Path(__file__))}
        write_batch(args, recognize, provenance)
    else:
        data = transport_text(recognize(args.input)) + legacy_suffix
        args.output.write_bytes(data.encode("utf-8"))
