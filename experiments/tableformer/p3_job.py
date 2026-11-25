"""Compile one immutable autonomous P3 job without uploading or launching."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from assets import BASE_SHA, CONFIG, CONFIG_SHA, MODEL_SUBDIR, WEIGHT, digest
from heocr_compute.remote.hf_job import check_repo_scope, parse, safe_path


IMAGE = "ghcr.io/heocr/heocr-compute-train@sha256:2970c56ef7f946f602b3408e88679a080eb543056898e5c044396ba147cf2e75"
CODE = ("assets.py", "baseline.py", "targets.py", "p3.py", "train.py", "train-p3.sh")


def compile_job(pins, manifest_name, manifest_sha, clean_sha, archive_sha, transport, repo_scope):
    if Path(safe_path(manifest_name)).name != manifest_name or manifest_name == "clean.json":
        raise ValueError("P3 TRAIN manifest needs its own basename")
    by_role = {pin["target"]: pin for pin in pins}
    model = Path("models") / MODEL_SUBDIR
    weights = (model / WEIGHT).as_posix()
    expected = {name: digest(Path(__file__).with_name(name)) for name in CODE}
    expected |= {
        (model / CONFIG).as_posix(): CONFIG_SHA,
        weights: BASE_SHA,
        "transport.py": digest(transport),
        "train.tar": archive_sha,
    }
    if len(by_role) != len(pins) or by_role.keys() != expected.keys() or any(
        by_role[name]["sha256"] != sha for name, sha in expected.items()
    ):
        raise ValueError("P3 immutable job roles or reviewed byte pins differ")
    if any(len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha)
           for sha in (manifest_sha, clean_sha, archive_sha)):
        raise ValueError("P3 needs exact TRAIN and CLEAN SHA-256 pins")
    job = {
        "schema": "heocr_hf_training.v2", "image": IMAGE, "entrypoint": "train-p3.sh",
        "args": [manifest_name, manifest_sha, clean_sha], "clean_sha256": clean_sha,
        "inputs": sorted(pins, key=lambda pin: pin["target"]),
        "output": {"repo": repo_scope, "type": "dataset"},
        "snapshots": [{"name": WEIGHT, "max_bytes": by_role[weights]["size"] + 1024 * 1024}],
        "telemetry": {"metrics_max_bytes": 65536, "trainer_log_max_bytes": 1048576,
                      "run_log_max_bytes": 1048576, "setup_log_max_bytes": 1048576},
    }
    raw = (json.dumps(job, sort_keys=True, separators=(",", ":")) + "\n").encode()
    approved = sha256(raw).hexdigest()
    check_repo_scope(parse(raw, approved), repo_scope)
    return raw, approved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pins", "transport", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    for key in ("manifest-name", "manifest-sha256", "clean-sha256", "archive-sha256", "repo-scope"):
        parser.add_argument(f"--{key}", required=True)
    args = parser.parse_args()
    raw, approved = compile_job(
        json.loads(args.pins.read_bytes()), args.manifest_name, args.manifest_sha256,
        args.clean_sha256, args.archive_sha256, args.transport, args.repo_scope,
    )
    with args.output.open("xb") as output:
        output.write(raw)
    print(approved)


if __name__ == "__main__":
    main()
