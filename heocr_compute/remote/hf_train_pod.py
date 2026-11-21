"""Pod-local training supervisor for pinned private Hub inputs."""
from __future__ import annotations

import datetime as dt
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from . import hf_job
from .manifest import sha256_file


def token(name):
    value = os.environ.get(name, "")
    hf_job.require(value.startswith("hf_") and value[3:].isalnum(), "missing Hub environment secret")
    return value


def fetch_inputs(job, root, hub):
    revisions = {(item["repo"], item["type"], item["revision"]) for item in job["inputs"]}
    for repo, kind, revision in sorted(revisions):
        hub.private_revision({"repo": repo, "type": kind}, revision)
    for item in job["inputs"]:
        source = hub.download(item, item["path"], item["revision"], item["size"])
        hf_job.verify_file(source, item["size"], item["sha256"])
        path = root / item["target"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)


def publish_ready(job, out, staging, hub, prefix, receipt):
    uploaded = {record["name"] for record in receipt["snapshots"]}
    for spec in job["snapshots"]:
        name = spec["name"]
        path = out / name
        if name in uploaded or not (path.exists() or path.is_symlink()):
            continue
        hf_job.require(not path.is_symlink() and path.is_file(), "snapshot must be a regular file")
        size = path.stat().st_size
        hf_job.require(0 < size <= spec["max_bytes"], "snapshot exceeds its declared limit")
        frozen = staging / name
        shutil.copyfile(path, frozen)
        sha = sha256_file(frozen)
        hf_job.require(frozen.stat().st_size == size and sha256_file(path) == sha,
                       "snapshot was modified instead of atomically published by the trainer")
        entry = {"name": name, "size": size, "sha256": sha}
        updated = {**receipt, "snapshots": [*receipt["snapshots"], entry]}
        hub.commit(job["output"], prefix, updated, frozen)
        receipt["snapshots"].append(entry)
        frozen.unlink()


def train(job, root, hub, receipt):
    inp, out, staging = (root / part for part in ("in", "out", "staging"))
    for directory in (inp, out, staging):
        directory.mkdir(mode=0o700)
    prefix = f"runs/{receipt['run_id']}"
    hub.private_revision(job["output"])
    child = None
    try:
        fetch_inputs(job, inp, hub)
        # An approved entrypoint is trusted code. Keep provider and Hub credentials out of its env.
        names = ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
        env = {key: os.environ[key] for key in names if key in os.environ}
        with (root / "training.log").open("wb") as log:
            child = subprocess.Popen(["/bin/bash", str(inp / job["entrypoint"]), *job["args"]],
                                     cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                publish_ready(job, out, staging, hub, prefix, receipt)
                time.sleep(1)
            publish_ready(job, out, staging, hub, prefix, receipt)
            hf_job.require(child.returncode == 0 and len(receipt["snapshots"]) == len(job["snapshots"]),
                           "trainer failed or omitted a snapshot")
        receipt["status"] = "ok"
    except hf_job.PublishError:
        # A commit may have landed after its response was lost. Preserve that receipt.
        raise
    except (Exception, KeyboardInterrupt):
        receipt["status"] = "failed"
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
    receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    hub.commit(job["output"], prefix, receipt)
    return receipt["status"]


def interrupt(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, interrupt)
    try:
        os.umask(0o077)
        raw = os.environ.pop("HEOCR_HF_JOB").encode("utf-8")
        spec_sha = os.environ["HEOCR_HF_JOB_SHA256"]
        job = hf_job.parse(raw, spec_sha)
        repo_scope = os.environ.get("HEOCR_HF_REPO_SCOPE")
        if repo_scope is not None or os.environ.get("HEOCR_HF_REQUIRE_REPO_SCOPE") == "1":
            hf_job.check_repo_scope(job, repo_scope)
        root = Path("/heocr")
        root.mkdir(mode=0o700, exist_ok=True)
        run_id = os.environ["HEOCR_HF_RUN_ID"]
        hf_job.require(hf_job.matches(hf_job.RUN_ID, run_id), "invalid run ID")
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        receipt = {
            "schema": hf_job.RECEIPT, "run_id": run_id, "pod_id": os.environ["RUNPOD_POD_ID"],
            "job_sha256": spec_sha, "clean_sha256": job["clean_sha256"],
            "started_utc": started, "finished_utc": None, "status": "running", "snapshots": [],
        }
        hub = hf_job.Hub(token("HEOCR_HF_TOKEN"), root / "hub-cache")
        status = train(job, root, hub, receipt)
    except (Exception, KeyboardInterrupt):
        print("Training supervisor failed before a final receipt could be committed.", flush=True)
        return 1
    print(f"Training supervisor finished: {status}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
