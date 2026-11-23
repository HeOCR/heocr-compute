"""Pod-local training supervisor for pinned private Hub inputs."""
from __future__ import annotations

import datetime as dt
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from . import hf_job, r2_input
from .manifest import sha256_file


def token(name):
    value = os.environ.get(name, "")
    hf_job.require(value.startswith("hf_") and value[3:].isalnum(), "missing Hub environment secret")
    return value


def fetch_inputs(job, root, hub, progress=lambda _: None):
    progress("Validating presigned input URLs.")
    urls = r2_input.read_urls(job, os.environ.pop(r2_input.ENV, None))
    hub_pins = [item for item in job["inputs"] if item.get("transport") != "r2"]
    revisions = {(item["repo"], item["type"], item["revision"]) for item in hub_pins}
    for repo, kind, revision in sorted(revisions):
        progress("Verifying a private Hub input revision.")
        hub.private_revision({"repo": repo, "type": kind}, revision)
    for item in job["inputs"]:
        path = root / item["target"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if item.get("transport") == "r2":
            progress("Downloading a pinned R2 input.")
            r2_input.download(item, urls[item["target"]], path)
        else:
            progress("Downloading a pinned private Hub input.")
            source = hub.download(item, item["path"], item["revision"], item["size"])
            hf_job.verify_file(source, item["size"], item["sha256"])
            shutil.copyfile(source, path)


def freeze_output(path, staging, limit):
    hf_job.require(not path.is_symlink() and path.is_file(), "output must be a regular file")
    size = path.stat().st_size
    hf_job.require(0 < size <= limit, "output exceeds its declared limit or is empty")
    frozen = staging / path.name
    shutil.copyfile(path, frozen)
    sha = sha256_file(frozen)
    hf_job.require(frozen.stat().st_size == size and sha256_file(path) == sha,
                   "output changed while being staged for publication")
    return frozen, {"name": path.name, "size": size, "sha256": sha}


def present(path):
    return path.exists() or path.is_symlink()


def publish_ready(job, out, staging, hub, prefix, receipt):
    uploaded = {record["name"] for record in receipt["snapshots"]}
    for group in hf_job.checkpoint_groups(job):
        if uploaded.intersection(group) or not all(present(out / name) for name in group):
            continue
        files, entries = zip(*(freeze_output(out / name, staging, limit) for name, limit in group.items()))
        updated = {**receipt, "snapshots": [*receipt["snapshots"], *entries]}
        hub.commit(job["output"], prefix, updated, files)
        receipt["snapshots"].extend(entries)
        for frozen in files:
            frozen.unlink()


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=5)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def train(job, root, hub, receipt, *, publish_startup=False):
    inp, out, staging = (root / part for part in ("in", "out", "staging"))
    for directory in (inp, out, staging):
        directory.mkdir(mode=0o700)
    prefix = f"runs/{receipt['run_id']}"
    hub.private_revision(job["output"])
    if publish_startup:
        print("Publishing private Hub startup receipt.", flush=True)
        hub.commit(job["output"], prefix, receipt)
    child = None
    try:
        # An approved entrypoint is trusted code. Keep provider and Hub credentials out of its env.
        names = ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
        env = {key: os.environ[key] for key in names if key in os.environ}
        env["HEOCR_HF_RUN_ID"] = receipt["run_id"]
        if "telemetry" not in job:
            fetch_inputs(job, inp, hub)
        with (root / "training.log").open("wb") as log:
            def progress(message):
                if "telemetry" in job:
                    log.write((message + "\n").encode("ascii"))
                    log.flush()
                if publish_startup:
                    print(message, flush=True)

            if "telemetry" in job:
                fetch_inputs(job, inp, hub, progress)
                progress("Starting training entrypoint.")
            child = subprocess.Popen(["/bin/bash", str(inp / job["entrypoint"]), *job["args"]],
                                     cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                publish_ready(job, out, staging, hub, prefix, receipt)
                time.sleep(1)
            publish_ready(job, out, staging, hub, prefix, receipt)
            expected = sum(len(group) for group in hf_job.checkpoint_groups(job))
            hf_job.require(child.returncode == 0 and len(receipt["snapshots"]) == expected,
                           "trainer failed or omitted a checkpoint file")
        receipt["status"] = "ok"
    except hf_job.PublishError:
        # A commit may have landed after its response was lost. Preserve that receipt.
        raise
    except (Exception, KeyboardInterrupt):
        receipt["status"] = "failed"
    finally:
        stop_child(child)
    receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    logs = []
    for name, limit in hf_job.final_logs(job).items():
        path = (root if name == "training.log" else out) / name
        if not present(path):
            if name == "training.log":
                receipt["status"] = "failed"
            continue
        if name == "partial.trainer.log":
            receipt["status"] = "failed"
        try:
            frozen, record = freeze_output(path, staging, limit)
        except ValueError:
            receipt["status"] = "failed"
            continue
        logs.append(frozen)
        receipt["snapshots"].append(record)
    if publish_startup:
        print(f"Publishing terminal {receipt['status']} receipt and available logs.", flush=True)
    hub.commit(job["output"], prefix, receipt, logs)
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
        if (repo_scope is not None or os.environ.get("HEOCR_HF_REQUIRE_REPO_SCOPE") == "1"
                or any(item.get("transport") == "r2" for item in job["inputs"])):
            hf_job.check_repo_scope(job, repo_scope)
        root = Path("/heocr")
        root.mkdir(mode=0o700, exist_ok=True)
        run_id = os.environ["HEOCR_HF_RUN_ID"]
        hf_job.require(hf_job.matches(hf_job.RUN_ID, run_id), "invalid run ID")
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        receipt = {
            "schema": hf_job.TELEMETRY_RECEIPT if "telemetry" in job else hf_job.RECEIPT,
            "run_id": run_id, "pod_id": os.environ["RUNPOD_POD_ID"],
            "job_sha256": spec_sha, "clean_sha256": job["clean_sha256"],
            "started_utc": started, "finished_utc": None, "status": "running", "snapshots": [],
        }
        hub = hf_job.Hub(token("HEOCR_HF_TOKEN"), root / "hub-cache")
        status = train(job, root, hub, receipt, publish_startup=os.environ.pop("HEOCR_HF_PUBLISH_STARTUP", "") == "1")
    except hf_job.PublishError as exc:
        # PublishError is generated by the Hub wrapper and contains no remote response body.
        print(str(exc), flush=True)
        return 1
    except (Exception, KeyboardInterrupt):
        print("Training supervisor failed before a final receipt could be committed.", flush=True)
        return 1
    print(f"Training supervisor finished: {status}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
