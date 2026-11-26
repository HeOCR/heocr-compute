"""Pinned predictions-only Paddle GPU transport and account-checked launch."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from . import budget, hf_job, profiles, r2_input, r2_output
from .api import RunPodError
from .cli import _connect
from .hf_train import watch_startup
from .runner import HF_PROFILE_MARKER, Runner, new_run_id, pod_name, terminate_confirmed

JOB_ENV = "HEOCR_PADDLE_JOB"
PIN_ENV = "HEOCR_PADDLE_JOB_SHA256"
GET_ENV = "HEOCR_PADDLE_R2_GET"
PUT_ENV = "HEOCR_PADDLE_R2_PUT"
IMAGE = re.compile(r"ghcr\.io/heocr/heocr-compute-comparator-paddle-cuda@sha256:[a-f0-9]{64}")
BUCKET = re.compile(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]")
RASTER = re.compile(r"rasters/[A-Za-z0-9_-]+\.(png|jpg|jpeg)")
MAX_ARCHIVE = 128 * 1024 * 1024
MAX_OUTPUT = 32 * 1024 * 1024


def parse(raw, pin):
    hf_job.require(len(raw) <= 8192 and hf_job.digest(raw) == pin, "Paddle job differs from approval pin")
    job = json.loads(raw)
    hf_job.fields(job, "schema split image authority_sha256 dev_receipt_sha256 manifest_sha256 count input output")
    hf_job.require(job["schema"] == "heocr.paddle_gpu_predictions.v1" and job["split"] in {"DEV", "SEALED"}
                   and hf_job.matches(IMAGE, job["image"]), "expected pinned Paddle CUDA inference job")
    for field in ("authority_sha256", "manifest_sha256"):
        hf_job.require(hf_job.matches(hf_job.SHA, job[field]), "expected exact inference authority and manifest pins")
    dev = job["split"] == "DEV"
    hf_job.require((job["dev_receipt_sha256"] is None if dev else hf_job.matches(hf_job.SHA, job["dev_receipt_sha256"]))
                   and type(job["count"]) is int and 0 < job["count"] <= 512,
                   "expected exact roster count and CUDA DEV receipt for sealed inference")
    src, dst = job["input"], job["output"]
    hf_job.fields(src, "account bucket path size sha256")
    hf_job.fields(dst, "account bucket path max_bytes")
    for entry in (src, dst):
        hf_job.require(hf_job.matches(hf_job.R2_ACCOUNT, entry["account"])
                       and hf_job.matches(BUCKET, entry["bucket"]) and entry["bucket"] != r2_input.BUCKET,
                       "expected an isolated private inference bucket")
        hf_job.require(hf_job.safe_path(entry["path"]).startswith("inference-runs/"), "expected isolated inference object")
    hf_job.require((src["account"], src["bucket"]) == (dst["account"], dst["bucket"])
                   and src["path"].endswith(".tar") and dst["path"].endswith("/predictions.jsonl")
                   and type(src["size"]) is int and 0 < src["size"] <= MAX_ARCHIVE
                   and type(dst["max_bytes"]) is int and 0 < dst["max_bytes"] <= MAX_OUTPUT
                   and hf_job.matches(hf_job.SHA, src["sha256"]), "invalid Paddle inference archive or output pin")
    return job


def validate_urls(job, get_url, put_url, now, remaining):
    src, dst = job["input"], job["output"]
    r2_input.check_url(src, get_url, now=now, min_remaining=remaining, bucket=src["bucket"])
    r2_input.check_url(dst, put_url, now=now, min_remaining=remaining, bucket=dst["bucket"],
                       method="PUT", signed_headers=r2_output.SIGNED_HEADERS)


def materialize(archive, root, manifest_sha, count):
    """Accept only the pinned manifest and its exact raster members."""
    with tarfile.open(archive, "r:") as tar:
        entries = {}
        for entry in tar:
            hf_job.require(entry.name not in entries and len(entries) < count + 2,
                           "archive has repeated or surplus members")
            entries[entry.name] = entry
        directory = entries.pop("rasters", None)
        hf_job.require(directory is None or directory.isdir(), "archive raster directory has invalid type")
        manifest = entries.pop("manifest.json", None)
        hf_job.require(manifest is not None and manifest.isfile() and 0 < manifest.size <= 512 * 1024,
                       "archive has no bounded raster manifest")
        with tar.extractfile(manifest) as source:
            raw = source.read()
        hf_job.require(hf_job.digest(raw) == manifest_sha, "archive manifest digest differs")
        rows = json.loads(raw)
        hf_job.require(isinstance(rows, list) and len(rows) == count, "archive raster count differs")
        ids, paths = set(), set()
        for row in rows:
            hf_job.fields(row, "id path sha256")
            name = row["path"]
            hf_job.require(hf_job.matches(RASTER, name) and name not in paths, "invalid or repeated raster path")
            entry = entries.get(name)
            hf_job.require(isinstance(row["id"], str) and row["id"] and row["id"] not in ids
                           and hf_job.matches(hf_job.SHA, row["sha256"]) and entry is not None
                           and entry.isfile() and 0 < entry.size <= MAX_ARCHIVE, "invalid raster or duplicate frozen identity")
            ids.add(row["id"])
            paths.add(name)
        hf_job.require(paths == entries.keys(), "archive contains a file outside the raster manifest")
        (root / "rasters").mkdir(mode=0o700)
        for row in rows:
            path = root / row["path"]
            entry = entries[row["path"]]
            with tar.extractfile(entry) as source, path.open("xb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            hf_job.verify_file(path, entry.size, row["sha256"])
        (root / "manifest.json").write_bytes(raw)
    return [row["id"] for row in rows]


def publish(job, root, ids, put_url):
    path = root / "predictions.jsonl"
    hf_job.require(0 < path.stat().st_size <= job["output"]["max_bytes"], "invalid predictions size")
    with path.open("rb") as stream:
        for item, expected in zip(stream, ids, strict=True):
            row = json.loads(item)
            hf_job.fields(row, "id text")
            hf_job.require(row["id"] == expected and isinstance(row["text"], str), "predictions do not match frozen order")
        stream.seek(0)
        publisher = r2_output.Publisher({"predictions.jsonl": put_url}, {"snapshots": []})
        publisher.put("predictions.jsonl", stream, path.stat().st_size)


def worker():
    try:
        os.umask(0o077)
        job = parse(os.environ.pop(JOB_ENV).encode(), os.environ.pop(PIN_ENV))
        get_url, put_url = os.environ.pop(GET_ENV), os.environ.pop(PUT_ENV)
        now = dt.datetime.now(dt.timezone.utc)
        remaining = int(os.environ["HEOCR_HF_DEADLINE"]) - now.timestamp()
        hf_job.require(remaining > 0, "Paddle inference deadline has elapsed")
        validate_urls(job, get_url, put_url, now, remaining)
        with TemporaryDirectory(prefix="paddle-cuda-") as tmp:
            root = Path(tmp)
            assets = root / "assets"
            names = ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                     "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            env = {name: os.environ[name] for name in names if name in os.environ}
            process = dict(env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True)
            subprocess.run(["/opt/paddle/bin/python", "-B", "/opt/comparators/runner.py", "fetch", "--model", "paddle",
                            "--assets", str(assets)], **process)
            archive = root / "input.tar"
            r2_input.download(job["input"], get_url, archive)
            ids = materialize(archive, root, job["manifest_sha256"], job["count"])
            archive.unlink()
            subprocess.run(["/opt/paddle/bin/python", "-B", "/opt/comparators/paddle_cuda.py",
                            "--manifest", str(root / "manifest.json"), "--manifest-sha256", job["manifest_sha256"],
                            "--assets", str(assets), "--output", str(root / "predictions.jsonl")], **process)
            publish(job, root, ids, put_url)
        print("Paddle CUDA completed the ordered predictions upload.", flush=True)
        return 0
    except (Exception, KeyboardInterrupt):
        print("Paddle CUDA inference failed before confirmed publication.", flush=True)
        return 1


def launch(args):
    raw = args.job.read_bytes()
    job = parse(raw, args.approved_sha256)
    minutes = 6 if job["split"] == "DEV" else 15
    profile = profiles.PROFILES["gpu"]
    api, ledger = _connect()
    runner = Runner(api, ledger)
    run_id = new_run_id(runner.now())
    spec = SimpleNamespace(profile=profile, hours=minutes / 60, run_cap=args.run_cap,
                           month_cap=None, dry_run=not args.launch, yes=args.launch,
                           capacity_wait_hours=None)
    with runner.launch_slot(spec, run_id) as (quote, _):
        if quote is None:
            return 0
        now = runner.now()
        deadline = now + dt.timedelta(hours=quote.hours)
        startup_deadline = now + dt.timedelta(hours=budget.STARTUP_HOURS)
        get_url, put_url = os.environ.pop(GET_ENV, None), os.environ.pop(PUT_ENV, None)
        if not args.launch and get_url is None and put_url is None:
            print("Dry run verified the job and quote. Presigned transport was not supplied.")
            return 0
        validate_urls(job, get_url, put_url, now, quote.hours * 3600)
        if not args.launch:
            print("Dry run verified the job, quote and exact-object URL scope. No pod created.")
            return 0
        hf_job.require(not any(run.get("reserve", {}).get("job_sha256") == args.approved_sha256
                               for run in ledger.runs().values()), "this approved Paddle job already has a reservation")
        ledger.reserve(run_id, profile=profile.name, pod_type=profile.pod_type,
                       usd_est=quote.usd_est, hours_max=quote.hours, deadman_utc=deadline.isoformat(),
                       startup_deadline_utc=startup_deadline.isoformat(), autonomous=True,
                       job_sha256=args.approved_sha256, purpose="paddle_cuda_inference")
    env = {JOB_ENV: raw.decode(), PIN_ENV: args.approved_sha256, GET_ENV: get_url, PUT_ENV: put_url,
           "HEOCR_HF_DEADLINE": str(int(deadline.timestamp()))}
    body = profile.body(pod_name(HF_PROFILE_MARKER, run_id), env)
    body.update(image=job["image"], ports=[], startSsh=False)
    print(f"Reserved bounded Paddle CUDA run {run_id} through {deadline.isoformat()}.", flush=True)
    try:
        pod = api.create_pod(body)
    except (RunPodError, OSError, ValueError, RuntimeError) as exc:
        if getattr(exc, "status", None) in (400, 401, 402, 403, 404, 422, 429):
            ledger.final(run_id, pod_type=profile.pod_type, hours=0, usd=0, status="refused")
        raise ValueError("Paddle pod create unconfirmed; reconcile the reservation before retrying") from None
    pod_id, rate = pod.get("id"), pod.get("cost")
    hf_job.require(hf_job.matches(hf_job.POD_ID, pod_id), "Paddle pod create did not confirm an ID")
    ledger.pod(run_id, pod_id=pod_id, usd_per_hr=rate, created_utc=pod.get("createdAt"))
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= quote.compute_usd_per_hour:
        terminate_confirmed(api, pod_id)
        raise ValueError("unverified or higher billed rate; reconcile Paddle run")
    print(f"Paddle CUDA pod {pod_id} created. Startup deadline {startup_deadline.isoformat()}.", flush=True)
    return watch_startup(runner, run_id, pod_id, startup_deadline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--approved-sha256")
    parser.add_argument("--run-cap", type=float)
    parser.add_argument("--launch", action="store_true", help="create the exact approved pod; default is dry run")
    args = parser.parse_args()
    if args.worker:
        return worker()
    if (args.job is None or not hf_job.matches(hf_job.SHA, args.approved_sha256) or args.run_cap is None
            or not math.isfinite(args.run_cap) or args.run_cap <= 0):
        parser.error("use --job, --approved-sha256 and the approved positive --run-cap")
    try:
        return launch(args)
    except (OSError, ValueError, budget.BudgetRefused) as exc:
        parser.exit(1, f"Paddle admission failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
