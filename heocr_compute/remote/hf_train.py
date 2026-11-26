"""Launch an autonomous pod, then collect its private Hub results separately."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from . import budget, hf_job, r2_input, r2_output
from .api import RunPodError, load_hf_token
from .policy import Policy, Refused
from .runner import (DEFAULT_CAPACITY_WAIT_HOURS, HF_PROFILE_MARKER, Runner, new_run_id, observe_startup, pod_name,
                     reconcile_startup_cost, record_startup, terminate_confirmed)

KRAKEN_TRAIN_IMAGE = "ghcr.io/heocr/heocr-compute-kraken-train@"
STARTUP_POLL_SECONDS = 10


def check_hub_inputs(job):
    policy = Policy.load()
    for item in job["inputs"]:
        source = (f"r2/{r2_input.BUCKET}/{item['path']}" if item.get("transport") == "r2"
                  else f"hf/{item['repo']}/{item['path']}")
        for path in (source, f"/heocr/in/{item['target']}"):
            if policy.deny_reason(path):
                raise Refused("a training input matches the local denylist")
            policy.check_hash(path, item["sha256"])


def output_client(output):
    return r2_output.LocalR2(output["account"], bucket=output.get("bucket"))


def read_worker_startup(token, run_id, pod_id, reserve, *, r2=None):
    if hf_job.is_r2_output(reserve):
        r2 = r2 if r2 is not None else output_client(reserve["output"])
        with tempfile.TemporaryDirectory(prefix="heocr-r2-startup-") as cache:
            receipt, _ = read_r2_receipt(r2, run_id, reserve, Path(cache), startup_only=True)
        if receipt is None:
            return None
        validate_receipt(receipt, run_id, pod_id, reserve)
        return receipt["status"]
    output = reserve["output"]
    with tempfile.TemporaryDirectory(prefix="heocr-worker-receipt-") as cache:
        hub = hf_job.Hub(token, Path(cache))
        revision = hub.private_revision(output)
        path = hub.find_receipt(output, f"runs/{run_id}", revision)
        if path is None:
            return None
        receipt = json.loads(path.read_bytes())
        validate_receipt(receipt, run_id, pod_id, reserve)
        return receipt["status"]


def watch_startup(runner, run_id, pod_id, deadline, token=None, r2=None):
    """Monitor provisioning from the account side until the pod starts or the wait expires."""
    required = token is not None or r2 is not None
    reserve = runner.ledger.runs()[run_id]["reserve"] if required else None
    while True:
        expired = runner.now() >= deadline
        state = observe_startup(runner.api, pod_id)
        if state == "running":
            receipt_status = None if required else "running"
            if required:
                try:
                    if r2 is None:
                        receipt_status = read_worker_startup(token, run_id, pod_id, reserve)
                    else:
                        receipt_status = read_worker_startup(None, run_id, pod_id, reserve, r2=r2)
                except (OSError, ValueError):
                    print("Worker receipt is not yet readable and verified.", flush=True)
            if receipt_status == "failed":
                raise ValueError(f"pod {pod_id} published a failure receipt during startup; collect after teardown")
            if receipt_status in ("running", "ok"):
                record_startup(runner.ledger, run_id, pod_id, state=state)
                if required:
                    transport = "R2" if r2 is not None else "Hub"
                    print(f"Verified the pod worker's private {transport} startup receipt.")
                print(f"Pod {pod_id} will pull verified inputs and publish results without a local supervisor.")
                return 0
        if state == "gone":
            if required:
                raise ValueError(f"pod {pod_id} exited before a worker receipt was verified; collect the run")
            print(f"Pod {pod_id} already left the account. Use collect-hf to check for a receipt.")
            return 0
        if expired:
            break
        remaining = (deadline - runner.now()).total_seconds()
        if remaining > 0:
            runner.sleep(min(remaining, STARTUP_POLL_SECONDS))
    if state == "unknown":
        raise ValueError(f"pod {pod_id} startup could not be verified; check Runpod or run heocr-remote sweep")
    if not terminate_confirmed(runner.api, pod_id, sleep=runner.sleep):
        raise ValueError(f"pod {pod_id} missed its startup deadline and teardown was not confirmed")
    record_startup(runner.ledger, run_id, pod_id, state="startup_timeout")
    run = runner.ledger.runs()[run_id]
    if not reconcile_startup_cost(runner.api, runner.ledger, run, now=runner.now):
        print("Provider billing is pending. The estimate remains reserved until a later sweep.")
    if state == "running" and required:
        raise ValueError(f"pod {pod_id} was terminated after its startup deadline without a verified worker receipt")
    raise ValueError(f"pod {pod_id} was terminated after its startup deadline without a container runtime")


def launch(a, profile, api, ledger):
    try:
        with Path(a.job).open("rb") as file:
            raw = file.read(hf_job.MAX_SPEC + 1)
    except OSError:
        raise ValueError("could not read the approved training job file") from None
    job = hf_job.parse(raw, a.approved_sha256)
    repo_scope = getattr(a, "hf_repo_scope", None)
    public_repos = getattr(a, "hf_public_train_repo", ())
    r2_enabled = hf_job.is_r2_output(job)
    if r2_enabled:
        hf_job.require(repo_scope is not None and getattr(a, "require_worker_receipt", False),
                       "R2 outputs require --hf-repo-scope and --require-worker-receipt")
        hf_job.require(a.hours * 3600 < r2_input.MAX_LIFETIME, "R2 outputs require a wall time below 24 hours")
    if any(item.get("transport") == "r2" for item in job["inputs"]):
        hf_job.require(repo_scope is not None, "presigned R2 inputs require --hf-repo-scope")
    if job["image"].startswith(KRAKEN_TRAIN_IMAGE):
        hf_job.require(repo_scope is not None, "Kraken training image requires --hf-repo-scope")
        hf_job.require(profile.kind == "cpu", "Kraken training image requires a CPU profile")
    if repo_scope is not None or public_repos or any(item.get("transport") == hf_job.PUBLIC_TRAIN
                                                     for item in job["inputs"]):
        hf_job.check_repo_scope(job, repo_scope, public_repos)
    hf_job.require(profile.cloud == "SECURE", "training requires a Secure Cloud profile")
    hf_job.require(hf_job.matches(hf_job.SECRET, a.hf_secret), "supply the name of a Runpod secret")
    registry = getattr(a, "registry_auth_id", None)
    hf_job.require(registry is None or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", registry) is not None,
                   "supply a Runpod managed registry credential ID")
    caps = (a.run_cap,) if a.month_cap is None else (a.run_cap, a.month_cap)
    hf_job.require(all(math.isfinite(x) and x > 0 for x in (a.hours, *caps)),
                   "training time and cost caps must be finite and positive")
    startup_limit = budget.STARTUP_HOURS * 60
    startup_minutes = getattr(a, "startup_minutes", startup_limit)
    hf_job.require(math.isfinite(startup_minutes) and 0 < startup_minutes <= startup_limit,
                   "startup minutes must be positive and within the quoted startup allowance")
    total = sum(item["size"] for item in job["inputs"])
    outputs = sum(hf_job.output_limits(job).values())
    largest_group = max(sum(group.values()) for group in hf_job.checkpoint_groups(job))
    staging = max(largest_group, sum(hf_job.final_logs(job).values()))
    # Inputs are cached and copied. Keep space for one atomic output commit.
    hf_job.require(2 * total + outputs + staging < profile.disk_gb * 10**9,
                   "declared input and output bytes exceed the pod disk")
    check_hub_inputs(job)

    runner = Runner(api, ledger)
    require_worker_receipt = getattr(a, "require_worker_receipt", False)
    startup_token = None
    r2 = output_client(job["output"]) if r2_enabled else None
    if require_worker_receipt and r2 is None:
        hf_job.require(repo_scope is not None, "worker receipt verification requires --hf-repo-scope")
        hf_job.require("telemetry" in job, "worker receipt verification requires the telemetry job schema")
        startup_token = load_hf_token()
        with tempfile.TemporaryDirectory(prefix="heocr-worker-receipt-") as cache:
            try:
                hf_job.Hub(startup_token, Path(cache)).private_revision(job["output"])
            except ImportError:
                raise ValueError("install the Hub client with pip install '.[render]'") from None
    url_seconds = max(startup_minutes * 60, a.hours * 3600 if require_worker_receipt else 0)
    urls = r2_input.read_urls(job, os.environ.get(r2_input.ENV), now=runner.now(), min_remaining=url_seconds)
    run_id = new_run_id(runner.now())
    name = pod_name(HF_PROFILE_MARKER, run_id)
    spec = SimpleNamespace(profile=profile, hours=a.hours, run_cap=a.run_cap, month_cap=a.month_cap,
                           dry_run=a.dry_run, yes=a.yes,
                           capacity_wait_hours=getattr(a, "capacity_wait_hours", DEFAULT_CAPACITY_WAIT_HOURS))
    print(f"Pinned inputs: {len(job['inputs'])} files, {total:,} bytes. Job spec: {len(raw):,} bytes.")
    with runner.launch_slot(spec, run_id) as (quote, waited):
        if quote is None:
            return 0
        if a.dry_run:
            if r2 is not None:
                r2.put_urls(job, run_id, now=runner.now(), min_remaining=a.hours * 3600)
            print("Dry run. No pod created. The worker checks private Hub access before training.")
            return 0
        if waited:
            check_hub_inputs(job)
        reserved = runner.now()
        put_env = r2.put_urls(job, run_id, now=reserved, min_remaining=a.hours * 3600) if r2 is not None else None
        if urls:
            for item in job["inputs"]:
                if item.get("transport") == "r2":
                    r2_input.check_url(item, urls[item["target"]], now=reserved, min_remaining=url_seconds)
        deadline = reserved + dt.timedelta(hours=a.hours)
        startup_deadline = min(deadline, reserved + dt.timedelta(minutes=startup_minutes))
        telemetry = {"telemetry": job["telemetry"]} if "telemetry" in job else {}
        if require_worker_receipt:
            telemetry["worker_receipt_required"] = True
        ledger.reserve(run_id, profile=profile.name, pod_type=profile.pod_type, vcpus=profile.vcpus,
                       usd_est=quote.usd_est, hours_max=quote.hours, deadman_utc=deadline.isoformat(),
                       startup_deadline_utc=startup_deadline.isoformat(), autonomous=True,
                       output=job["output"], job_sha256=a.approved_sha256, clean_sha256=job["clean_sha256"],
                       snapshots=job["snapshots"], **telemetry)
    env = {
        "HEOCR_HF_JOB": raw.decode("utf-8"),
        "HEOCR_HF_JOB_SHA256": a.approved_sha256,
        "HEOCR_HF_RUN_ID": run_id,
        "HEOCR_HF_DEADLINE": str(int(deadline.timestamp())),
        "HEOCR_HF_TOKEN": "{{ RUNPOD_SECRET_" + a.hf_secret + " }}",
    }
    if repo_scope is not None:
        env["HEOCR_HF_REPO_SCOPE"] = repo_scope
    if public_repos:
        env["HEOCR_HF_PUBLIC_TRAIN_REPOS"] = " ".join(sorted(set(public_repos)))
    if urls:
        env[r2_input.ENV] = json.dumps(urls, separators=(",", ":"))
    if put_env is not None:
        env[r2_output.ENV] = put_env
    if "telemetry" in job:
        env["HEOCR_HF_FINALIZE_LOGS"] = "1"
    if require_worker_receipt:
        env["HEOCR_HF_PUBLISH_STARTUP"] = "1"
    body = profile.body(name, env)
    body.update(image=job["image"], ports=[], startSsh=False)
    if registry is not None:
        body["registry"] = registry
    print(f"Reserved run {run_id}. Collect later with heocr-remote collect-hf {run_id}.", flush=True)
    try:
        pod = api.create_pod(body)
    except (RunPodError, OSError) as exc:
        # Avoid duplicate pods and do not echo provider errors that might include environment settings.
        if getattr(exc, "status", None) in (400, 401, 402, 403, 404, 422, 429):
            ledger.final(run_id, pod_type=profile.pod_type, hours=0, usd=0, status="refused")
        raise ValueError("pod create did not confirm a pod; reconcile the named reservation before retrying") from None
    pod_id = pod.get("id")
    hf_job.require(hf_job.matches(hf_job.POD_ID, pod_id),
                   "pod create gave no ID; reconcile the named reservation")
    rate = pod.get("cost")
    ledger.pod(run_id, pod_id=pod_id, usd_per_hr=rate, created_utc=pod.get("createdAt"))
    if type(rate) not in (float, int) or not math.isfinite(rate) or not 0 < rate <= quote.compute_usd_per_hour:
        terminate_confirmed(api, pod_id)
        raise ValueError("unverified or higher billed rate; reconcile the pod and reservation")
    print(f"Pod {pod_id} created. Waiting for container runtime until {startup_deadline.isoformat()}.", flush=True)
    return watch_startup(runner, run_id, pod_id, startup_deadline, token=startup_token, r2=r2)


def validate_receipt(receipt, run_id, pod_id, reserve):
    hf_job.fields(receipt, "schema run_id pod_id job_sha256 clean_sha256 started_utc finished_utc status snapshots")
    telemetry = "telemetry" in reserve
    schema = hf_job.TELEMETRY_RECEIPT if telemetry else hf_job.RECEIPT
    hf_job.require(receipt["schema"] == schema and receipt["run_id"] == run_id
                   and receipt["pod_id"] == pod_id
                   and receipt["job_sha256"] == reserve["job_sha256"]
                   and receipt["clean_sha256"] == reserve["clean_sha256"]
                   and receipt["status"] in ("running", "ok", "failed")
                   and isinstance(receipt["snapshots"], list), "run receipt does not match the reservation")
    limits = hf_job.output_limits(reserve)
    names = set()
    for entry in receipt["snapshots"]:
        hf_job.fields(entry, "name size sha256")
        name = hf_job.safe_path(entry["name"])
        hf_job.require(name in limits and name not in names and hf_job.size_ok(entry["size"])
                       and entry["size"] <= limits[name] and hf_job.matches(hf_job.SHA, entry["sha256"]),
                       "unlisted or oversized output in receipt")
        names.add(name)
    required = {name for group in hf_job.checkpoint_groups(reserve) for name in group}
    if telemetry:
        for group in hf_job.checkpoint_groups(reserve):
            hf_job.require(not names.intersection(group) or group.keys() <= names,
                           "receipt contains an incomplete checkpoint group")
        hf_job.require(receipt["status"] != "running" or not names.intersection(hf_job.final_logs(reserve)),
                       "running receipt contains final logs")
        required.add("training.log")
    if receipt["status"] == "ok":
        hf_job.require(required <= names and "partial.trainer.log" not in names,
                       "successful receipt is missing committed outputs or contains a partial log")


def materialize(hub, output, prefix, revision, receipt, destination, cache):
    def fetch(name, size):
        return hub.download(output, f"{prefix}/{name}", revision, size)

    materialize_from(fetch, receipt, destination, cache)


def materialize_from(fetch, receipt, destination, cache):
    if destination.exists():
        hf_job.require(json.loads((destination / "receipt.json").read_bytes()) == receipt,
                       "existing result directory differs from the verified receipt")
        for entry in receipt["snapshots"]:
            hf_job.verify_file(destination / entry["name"], entry["size"], entry["sha256"])
        return
    staging = cache / "result"
    staging.mkdir(mode=0o700)
    for entry in receipt["snapshots"]:
        name, size = entry["name"], entry["size"]
        path = fetch(name, size)
        hf_job.verify_file(path, size, entry["sha256"])
        shutil.copyfile(path, staging / name)
    (staging / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    staging.rename(destination)


def read_r2_receipt(r2, run_id, reserve, cache, *, startup_only=False):
    names = [r2_output.TERMINAL, r2_output.STARTUP]
    if not startup_only:
        names[1:1] = [r2_output.progress_name(i) for i in range(len(reserve["snapshots"]), 0, -1)]
    for name in names:
        path = r2.fetch(run_id, name, cache / "r2-receipt.json", r2_output.RECEIPT_LIMIT, missing_ok=True)
        if path is not None:
            receipt = json.loads(path.read_bytes())
            hf_job.require(isinstance(receipt, dict)
                           and (receipt.get("status") == "running") == (name != r2_output.TERMINAL),
                           "R2 receipt status does not match its object key")
            return receipt, r2_output.object_key(run_id, name)
    return None, None


def settle_collection(api, ledger, run_id, run, pod_id, revision, receipt):
    old = run.get("final")
    if old and (old["status"] != "missing_receipt" or receipt is None):
        return
    reserve, pod = run["reserve"], run.get("pod", {})
    start = pod.get("created_utc") or pod.get("date") or reserve["date"]
    now = dt.datetime.now(dt.timezone.utc)
    usd = api.billed_usd(pod_id, start, now.isoformat())
    if usd is None or not math.isfinite(usd) or usd <= 0:
        print("Provider billing is pending. The estimate remains reserved.")
        return
    since = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
    finished = receipt and receipt["finished_utc"]
    if finished:
        finish = dt.datetime.fromisoformat(finished)
    else:
        finish = min(now, dt.datetime.fromisoformat(reserve["deadman_utc"]))
    status = "missing_receipt" if receipt is None else receipt["status"]
    location = {"r2_receipt_key": revision} if hf_job.is_r2_output(reserve) else {"hub_revision": revision}
    with ledger.locked():
        if ledger.runs()[run_id].get("final") == old:
            ledger.final(run_id, pod_type=reserve["pod_type"],
                         hours=max(0, (finish - since).total_seconds() / 3600), usd=usd,
                         status="incomplete" if status == "running" else status, pod_id=pod_id,
                         **location, usd_source="provider_billing",
                         hours_source="worker_receipt" if finished else "observation_or_deadline_bound")


def collect(a, api, ledger):
    run_id = a.run_id
    hf_job.require(hf_job.matches(hf_job.RUN_ID, run_id), "invalid run ID")
    run = ledger.runs().get(run_id, {})
    reserve = run.get("reserve", {})
    hf_job.require(reserve.get("autonomous") and not reserve.get("r2_archive") and reserve.get("account") == ledger.account,
                   "unknown autonomous reservation for this account")
    pod_record = run.get("pod", {})
    pod_id = pod_record.get("pod_id")
    if not pod_id:
        name = pod_name(HF_PROFILE_MARKER, run_id)
        live = [x for x in api.list_pods() if x.get("name") == name]
        if live:
            print(f"Run {run_id} still has a pod; collect after teardown.")
            return 1
    pod = api.get_pod(pod_id) if pod_id else None
    if pod and pod.get("status") != "TERMINATED":
        print(f"Pod {pod_id} still exists; collect after teardown.")
        return 1

    results = Path(a.results).expanduser()
    results.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hf-collect-", dir=results) as temp:
        root = Path(temp)
        output = reserve["output"]
        prefix = f"runs/{run_id}"
        r2 = None
        if hf_job.is_r2_output(reserve):
            r2 = output_client(output)
            receipt, revision = read_r2_receipt(r2, run_id, reserve, root)
        else:
            try:
                hub = hf_job.Hub(load_hf_token(), root / "cache")
            except ImportError:
                raise ValueError("install the Hub client with pip install '.[render]'") from None
            revision = hub.private_revision(output)
            data = hub.find_receipt(output, prefix, revision)
            receipt = json.loads(data.read_bytes()) if data is not None else None
        if receipt is None:
            hf_job.require(pod_id is not None, "no private receipt and no confirmed pod ID for the reservation")
            settle_collection(api, ledger, run_id, run, pod_id, revision, None)
            source = "private R2" if r2 is not None else f"verified private Hub revision {revision}"
            print(f"Pod {pod_id} is gone and no receipt was found in {source}.")
            return 1
        if not pod_id:
            pod_id = receipt.get("pod_id")
            hf_job.require(hf_job.matches(hf_job.POD_ID, pod_id),
                           "private receipt has no valid pod ID for the ambiguous create")
            pod = api.get_pod(pod_id)
            hf_job.require(not pod or pod.get("status") == "TERMINATED",
                           "recovered pod still exists; collect after teardown")
        validate_receipt(receipt, run_id, pod_id, reserve)
        dest = results / run_id
        if r2 is not None:
            def fetch(name, size):
                return r2.fetch(run_id, name, root / name, size)

            materialize_from(fetch, receipt, dest, root)
        else:
            materialize(hub, output, prefix, revision, receipt, dest, root)
    settle_collection(api, ledger, run_id, run, pod_id, revision, receipt)
    print(f"Collected {len(receipt['snapshots'])} verified outputs from {revision} into {dest}.")
    return 0 if receipt["status"] == "ok" else 1
