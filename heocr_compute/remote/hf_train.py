"""Launch an autonomous pod, then collect its private Hub results separately."""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from . import budget, hf_job
from .api import RunPodError, load_hf_token
from .policy import Policy, Refused
from .runner import Runner, new_run_id, pod_name, terminate_confirmed

PROFILE_MARKER = "hf"


def launch(a, profile, api, ledger):
    try:
        with Path(a.job).open("rb") as file:
            raw = file.read(hf_job.MAX_SPEC + 1)
    except OSError:
        raise ValueError("could not read the approved training job file") from None
    job = hf_job.parse(raw, a.approved_sha256)
    hf_job.require(profile.cloud == "SECURE", "training requires a Secure Cloud profile")
    hf_job.require(hf_job.matches(hf_job.SECRET, a.hf_secret), "supply the name of a Runpod secret")
    registry = getattr(a, "registry_auth_id", None)
    hf_job.require(registry is None or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", registry) is not None,
                   "supply a Runpod managed registry credential ID")
    hf_job.require(all(math.isfinite(x) and x > 0 for x in (a.hours, a.run_cap, a.month_cap)),
                   "training time and cost caps must be finite and positive")
    policy = Policy.load()
    total = sum(item["size"] for item in job["inputs"])
    largest_snapshot = max(item["max_bytes"] for item in job["snapshots"])
    # Inputs are cached and copied. Keep space for the staging copy of one snapshot.
    hf_job.require(2 * total + sum(s["max_bytes"] for s in job["snapshots"]) + largest_snapshot
                   < profile.disk_gb * 10**9, "declared input and snapshot bytes exceed the pod disk")
    for item in job["inputs"]:
        for path in (f"hf/{item['repo']}/{item['path']}", f"/heocr/in/{item['target']}"):
            if policy.deny_reason(path):
                raise Refused("a Hub input matches the local denylist")
            policy.check_hash(path, item["sha256"])

    runner = Runner(api, ledger)
    run_id = new_run_id(runner.now())
    name = pod_name(PROFILE_MARKER, run_id)
    spec = SimpleNamespace(profile=profile, hours=a.hours, run_cap=a.run_cap, month_cap=a.month_cap)
    with ledger.locked():
        quote = runner.price(spec, ledger.month_to_date()["total"])
        for line in quote.lines():
            print(line)
        budget.check(quote)
        print(f"Hub inputs: {len(job['inputs'])} files, {total:,} bytes. Job spec: {len(raw):,} bytes.")
        if a.dry_run:
            print("Dry run. No pod created. The worker checks private Hub access before training.")
            return 0
        if not a.yes and input(f"Start {profile.pod_type}, worst case USD {quote.usd_est:.2f}? [y/N] ").lower().strip() != "y":
            return 0
        deadline = runner.now() + dt.timedelta(hours=a.hours)
        ledger.reserve(run_id, profile=profile.name, pod_type=profile.pod_type, usd_est=quote.usd_est,
                       hours_max=quote.hours, deadman_utc=deadline.isoformat(), autonomous=True,
                       output=job["output"], job_sha256=a.approved_sha256, clean_sha256=job["clean_sha256"],
                       snapshots=job["snapshots"])
    env = {
        "HEOCR_HF_JOB": raw.decode("utf-8"),
        "HEOCR_HF_JOB_SHA256": a.approved_sha256,
        "HEOCR_HF_RUN_ID": run_id,
        "HEOCR_HF_DEADLINE": str(int(deadline.timestamp())),
        "HEOCR_HF_TOKEN": "{{ RUNPOD_SECRET_" + a.hf_secret + " }}",
    }
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
    print(f"Pod {pod_id} will pull verified inputs and publish results without a local supervisor.")
    return 0


def validate_receipt(receipt, run_id, pod_id, reserve):
    hf_job.fields(receipt, "schema run_id pod_id job_sha256 clean_sha256 started_utc finished_utc status snapshots")
    hf_job.require(receipt["schema"] == hf_job.RECEIPT and receipt["run_id"] == run_id
                   and receipt["pod_id"] == pod_id
                   and receipt["job_sha256"] == reserve["job_sha256"]
                   and receipt["clean_sha256"] == reserve["clean_sha256"]
                   and receipt["status"] in ("running", "ok", "failed")
                   and isinstance(receipt["snapshots"], list), "run receipt does not match the reservation")
    limits = {s["name"]: s["max_bytes"] for s in reserve["snapshots"]}
    names = set()
    for entry in receipt["snapshots"]:
        hf_job.fields(entry, "name size sha256")
        name = hf_job.safe_path(entry["name"])
        hf_job.require(name in limits and name not in names and hf_job.size_ok(entry["size"])
                       and entry["size"] <= limits[name] and hf_job.matches(hf_job.SHA, entry["sha256"]),
                       "unlisted or oversized snapshot in receipt")
        names.add(name)
    if receipt["status"] == "ok":
        hf_job.require(names == limits.keys(), "successful receipt is missing snapshots")


def materialize(hub, output, prefix, revision, receipt, destination, cache):
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
        path = hub.download(output, f"{prefix}/{name}", revision, size)
        hf_job.verify_file(path, size, entry["sha256"])
        shutil.copyfile(path, staging / name)
    (staging / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    staging.rename(destination)


def collect(a, api, ledger):
    run_id = a.run_id
    hf_job.require(hf_job.matches(hf_job.RUN_ID, run_id), "invalid run ID")
    run = ledger.runs().get(run_id, {})
    reserve = run.get("reserve", {})
    hf_job.require(reserve.get("autonomous") and reserve.get("account") == ledger.account,
                   "unknown autonomous reservation for this account")
    pod_record = run.get("pod", {})
    pod_id = pod_record.get("pod_id")
    if not pod_id:
        name = pod_name(PROFILE_MARKER, run_id)
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
        try:
            hub = hf_job.Hub(load_hf_token(), root / "cache")
        except ImportError:
            raise ValueError("install the Hub client with pip install '.[render]'") from None
        output = reserve["output"]
        revision = hub.private_revision(output)
        prefix = f"runs/{run_id}"
        data = hub.download(output, f"{prefix}/receipt.json", revision, hf_job.MAX_SPEC)
        receipt = json.loads(data.read_bytes())
        if not pod_id:
            pod_id = receipt.get("pod_id")
            hf_job.require(hf_job.matches(hf_job.POD_ID, pod_id),
                           "private receipt has no valid pod ID for the ambiguous create")
            pod = api.get_pod(pod_id)
            hf_job.require(not pod or pod.get("status") == "TERMINATED",
                           "recovered pod still exists; collect after teardown")
        validate_receipt(receipt, run_id, pod_id, reserve)
        dest = results / run_id
        materialize(hub, output, prefix, revision, receipt, dest, root)
    if "final" not in run:
        start = pod_record.get("created_utc") or pod_record.get("date") or reserve["date"]
        end = dt.datetime.now(dt.timezone.utc).isoformat()
        usd = api.billed_usd(pod_id, start, end)
        if usd is not None and math.isfinite(usd) and usd > 0:
            since = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
            finish = receipt["finished_utc"] or reserve["deadman_utc"]
            hours = max(0, (dt.datetime.fromisoformat(finish) - since).total_seconds() / 3600)
            with ledger.locked():
                if "final" not in ledger.runs()[run_id]:
                    ledger.final(run_id, pod_type=reserve["pod_type"], hours=hours, usd=usd,
                                 status=receipt["status"] if receipt["status"] != "running" else "incomplete",
                                 pod_id=pod_id, hub_revision=revision)
        else:
            print("Provider billing is pending. The estimate remains reserved.")
    print(f"Collected {len(receipt['snapshots'])} verified snapshots from {revision} into {dest}.")
    return 0 if receipt["status"] == "ok" else 1
