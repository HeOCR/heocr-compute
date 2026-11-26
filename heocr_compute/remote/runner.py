"""One remote run, start to finish, with the pod always terminated at the end.

1. Sweep orphaned heocr pods.
2. Check the inputs: manifest hashes, open roots and the local denylist (policy.py), upload size.
3. Check the money: live price, balance reserve and per-run cap (budget.py), under the ledger's
   lock; check CPU admission and reserve the worst case in the ledger.
4. Create the pod, check its billed price against the quote, wait for SSH, arm a dead-man's switch on the
   pod (it removes itself after the wall-time limit plus a grace period, should this machine sleep or
   lose its network).
5. Upload by manifest; ``sha256sum -c`` on the pod.
6. Start the job detached, under ``timeout``; poll for its exit code with short SSH calls.
7. Pull back what the job left in ``/heocr/out``: checkpoints, logs and scores only, final ones by
   default; verify every file's sha256 against the pod's.
8. Terminate the pod (on success, error, timeout or Ctrl-C), confirm it is gone, write the ledger's final.
"""
from __future__ import annotations

import datetime as dt
import fnmatch
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from . import budget, manifest, profiles, render_job
from .api import RunPodError
from .ledger import Ledger, pid_alive as _pid_alive, utcnow
from .policy import Policy, Refused
from .ssh import Remote, SSHError, public_key

POD_PREFIX = "heocr-"
HF_PROFILE_MARKER = "hf"
ARCHIVE_PROFILE_MARKER = "archive"
POD_IN, POD_OUT, POD_EXIT = "/heocr/in", "/heocr/out", "/heocr/exit_code"
POD_DEADMAN = "/heocr/deadman.sh"
POD_STARTED = "/heocr/job.started"
CREATE_TRIES = 4                  # a refused create (400 with no pod, 429, 5xx) is tried again, after a pause
CREATE_PAUSES = (30, 60, 120)
START_TRIES = 5                   # the job's start is idempotent, so a dropped SSH call is simply repeated
READY_TIMEOUT = 20 * 60          # provisioning + image pull
DEADMAN_GRACE = 15 * 60          # the pod removes itself this long after the wall-time limit
PULL_RESERVE = 10 * 60           # wall time kept back for pulling results
LOCAL_FREE_MARGIN = 1 << 30      # keep 1 GB free on this machine after a pull
CPU_ACCOUNT_VCPU_CAP = 64
DEFAULT_CAPACITY_WAIT_HOURS = 6.0
CAPACITY_POLL_SECONDS = 5 * 60

# The pod-side dead-man's switch. ``deadman.sh SECONDS`` arms it in the background; ``deadman.sh now``
# fires it at once (the smoke run's proof). An SSH session does not inherit the container's environment,
# so RunPod's pod id and pod-scoped key are read from PID 1's. It prints what is missing, never values.
DEADMAN_SH = r"""#!/bin/sh
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do
  eval "cur=\${$v:-}"
  if [ -z "$cur" ] && [ -r /proc/1/environ ]; then
    val=$(tr '\000' '\n' < /proc/1/environ | sed -n "s/^$v=//p" | head -n 1)
    [ -n "$val" ] && export "$v=$val"
  fi
done
log="${HEOCR_DEADMAN_LOG:-/heocr/deadman.log}"
missing=""
[ -n "${RUNPOD_POD_ID:-}" ] || missing="$missing pod-id"
[ -n "${RUNPOD_API_KEY:-}" ] || missing="$missing pod-key"
command -v runpodctl >/dev/null 2>&1 || missing="$missing runpodctl"
if [ -n "$missing" ]; then echo "unavailable:$missing"; exit 0; fi
remove='runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID"'
if [ "$1" = now ]; then sh -c "$remove" >"$log" 2>&1; echo fired; exit 0; fi
detach=""; command -v setsid >/dev/null 2>&1 && detach=setsid
nohup $detach sh -c "sleep $1; $remove" >"$log" 2>&1 </dev/null &
echo armed
"""

# What may come back: checkpoints, logs and scores.
RESULT_SUFFIXES = (".checkpoint", "_checkpoint", ".traineddata", ".lstm", ".pt", ".pth", ".ckpt",
                   ".safetensors", ".mlmodel", ".log", ".txt", ".json", ".jsonl", ".tsv", ".csv", ".md",
                   ".yaml", ".yml", ".sha256")
# lstmtraining's intermediate checkpoints (``<name>_<error>_<iter>_<iter>.checkpoint``): left on the pod
# unless --pull-intermediate.
INTERMEDIATE = re.compile(r"_\d+(\.\d+)?_\d+_\d+\.checkpoint$")


class Interrupted(Exception):
    pass


class RunFailed(Exception):
    pass


class DeadlineExceeded(RunFailed):
    pass


class CpuCapacityUnavailable(Refused):
    """Verified free capacity is below the requested CPU pod size."""


def _remaining(clock, deadline: float | None, limit: float) -> float:
    if deadline is None:
        return limit
    remaining = deadline - clock()
    if remaining <= 0:
        raise DeadlineExceeded("render wall-time limit reached")
    return min(remaining, limit)


@dataclass
class Spec:
    profile: profiles.Profile
    manifest_path: Path
    root: Path
    cmd: str
    hours: float
    run_cap: float = budget.RUN_CAP_USD
    month_cap: float | None = None
    pulls: tuple = ("**",)
    pull_intermediate: bool = False
    results: Path | None = None
    max_pull_gb: float = 2.0
    max_upload_gb: float = 20.0
    dry_run: bool = False
    yes: bool = False
    policy: object = None     # policy.Policy; loaded from the local files when None
    prove_deadman: bool = False  # smoke only: fire the pod's switch at the end and check the pod is gone
    render_bundle: bool = False  # fixed render recipe only: bounded tar + its checksum manifest
    capacity_wait_hours: float | None = DEFAULT_CAPACITY_WAIT_HOURS
    archive_credentials: Path | None = None  # optional run-scoped, autonomous R2 archive


def new_run_id(now: dt.datetime) -> str:
    return now.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def pod_name(profile: str, run_id: str) -> str:
    return f"{POD_PREFIX}{profile}-{run_id}"


def run_id_of(name: str) -> str | None:
    m = re.fullmatch(rf"{POD_PREFIX}[a-z0-9]+-(\d{{8}}-\d{{6}}-[0-9a-f]{{4}})", name or "")
    return m.group(1) if m else None


def _cpu_pod_vcpus(pod: dict) -> int:
    status = pod.get("status")
    if status in ("EXITED", "TERMINATED"):
        return 0
    if not isinstance(status, str) or not status:
        raise Refused("CPU admission requires a verifiable RunPod pod status")
    gpu = pod.get("gpu")
    if isinstance(gpu, dict) and gpu:
        return 0
    cpu = pod.get("cpu")
    count = cpu.get("vcpuCount") if isinstance(cpu, dict) else None
    if gpu is not None or type(count) is not int or count < 1:
        raise Refused("CPU admission requires a verifiable RunPod compute type and vCPU count")
    return count


def _reserved_cpu_vcpus(reserve: dict) -> int:
    if "vcpus" in reserve:
        count = reserve["vcpus"]
        if count is None:
            return 0
        if type(count) is not int or count < 1:
            raise Refused("CPU admission requires a valid local vCPU reservation")
        return count
    # Earlier ledgers stored the CPU flavour and size together, for example cpu3cx32.
    legacy = re.fullmatch(r"cpu[\w-]+x([1-9]\d*)", reserve.get("pod_type") or "")
    return int(legacy[1]) if legacy else 0


def returnable(rel: str, pulls, pull_intermediate: bool) -> str | None:
    """Why ``rel`` is not pulled back, or None if it is."""
    if not any(fnmatch.fnmatch(rel, g) for g in pulls):
        return "not matched by --pull"
    if not rel.lower().endswith(RESULT_SUFFIXES):
        return "not a checkpoint, log or score file"
    if not pull_intermediate and INTERMEDIATE.search(rel):
        return "intermediate checkpoint"
    return None


def _parse_ts(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def terminate_confirmed(api, pod_id: str, sleep=time.sleep, attempts: int = 6) -> bool:
    for i in range(attempts):
        try:
            api.terminate(pod_id)
            pod = api.get_pod(pod_id)
            if pod is None or pod.get("status") == "TERMINATED":
                return True
        except Exception:  # noqa: BLE001 - keep trying; the caller reports failure loudly
            pass
        sleep(min(30, 2 ** i))
    return False


def observe_startup(api, pod_id: str) -> str:
    try:
        return api.pod_startup_state(pod_id)
    except RunPodError:
        return "unknown"


def record_startup(ledger: Ledger, run_id: str, pod_id: str, *, state: str) -> None:
    with ledger.locked():
        run = ledger.runs()[run_id]
        if not any(key in run for key in ("startup", "final")):
            ledger.append({"event": "startup", "run_id": run_id, "pod_id": pod_id, "status": state})


def reconcile_startup_cost(api, ledger: Ledger, run: dict, now=utcnow) -> bool:
    """Settle a confirmed startup timeout only against positive provider billing."""
    event = run.get("startup", {})
    reserve = run.get("reserve", {})
    if ("final" in run or event.get("status") != "startup_timeout"
            or reserve.get("account") != ledger.account):
        return False
    pod_id = event["pod_id"]
    created = run.get("pod", {}).get("created_utc") or reserve["date"]
    try:
        usd = api.billed_usd(pod_id, created, now().isoformat())
    except (RunPodError, ValueError, TypeError):
        return False
    if type(usd) not in (int, float) or not math.isfinite(usd) or usd <= 0:
        return False
    ended_at = _parse_ts(event["date"])
    created_at = _parse_ts(created) or _parse_ts(reserve["date"])
    hours = max(0, (ended_at - created_at).total_seconds() / 3600)
    with ledger.locked():
        if "final" not in ledger.runs()[run["run_id"]]:
            ledger.final(run["run_id"], pod_type=reserve["pod_type"], hours=hours, usd=usd,
                         status="startup_timeout", pod_id=pod_id, usd_source="provider_billing")
    return True


def sweep(api, ledger: Ledger, *, everything: bool = False, dry_run: bool = False, now=utcnow,
          pid_alive=_pid_alive, out=print, sleep=time.sleep) -> list[str]:
    """Terminate heocr pods with no live runner: finished, past their deadline, or unknown here."""
    host = socket.gethostname()
    runs = ledger.runs()
    killed = []
    ours = [p for p in api.list_pods() if (p.get("name") or "").startswith(POD_PREFIX)]
    out(f"sweep: {len(ours)} heocr pods on the account")
    for pod in ours:
        name = pod.get("name") or ""
        rid = run_id_of(name)
        r = runs.get(rid) if rid else None
        res = (r or {}).get("reserve") or {}
        deadline = _parse_ts(res.get("deadman_utc"))
        startup_deadline = _parse_ts(res.get("startup_deadline_utc"))
        startup_timeout = False
        if everything:
            why = "--all"
        elif r is None:
            if rid and name.startswith(("heocr-hf-", "heocr-archive-")):
                # An autonomous pod can have been launched from another Mac.
                continue
            why = "not in this machine's ledger"
        elif "final" in r:
            why = "its run already finished"
        elif deadline and now() > deadline:
            why = "past its deadline"
        elif (res.get("autonomous") and res.get("account") == ledger.account and startup_deadline
              and now() >= startup_deadline and "startup" not in r
              and (r.get("pod") or {}).get("pod_id") == pod["id"]):
            state = observe_startup(api, pod["id"])
            if state == "running" and res.get("worker_receipt_required"):
                from .api import load_hf_token
                from .hf_job import is_r2_output
                from .hf_train import read_worker_startup

                try:
                    token = None if is_r2_output(res) else load_hf_token()
                    if read_worker_startup(token, rid, pod["id"], res) is None:
                        state = "missing_receipt"
                except (ImportError, OSError, RunPodError, ValueError):
                    out(f"sweep: worker receipt unavailable for pod {pod['id']}; verify private output access")
                    continue
            if state == "running" and not dry_run:
                record_startup(ledger, rid, pod["id"], state=state)
            if state == "unknown":
                out(f"sweep: startup telemetry unavailable for pod {pod['id']}; verify provider state")
            if state not in ("pending", "stopped", "missing_receipt"):
                continue
            why = ("no verified worker receipt by its startup deadline" if state == "missing_receipt"
                   else "no container runtime by its startup deadline")
            startup_timeout = True
        elif (not res.get("autonomous") or (res.get("r2_archive") and "startup" not in r)) \
                and res.get("host") == host and not pid_alive(res.get("pid", -1)):
            why = "its runner is gone"
        else:
            continue
        out(f"sweep: {pod.get('id')} ({name}): {why}" + (" [dry run]" if dry_run else ""))
        if dry_run:
            continue
        ok = terminate_confirmed(api, pod["id"], sleep=sleep)
        if not ok:
            out(f"sweep: POD {pod['id']} IS STILL RUNNING: terminate it in the RunPod console")
            continue
        killed.append(pod["id"])
        if startup_timeout:
            record_startup(ledger, rid, pod["id"], state="startup_timeout")
        elif rid and r is not None and "final" not in r:
            created = _parse_ts(pod.get("createdAt")) or _parse_ts(res.get("date")) or now()
            secs = max(0.0, (now() - created).total_seconds())
            rate = float(pod.get("cost") or (r.get("pod") or {}).get("usd_per_hr") or 0)
            ledger.final(rid, pod_type=res.get("pod_type", "?"), hours=secs / 3600,
                         usd=budget.billed(rate, secs), status="swept", pod_id=pod["id"])
    if not dry_run:
        for run in ledger.runs().values():
            reconcile_startup_cost(api, ledger, run, now=now)
    return killed


@dataclass
class Runner:
    api: object
    ledger: Ledger
    out: object = partial(print, flush=True)
    sleep: object = time.sleep
    clock: object = time.monotonic
    now: object = utcnow
    poll_seconds: float = 60
    remote_factory: object = Remote.from_pod
    confirm: object = input
    keep_awake: bool = True
    _terminating: bool = field(default=False, init=False)

    # -- preflight ---------------------------------------------------------------------------------
    def check_inputs(self, spec: Spec):
        entries = manifest.load(spec.manifest_path)
        size = manifest.verify_local(entries, spec.root, spec.policy or Policy.load())
        if size > spec.max_upload_gb * (1 << 30):
            raise RunFailed(f"upload of {size / (1 << 30):.2f} GB is over --max-upload-gb {spec.max_upload_gb}")
        return entries, size

    def price(self, spec: Spec, month_spent: float) -> budget.Quote:
        compute = profiles.live_compute_price(self.api, spec.profile)
        return budget.quote(compute_usd_per_hour=compute, disk_gb=spec.profile.disk_gb, wall_hours=spec.hours,
                            balance=self.api.balance(), month_spent=month_spent,
                            open_reservations=self.ledger.open_reservations_usd(),
                            run_cap=spec.run_cap, month_cap=spec.month_cap)

    def require_cpu_capacity(self, profile: profiles.Profile) -> None:
        """Check admission under the ledger lock, immediately before reserving."""
        if profile.kind != "cpu":
            return
        if profile.vcpus > CPU_ACCOUNT_VCPU_CAP:
            raise Refused(f"requested {profile.vcpus} vCPUs exceeds the {CPU_ACCOUNT_VCPU_CAP}-vCPU account limit")
        runs = self.ledger.runs()
        listed, names = {}, set()
        for pod in self.api.list_pods(include_cluster_pods=True):
            pid, name = pod.get("id"), pod.get("name")
            if not isinstance(pid, str) or not pid or not isinstance(name, str) or not name:
                raise Refused("CPU admission requires a verifiable RunPod pod identity")
            listed[pid] = max(listed.get(pid, 0), _cpu_pod_vcpus(pod))
            names.add(name)

        pending = 0
        for run_id, run in runs.items():
            reserve = run.get("reserve") or {}
            if "final" in run or reserve.get("account") != self.ledger.account:
                continue
            count = _reserved_cpu_vcpus(reserve)
            if not count:
                continue
            pid = run.get("pod", {}).get("pod_id")
            if pid:
                if pid in listed:
                    continue
                # An exact-ID read covers list lag and releases departed pods even if billing is pending.
                pod = self.api.get_pod(pid)
                if pod is not None:
                    if not isinstance(pod, dict) or pod.get("id") != pid:
                        raise Refused("CPU admission requires a verifiable RunPod pod identity")
                    listed[pid] = _cpu_pod_vcpus(pod)
                continue
            marker = (ARCHIVE_PROFILE_MARKER if reserve.get("r2_archive") else HF_PROFILE_MARKER
                      if reserve.get("autonomous") else reserve["profile"])
            if pod_name(marker, run_id) not in names:
                pending += count

        used = sum(listed.values())
        free = max(0, CPU_ACCOUNT_VCPU_CAP - used - pending)
        if profile.vcpus > free:
            raise CpuCapacityUnavailable(f"CPU admission refused: requested {profile.vcpus} vCPUs, {free} of "
                                         f"{CPU_ACCOUNT_VCPU_CAP} available ({used} in pods, "
                                         f"{pending} in unconfirmed reservations)")

    @contextmanager
    def launch_slot(self, spec, run_id: str):
        """Yield (quote, waited) under the lock for the caller to reserve, or (None, False) on decline."""
        wait_hours = spec.capacity_wait_hours
        wait_seconds = None if wait_hours is None else wait_hours * 3600
        if wait_seconds is not None and (not math.isfinite(wait_seconds) or wait_seconds <= 0):
            raise ValueError("capacity wait must be a finite, positive number of hours")
        deadline = approved_cost = None
        queue_end = "failed"
        try:
            while True:
                with self.ledger.locked():
                    if deadline is not None and self.clock() >= deadline:
                        break
                    quote = self.price(spec, self.ledger.month_to_date()["total"])
                    new_cost = approved_cost is None or quote.usd_est > approved_cost
                    if new_cost:
                        for line in quote.lines():
                            self.out(line)
                    budget.check(quote)
                    if new_cost:
                        if not (spec.dry_run or spec.yes) and self.confirm(
                                f"start {spec.profile.pod_type}, worst case ${quote.usd_est:.2f}? [y/N] ").strip().lower() != "y":
                            queue_end = "declined"
                            yield None, False
                            return
                        approved_cost = quote.usd_est
                    if spec.dry_run:
                        yield quote, False
                        return
                    try:
                        self.require_cpu_capacity(spec.profile)
                    except CpuCapacityUnavailable as error:
                        if wait_seconds is None:
                            raise
                        shortage = str(error)
                        if deadline is None:
                            self.ledger.queue(run_id, vcpus=spec.profile.vcpus, wait_seconds=wait_seconds)
                            deadline = self.clock() + wait_seconds
                    else:
                        if deadline is not None and self.clock() >= deadline:
                            break
                        if deadline is not None and not new_cost:
                            for line in quote.lines():
                                self.out(line)
                        yield quote, deadline is not None
                        queue_end = None  # The reservation closes the wait in the ledger.
                        return
                delay = min(CAPACITY_POLL_SECONDS, deadline - self.clock())
                if delay > 0:
                    self.out(f"Run {run_id}: {shortage}. Waiting {delay:.0f}s for CPU capacity.")
                    self.sleep(delay)
            queue_end = "expired"
            raise Refused(f"capacity wait expired after {wait_hours:g} h ({shortage})")
        except KeyboardInterrupt:
            queue_end = "cancelled"
            raise
        finally:
            if deadline is not None and queue_end is not None:
                with self.ledger.locked():
                    self.ledger.end_queue(run_id, status=queue_end)

    # -- the run -----------------------------------------------------------------------------------
    def run(self, spec: Spec) -> dict:
        sweep(self.api, self.ledger, dry_run=spec.dry_run, out=self.out, sleep=self.sleep, now=self.now)
        entries, size = self.check_inputs(spec)
        self.out(f"inputs: {len(entries)} files, {size / (1 << 20):.1f} MB, hashes, open roots and local denylist checked")
        archive = None
        if spec.archive_credentials:
            from .r2_archive import ArchiveStore

            if spec.render_bundle or spec.prove_deadman or spec.pull_intermediate or spec.pulls != ("**",):
                raise Refused("the archive return uses its own fixed output contract")
            if not math.isfinite(spec.max_pull_gb) or not 0 < spec.max_pull_gb <= 2:
                raise Refused("the archive return supports up to 2 GiB")
            if not math.isfinite(spec.hours) or not PULL_RESERVE < spec.hours * 3600 <= 23 * 3600:
                raise Refused("archive runs must leave at least ten minutes for R2 verification and fit within 23 hours")
            archive = ArchiveStore(spec.archive_credentials, int(spec.max_pull_gb * (1 << 30)))
        run_id = new_run_id(self.now())
        name = pod_name(ARCHIVE_PROFILE_MARKER if archive else spec.profile.name, run_id)
        with self.launch_slot(spec, run_id) as (q, waited):
            if q is None:
                return {"run_id": run_id, "status": "declined"}
            if spec.dry_run:
                body = spec.profile.body(name, {"PUBLIC_KEY": "<heocr-remote ssh public key>"})
                self.out("dry run: would create " + json.dumps(body))
                self.out(f"dry run: would run under timeout {self._job_seconds(spec):.0f}s: {spec.cmd}")
                if archive:
                    self.out("dry run: would stage one autonomous R2 archive and a SHA-256 manifest")
                return {"run_id": run_id, "status": "dry_run", "quote": q.__dict__}
            if waited:
                manifest.verify_local(entries, spec.root, spec.policy or Policy.load())
            archive_return = (archive.scope, archive.urls(run_id, math.ceil(spec.hours * 3600) + 900, self.now())) if archive else None
            guard_seconds = 0 if spec.render_bundle or archive else DEADMAN_GRACE + budget.STARTUP_HOURS * 3600
            deadman = self.now() + dt.timedelta(hours=spec.hours, seconds=guard_seconds)
            transport = {"autonomous": True, "r2_archive": archive.scope,
                         "output": {"transport": "r2"}} if archive else {}
            self.ledger.reserve(run_id, profile=spec.profile.name, pod_type=spec.profile.pod_type,
                                vcpus=spec.profile.vcpus, usd_est=q.usd_est, hours_max=q.hours,
                                host=socket.gethostname(), deadman_utc=deadman.isoformat(timespec="seconds"), **transport)
        if archive_return is None:
            return self._execute(spec, entries, run_id, name, q)
        return self._execute(spec, entries, run_id, name, q, archive_return=archive_return)

    def _job_seconds(self, spec: Spec) -> float:
        return max(60.0, spec.hours * 3600 - PULL_RESERVE)

    def _execute(self, spec, entries, run_id, name, q, *, archive_return=None) -> dict:
        pod_id, rate, t_created, status, error, pulled = None, q.usd_per_hour, None, "error", None, []
        deadman = proof = None
        handlers = self._trap()
        caffeinate = self._keep_awake()
        t0 = self.clock()
        deadline = t0 + spec.hours * 3600
        strict = spec.render_bundle or archive_return is not None
        hard = deadline if strict else None
        detached = False
        try:
            env = {"PUBLIC_KEY": public_key(), "HEOCR_RUN_ID": run_id}
            pod = self._create(spec.profile.body(name, env), name, hard)
            pod_id, t_created = pod["id"], self.clock()
            billed_rate = float(pod.get("cost") or 0)
            if strict and (not math.isfinite(billed_rate) or billed_rate <= 0):
                raise RunFailed("created pod has no verifiable hourly price")
            billed_rate = billed_rate or q.compute_usd_per_hour
            rate = max(rate, billed_rate + budget.hourly(0, spec.profile.disk_gb))
            self.ledger.pod(run_id, pod_id=pod_id, usd_per_hr=billed_rate)
            self.out(f"pod {pod_id} created at ${billed_rate:.4f}/h")
            if budget.overpriced(q, billed_rate) or (strict and billed_rate > q.compute_usd_per_hour):
                raise RunFailed(f"pod billed ${billed_rate:.4f}/h, over the quote ${q.compute_usd_per_hour:.4f}/h")
            remote = self._wait_ready(pod_id, hard)
            deadman = self._arm_deadman(remote, deadline, strict=strict)
            if strict and deadman != "armed":
                raise RunFailed("bounded return requires an armed pod self-termination timer")
            self._record_pod(remote, run_id, pod_id, spec.profile.pod_type, billed_rate, hard)
            self._upload(remote, spec, entries, hard)
            if archive_return is not None:
                self._launch_archive(remote, spec, run_id, pod_id, deadline, archive_return)
                record_startup(self.ledger, run_id, pod_id, state="running")
                detached = True
                self.out("Archive supervisor confirmed. The pod verifies its R2 return and terminates itself.")
                return {"run_id": run_id, "pod_id": pod_id, "status": "archive_running", "deadman": deadman}
            status = self._run_job(remote, spec, deadline)
            pulled = self._pull(remote, spec, run_id, hard, bundle=spec.render_bundle and status == "ok")
            if spec.render_bundle:
                _remaining(self.clock, hard, 1)
            if spec.prove_deadman and deadman == "armed":
                proof = self._prove_self_removal(remote, pod_id)
            if status == "ok":
                self.out(f"done: {len(pulled)} files in {self._results_dir(spec, run_id)}")
        except Interrupted:
            status, error = "interrupted", "interrupted"
            self.out("interrupted: terminating the pod")
        except DeadlineExceeded as e:
            status, error = "timeout", str(e)
        except Exception as e:  # noqa: BLE001 - every failure ends with the pod terminated
            status = "timeout" if hard is not None and self.clock() >= hard else "error"
            error = f"{type(e).__name__}: {e}"
            self.out(f"error: {error}")
        finally:
            if not detached:
                self._terminating = True
                gone = True
                if pod_id is None:
                    try:
                        pod_id = self._find_by_name(name, required=strict)
                    except RunFailed:
                        gone = False  # a lost create response may conceal a billed pod
                if pod_id:
                    gone = terminate_confirmed(self.api, pod_id, sleep=self.sleep)
                    self.out(f"pod {pod_id} terminated" if gone else
                             f"POD {pod_id} IS STILL RUNNING: run `heocr-remote sweep`, or terminate it in the console")
                if strict and not gone:
                    status, error = "error", "pod termination could not be confirmed"
                secs = (self.clock() - t_created) if t_created is not None else 0.0
                if gone:
                    usd = budget.billed(rate, secs) if pod_id else 0.0
                    self.ledger.final(run_id, pod_type=spec.profile.pod_type, hours=secs / 3600, usd=usd,
                                      status=status, pod_id=pod_id, error=error, pulled=len(pulled),
                                      deadman=deadman, self_removal=proof)
            self._untrap(handlers)
            if caffeinate:
                caffeinate.terminate()
        return {"run_id": run_id, "pod_id": pod_id, "status": status, "error": error, "pulled": pulled,
                "deadman": deadman, "self_removal": proof}

    def _create(self, body: dict, name: str, deadline: float | None = None) -> dict:
        """Create the pod. RunPod sometimes refuses a valid create (HTTP 400 with no capacity, 429, 5xx):
        after checking by name that no pod was made, the same create is tried again after a pause."""
        from .api import RunPodError
        for attempt in range(CREATE_TRIES):
            _remaining(self.clock, deadline, 1)
            try:
                return self.api.create_pod(body)
            except (RunPodError, OSError) as e:
                status = getattr(e, "status", None)
                if status is not None and status not in (400, 429) and status < 500:
                    raise
                found = self._find_by_name(name, required=deadline is not None)
                if found:
                    return self.api.get_pod(found) or {"id": found}
                if attempt == CREATE_TRIES - 1:
                    raise
                pause = CREATE_PAUSES[min(attempt, len(CREATE_PAUSES) - 1)]
                self.out(f"create refused ({e}); trying again in {pause}s")
                self.sleep(_remaining(self.clock, deadline, pause))
        raise RunFailed("unreachable")

    def _find_by_name(self, name: str, *, required: bool = False) -> str | None:
        try:
            for p in self.api.list_pods():
                if p.get("name") == name:
                    self.out(f"found pod {p['id']} from a create that reported failure")
                    return p["id"]
        except Exception:  # noqa: BLE001 - the sweeper catches it later
            self.out(f"could not check for a pod named {name}: run `heocr-remote sweep`")
            if required:
                raise RunFailed("could not verify pod creation after an ambiguous response") from None
        return None

    # -- steps -------------------------------------------------------------------------------------
    def _wait_ready(self, pod_id: str, deadline: float | None = None) -> Remote:
        start = self.clock()
        remote = None
        while self.clock() - start < READY_TIMEOUT:
            _remaining(self.clock, deadline, 1)
            pod = self.api.get_pod(pod_id)
            if pod is None or pod.get("status") in ("ERROR", "TERMINATED", "EXITED"):
                raise RunFailed(f"pod {pod_id} is {pod and pod.get('status')}")
            if pod.get("status") == "RUNNING":
                remote = self.remote_factory(pod)
                if remote is not None:
                    try:
                        remote.run("true", timeout=_remaining(self.clock, deadline, 60))
                        return remote
                    except SSHError:
                        pass
            self.sleep(_remaining(self.clock, deadline, 10))
        raise RunFailed(f"pod {pod_id} not reachable over SSH within {READY_TIMEOUT // 60} min")

    def _arm_deadman(self, remote, deadline: float, *, strict: bool = False) -> str:
        hard = deadline if strict else None
        remote.run(f"mkdir -p /heocr && cat > {POD_DEADMAN}", input=DEADMAN_SH,
                   timeout=_remaining(self.clock, hard, 60))
        secs = (int(_remaining(self.clock, hard, float("inf"))) if strict else
                int(max(60, deadline - self.clock()) + DEADMAN_GRACE))
        r = remote.run(f"sh {POD_DEADMAN} {secs}", timeout=_remaining(self.clock, hard, 60), check=False)
        state = (r.stdout or "").strip().splitlines()[-1:] or ["unavailable: no answer"]
        state = state[0]
        self.out(f"dead-man's switch: {state} ({secs // 60} min)" +
                 ("" if state == "armed" else "; the sweeper is the only backstop"))
        return state

    def _record_pod(self, remote, run_id: str, pod_id: str, pod_type: str, usd_per_hr: float,
                    deadline: float | None = None) -> None:
        """out/pod.json: which machine the job ran on. It comes back with the results, next to the job's own
        records, so a model's provenance names its hardware."""
        info = {"run_id": run_id, "pod_id": pod_id, "pod_type": pod_type, "usd_per_hr": usd_per_hr,
                "provider": "runpod"}
        remote.run(f"mkdir -p {POD_OUT} && cat > {POD_OUT}/pod.json", input=json.dumps(info, indent=1) + "\n",
                   timeout=_remaining(self.clock, deadline, 60))

    def _prove_self_removal(self, remote, pod_id: str) -> str:
        """Smoke runs only: fire the switch now, and check through the API that the pod is gone."""
        try:
            remote.run(f"sh {POD_DEADMAN} now", timeout=120, check=False)
        except SSHError:
            pass  # the connection may drop as the pod goes
        for _ in range(12):
            self.sleep(10)
            pod = self.api.get_pod(pod_id)
            if pod is None or pod.get("status") == "TERMINATED":
                self.out("self-removal: works (the pod removed itself)")
                return "works"
        self.out("self-removal: FAILED (the pod is still there; the runner terminates it)")
        return "failed"

    def _upload(self, remote, spec, entries, deadline: float | None = None) -> None:
        kwargs = {"timeout": _remaining(self.clock, deadline, 60)} if deadline is not None else {}
        remote.upload(spec.root, [e.path for e in entries], POD_IN, **kwargs)
        remote.run(f"cat > /heocr/in.sha256", input=manifest.sha256sum_text(entries),
                   timeout=_remaining(self.clock, deadline, 120))
        r = remote.run(f"cd {POD_IN} && sha256sum -c --quiet /heocr/in.sha256",
                       timeout=_remaining(self.clock, deadline, 3600), check=False)
        if r.returncode != 0:
            raise RunFailed("inputs failed sha256 on the pod: " + (r.stdout + r.stderr).strip()[-300:])
        self.out(f"uploaded {len(entries)} files, sha256 verified on the pod")

    def _start_job(self, remote, inner, hard):
        # Idempotent: the marker is written once, before the job starts, so repeating the call after a
        # dropped connection never starts a second job.
        # The whole backgrounded group has its streams redirected, so nothing keeps the SSH channel open:
        # "a && b > /dev/null &" would background a subshell that holds SSH's stdout and waits for the job.
        start = (f"mkdir -p {POD_OUT} /heocr/work && cd /heocr && if [ ! -e {POD_STARTED} ]; then "
                 f"date -u > {POD_STARTED}; ( exec setsid bash -c {shlex.quote(inner)} ) > /dev/null 2>&1 < /dev/null & "
                 f"fi; test -e {POD_STARTED} && echo started")
        for attempt in range(START_TRIES):
            try:
                r = remote.run(start, timeout=_remaining(self.clock, hard, 60), check=False)
                if "started" in (r.stdout or ""):
                    break
            except SSHError as e:
                self.out(f"job start: {e}; trying again")
            self.sleep(_remaining(self.clock, hard, 10))
        else:
            raise RunFailed(f"the job could not be started over SSH in {START_TRIES} tries")

    def _launch_archive(self, remote, spec, run_id, pod_id, deadline, transport):
        scope, urls = transport
        seconds = int(deadline - self.clock() - PULL_RESERVE)
        if seconds < 1:
            raise DeadlineExceeded("no time left after reserving the R2 verification allowance")
        root = "/heocr/archive-supervisor"
        package = f"{root}/heocr_compute/remote"
        remote.run("python3 -c 'import hashlib; assert hasattr(hashlib, \"file_digest\")'"
                   f" && command -v timeout >/dev/null && mkdir -p {package}",
                   timeout=_remaining(self.clock, deadline, 60))
        for module in ("r2_archive_worker.py", "r2_input.py"):
            source = Path(__file__).with_name(module).read_text(encoding="utf-8")
            remote.run(f"cat > {package}/{module}", input=source, timeout=_remaining(self.clock, deadline, 60))
        settings = {"scope": scope, "urls": urls, "run_id": run_id, "pod_id": pod_id,
                    "seconds": seconds, "cmd": spec.cmd}
        try:
            result = remote.run("umask 077; cat > /heocr/.archive-secret", input=json.dumps(settings),
                                timeout=_remaining(self.clock, deadline, 60), check=False)
        except SSHError:
            raise RunFailed("could not transfer the pod-scoped archive secret") from None
        if result.returncode:
            raise RunFailed("could not transfer the pod-scoped archive secret")
        self._start_job(remote, f"PYTHONPATH={root} python3 -m heocr_compute.remote.r2_archive_worker", deadline)
        ready_by = min(deadline, self.clock() + 60)
        while self.clock() < ready_by:
            try:
                result = remote.run("test -f /heocr/archive.ready", timeout=_remaining(self.clock, ready_by, 10), check=False)
                if result.returncode == 0:
                    return
            except SSHError:
                pass
            self.sleep(_remaining(self.clock, ready_by, 2))
        raise RunFailed("the autonomous archive supervisor did not acknowledge startup")

    def _run_job(self, remote, spec, deadline: float) -> str:
        hard = deadline if spec.render_bundle else None
        remaining = min(self._job_seconds(spec), deadline - self.clock() - PULL_RESERVE)
        if spec.render_bundle and remaining < 1:
            raise DeadlineExceeded("no render time left after setup and bundle pull allowance")
        secs = int(max(1 if spec.render_bundle else 60, remaining))
        inner = f"timeout --signal=TERM --kill-after=120 {secs} bash -lc {shlex.quote(spec.cmd)} > {POD_OUT}/job.log 2>&1; echo $? > {POD_EXIT}"
        self._start_job(remote, inner, hard)
        self.out(f"job started, limit {secs // 60} min")
        failures = 0
        while self.clock() < deadline:
            self.sleep(_remaining(self.clock, hard, self.poll_seconds))
            if hard is not None and self.clock() >= hard:
                return "timeout"
            try:
                r = remote.run(f"cat {POD_EXIT} 2>/dev/null || true",
                               timeout=_remaining(self.clock, hard, 60), check=False)
                failures = 0
            except SSHError:
                failures += 1
                self.out(f"poll: SSH failed ({failures} in a row)")
                continue
            code = (r.stdout or "").strip()
            if code:
                code = int(code)
                self.out(f"job exited {code}")
                return "ok" if code == 0 else ("timeout" if code == 124 else "job_failed")
        return "timeout"

    def _results_dir(self, spec, run_id) -> Path:
        return Path(spec.results or Path("~/heocr-remote/results").expanduser()) / run_id

    def _pull(self, remote, spec, run_id, deadline: float | None = None, *, bundle: bool = False) -> list[str]:
        r = remote.run(f"cd {POD_OUT} && find . -type f -printf '%s\\t%P\\n'",
                       timeout=_remaining(self.clock, deadline, 600))
        sizes = {}
        for line in r.stdout.splitlines():
            if "\t" in line:
                n, rel = line.split("\t", 1)
                sizes[manifest.safe_rel(rel)] = int(n)
        chosen, skipped = [], {}
        required = {render_job.ARCHIVE, render_job.ARTIFACT_MANIFEST} if bundle else set()
        if required - sizes.keys() or (bundle and not 0 < sizes[render_job.ARCHIVE] <= render_job.MAX_ARCHIVE):
            raise RunFailed("render completed without a valid archive and artifact manifest")
        for rel in sorted(sizes):
            if rel in required:
                chosen.append(rel)
                continue
            if spec.render_bundle and rel not in ("job.log", "pod.json"):
                continue
            why = returnable(rel, spec.pulls, spec.pull_intermediate)
            (skipped.__setitem__(rel, why) if why else chosen.append(rel))
        for rel, why in skipped.items():
            self.out(f"left on the pod: {rel} ({why})")
        if not chosen:
            return []
        total = sum(sizes[r] for r in chosen)
        if total > spec.max_pull_gb * (1 << 30):
            raise RunFailed(f"results are {total / (1 << 30):.2f} GB, over --max-pull-gb {spec.max_pull_gb}")
        dest = self._results_dir(spec, run_id)
        dest.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(dest).free < total + LOCAL_FREE_MARGIN:
            raise RunFailed(f"not enough free disk here for {total / (1 << 20):.0f} MB of results")
        listing = remote.run(f"cd {POD_OUT} && sha256sum -- " + " ".join(shlex.quote(c) for c in chosen),
                             timeout=_remaining(self.clock, deadline, 3600))
        want = manifest.parse_sha256sum(listing.stdout)
        bad = self._fetch(remote, chosen, want, dest, deadline, total)
        if bad:
            bad = self._fetch(remote, bad, want, dest, deadline, sum(sizes[r] for r in bad))
        if bad:
            raise RunFailed(f"sha256 mismatch after retry: {', '.join(bad)}")
        (dest / "SHA256SUMS").write_text("".join(f"{want[c]}  {c}\n" for c in chosen), encoding="utf-8")
        self.out(f"pulled {len(chosen)} files ({total / (1 << 20):.1f} MB), sha256 verified")
        return chosen

    def _fetch(self, remote, rels, want, dest, deadline: float | None, max_bytes: int) -> list[str]:
        kwargs = ({"timeout": _remaining(self.clock, deadline, 3600), "max_bytes": max_bytes}
                  if deadline is not None else {})
        remote.download(POD_OUT, rels, dest, **kwargs)
        bad = []
        for rel in rels:
            f = dest / rel
            if not f.is_file() or manifest.sha256_file(f) != want.get(rel):
                bad.append(rel)
                if f.exists():
                    f.unlink()
        return bad

    # -- signals and sleep -------------------------------------------------------------------------
    def _trap(self):
        def handler(signum, frame):
            if not self._terminating:
                raise Interrupted(signal.Signals(signum).name)
        saved = {}
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                saved[s] = signal.signal(s, handler)
            except ValueError:  # not the main thread (tests)
                pass
        return saved

    def _untrap(self, saved):
        for s, h in saved.items():
            signal.signal(s, h)

    def _keep_awake(self):
        exe = shutil.which("caffeinate") if self.keep_awake else None
        if not exe:
            return None
        return subprocess.Popen([exe, "-i", "-w", str(os.getpid())])
