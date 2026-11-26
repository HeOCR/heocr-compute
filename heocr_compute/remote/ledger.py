"""The spend ledger: one JSON line per event, outside any repository.

Location: ``$HEOCR_REMOTE_STATE/spend.jsonl``, by default ``~/.config/heocr/remote/spend.jsonl``.

Events of a run, all with ``run_id``:
- ``queue``: the local launcher is waiting for CPU capacity;
- ``queue_end``: the wait ended before a reservation was made;
- ``reserve``: written before a pod is created, with the run's worst-case cost (``usd_est``);
- ``pod``: the pod's id and its actual hourly price, once created;
- ``final``: date, pod type, hours and dollars actually spent, and how the run ended.

Month-to-date spend counts every ``final``, plus the estimate of every reservation without one (a run in
progress, or one whose runner died before finishing). So a crash can only overstate the month, never
understate it. ``sweep`` writes the missing ``final`` when it terminates an orphaned pod.

Every event carries the RunPod account it ran on (``account``, the label from the account file). The
month's total counts only the current account's runs. Runs on other accounts,
or from before accounts were tagged, stay in the file as history and are reported apart.

Reserving takes an exclusive lock on the ledger, so simultaneous dispatches account for each
other's open reservations.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
import socket
from pathlib import Path

STATE_ENV = "HEOCR_REMOTE_STATE"
STATE_DEFAULT = "~/.config/heocr/remote"


def state_dir() -> Path:
    d = Path(os.path.expanduser(os.environ.get(STATE_ENV) or STATE_DEFAULT))
    d.mkdir(parents=True, exist_ok=True)
    return d


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def month_of(ts: str) -> str:
    return ts[:7]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Ledger:
    def __init__(self, path: Path | None = None, now=utcnow, account: str | None = None):
        self.path = Path(path) if path else state_dir() / "spend.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.now = now
        self.account = account

    @contextlib.contextmanager
    def locked(self):
        with open(self.path.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def append(self, event: dict) -> dict:
        event = {"date": self.now().isoformat(timespec="seconds"),
                 **({"account": self.account} if self.account else {}), **event}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return event

    def runs(self) -> dict[str, dict]:
        """Group the latest event of each type by run."""
        runs: dict[str, dict] = {}
        for e in self.events():
            r = runs.setdefault(e["run_id"], {"run_id": e["run_id"]})
            r[e["event"]] = e
        return runs

    def _counts(self, r: dict) -> bool:
        """Whether a run counts towards this ledger's account (all runs when no account is set)."""
        if self.account is None:
            return True
        first = r.get("reserve") or r.get("final") or {}
        return first.get("account") == self.account

    def month_to_date(self, month: str | None = None) -> dict:
        month = month or self.now().strftime("%Y-%m")
        spent = open_est = other = 0.0
        finals, open_runs, others = [], [], []
        for r in self.runs().values():
            ev = r.get("final") or r.get("reserve")
            if not ev or month_of(ev["date"]) != month:
                continue
            if not self._counts(r):
                if "final" in r:
                    other += float(r["final"]["usd"])
                others.append(r)
            elif "final" in r:
                spent += float(r["final"]["usd"])
                finals.append(r)
            else:
                open_est += float(r["reserve"]["usd_est"])
                open_runs.append(r)
        return {"month": month, "account": self.account, "spent": round(spent, 4),
                "open_estimate": round(open_est, 4), "total": round(spent + open_est, 4),
                "finals": finals, "open": open_runs, "other_spent": round(other, 4), "others": others}

    def open_reservations_usd(self) -> float:
        """Worst-case reservations for this account that have yet to be reconciled, across months."""
        return round(sum(float(run["reserve"]["usd_est"]) for run in self.runs().values()
                         if "reserve" in run and "final" not in run and self._counts(run)), 4)

    def waiting(self) -> list[dict]:
        """Current capacity waiters for this account, independent of the spend month."""
        now, host = self.now().timestamp(), socket.gethostname()
        queued = []
        for r in self.runs().values():
            event = r.get("queue")
            if not event or any(k in r for k in ("queue_end", "reserve", "final")):
                continue
            if self.account is not None and event.get("account") != self.account:
                continue
            if now >= dt.datetime.fromisoformat(event["date"]).timestamp() + event["wait_seconds"]:
                continue
            if event["host"] == host and not pid_alive(event["pid"]):
                continue
            queued.append(event)
        return queued

    def queue(self, run_id: str, *, vcpus: int, wait_seconds: float) -> dict:
        return self.append({"event": "queue", "run_id": run_id, "pid": os.getpid(),
                            "host": socket.gethostname(), "vcpus": vcpus, "wait_seconds": wait_seconds})

    def end_queue(self, run_id: str, *, status: str) -> dict:
        return self.append({"event": "queue_end", "run_id": run_id, "status": status})

    def reserve(self, run_id: str, **fields) -> dict:
        return self.append({"event": "reserve", "run_id": run_id, "pid": os.getpid(), **fields})

    def pod(self, run_id: str, **fields) -> dict:
        return self.append({"event": "pod", "run_id": run_id, **fields})

    def final(self, run_id: str, *, pod_type: str, hours: float, usd: float, status: str, **fields) -> dict:
        return self.append({"event": "final", "run_id": run_id, "pod_type": pod_type,
                            "hours": round(hours, 4), "usd": round(usd, 4), "status": status, **fields})
