"""The spend ledger: one JSON line per event, outside any repository.

Location: ``$HEOCR_REMOTE_STATE/spend.jsonl``, by default ``~/.config/heocr/remote/spend.jsonl``.

Events of a run, all with ``run_id``:
- ``reserve``: written before a pod is created, with the run's worst-case cost (``usd_est``);
- ``pod``: the pod's id and its actual hourly price, once created;
- ``final``: date, pod type, hours and dollars actually spent, and how the run ended.

Month-to-date spend counts every ``final``, plus the estimate of every reservation without one (a run in
progress, or one whose runner died before finishing). So a crash can only overstate the month, never
understate it. ``sweep`` writes the missing ``final`` when it terminates an orphaned pod.

Reserving takes an exclusive lock on the ledger, so two runners cannot both pass the monthly cap check.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
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


class Ledger:
    def __init__(self, path: Path | None = None, now=utcnow):
        self.path = Path(path) if path else state_dir() / "spend.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.now = now

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
        event = {"date": self.now().isoformat(timespec="seconds"), **event}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return event

    def runs(self) -> dict[str, dict]:
        """Per run: its reservation, pod and final events merged."""
        runs: dict[str, dict] = {}
        for e in self.events():
            r = runs.setdefault(e["run_id"], {"run_id": e["run_id"]})
            r[e["event"]] = e
        return runs

    def month_to_date(self, month: str | None = None) -> dict:
        month = month or self.now().strftime("%Y-%m")
        spent = open_est = 0.0
        finals, open_runs = [], []
        for r in self.runs().values():
            if "final" in r:
                if month_of(r["final"]["date"]) == month:
                    spent += float(r["final"]["usd"])
                    finals.append(r)
            elif "reserve" in r and month_of(r["reserve"]["date"]) == month:
                open_est += float(r["reserve"]["usd_est"])
                open_runs.append(r)
        return {"month": month, "spent": round(spent, 4), "open_estimate": round(open_est, 4),
                "total": round(spent + open_est, 4), "finals": finals, "open": open_runs}

    def reserve(self, run_id: str, **fields) -> dict:
        return self.append({"event": "reserve", "run_id": run_id, "pid": os.getpid(), **fields})

    def pod(self, run_id: str, **fields) -> dict:
        return self.append({"event": "pod", "run_id": run_id, **fields})

    def final(self, run_id: str, *, pod_type: str, hours: float, usd: float, status: str, **fields) -> dict:
        return self.append({"event": "final", "run_id": run_id, "pod_type": pod_type,
                            "hours": round(hours, 4), "usd": round(usd, 4), "status": status, **fields})
