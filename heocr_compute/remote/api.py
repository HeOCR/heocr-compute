"""A small RunPod client (REST API v2, plus GraphQL for the balance), standard library only.

The API key comes from ``RUNPOD_API_KEY`` or ``~/.runpod/config.toml`` (``apikey``), read at run time.
It is sent only in the ``Authorization`` header, never in a URL, and never printed: errors carry the
method, path and status, not the request.
"""
from __future__ import annotations

import json
import os
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.runpod.io"
CONFIG = "~/.runpod/config.toml"
USER_AGENT = "heocr-remote/0.1"


class RunPodError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def load_key(env=None, config: str = CONFIG) -> str:
    env = os.environ if env is None else env
    key = env.get("RUNPOD_API_KEY")
    if key:
        return key
    path = Path(os.path.expanduser(config))
    if path.exists():
        key = tomllib.loads(path.read_text(encoding="utf-8")).get("apikey")
        if key:
            return key
    raise RunPodError(f"no RunPod API key: set RUNPOD_API_KEY or add apikey to {config}")


def urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: float = 60):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class RunPod:
    def __init__(self, key: str, transport=urllib_transport, retries: int = 3, sleep=time.sleep):
        self._key = key
        self._transport = transport
        self._retries = retries
        self._sleep = sleep

    def __repr__(self):
        return "RunPod(key=<redacted>)"

    def _call(self, method: str, path: str, body: dict | None = None, query: dict | None = None, ok=(200, 201, 204)):
        url = API + path + ("?" + urllib.parse.urlencode(query) if query else "")
        headers = {"Authorization": f"Bearer {self._key}", "User-Agent": USER_AGENT,
                   "Accept": "application/json"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        last = None
        for attempt in range(self._retries):
            try:
                status, raw = self._transport(method, url, headers, data)
            except OSError as e:  # network trouble: retry
                last = RunPodError(f"{method} {path}: {type(e).__name__}")
                self._sleep(2 ** attempt)
                continue
            if status in ok:
                return json.loads(raw) if raw else {}
            if status == 429 or status >= 500:
                last = RunPodError(f"{method} {path}: HTTP {status}", status)
                self._sleep(2 ** attempt)
                continue
            detail = ""
            try:
                detail = json.loads(raw).get("message", "") or json.loads(raw).get("error", "")
            except (ValueError, AttributeError):
                pass
            raise RunPodError(f"{method} {path}: HTTP {status} {detail}".strip(), status)
        raise last or RunPodError(f"{method} {path}: failed")

    # account
    def balance(self) -> float:
        q = {"query": "query { myself { clientBalance } }"}
        r = self._call("POST", "/graphql", q)
        try:
            return float(r["data"]["myself"]["clientBalance"])
        except (KeyError, TypeError) as e:
            raise RunPodError("balance: unexpected response") from e

    # live prices
    def cpu_price_per_vcpu(self, flavor: str) -> float:
        for c in self._call("GET", "/v2/catalog/cpus").get("cpus", []):
            if c.get("id") == flavor:
                return float(c["price"]["securePerVcpu"])
        raise RunPodError(f"CPU flavour {flavor!r} not in the catalogue")

    def cpu_vcpu_range(self, flavor: str) -> tuple[int, int]:
        for c in self._call("GET", "/v2/catalog/cpus").get("cpus", []):
            if c.get("id") == flavor:
                return int(c["vcpu"]["min"]), int(c["vcpu"]["max"])
        raise RunPodError(f"CPU flavour {flavor!r} not in the catalogue")

    def gpu_price(self, gpu_id: str, cloud: str) -> float:
        g = self._call("GET", "/v2/catalog/gpus/" + urllib.parse.quote(gpu_id, safe=""))
        price = (g.get("price") or {}).get(cloud.lower())
        if not price:
            raise RunPodError(f"no {cloud} price for {gpu_id}")
        return float(price)

    # pods
    def create_pod(self, body: dict) -> dict:
        # Never retried: a retry after a lost response could start a second pod. The runner looks the
        # pod up by name after any failure instead.
        saved, self._retries = self._retries, 1
        try:
            return self._call("POST", "/v2/pods", body)
        finally:
            self._retries = saved

    def get_pod(self, pod_id: str) -> dict | None:
        try:
            return self._call("GET", f"/v2/pods/{urllib.parse.quote(pod_id, safe='')}")
        except RunPodError as e:
            if e.status == 404:
                return None
            raise

    def list_pods(self) -> list[dict]:
        pods, cursor = [], None
        while True:
            q = {"limit": 1000, **({"cursor": cursor} if cursor else {})}
            r = self._call("GET", "/v2/pods", query=q)
            pods += r.get("pods", [])
            page = r.get("pagination") or {}
            if not page.get("hasNextPage"):
                return pods
            cursor = page.get("nextCursor")

    def terminate(self, pod_id: str) -> None:
        try:
            self._call("DELETE", f"/v2/pods/{urllib.parse.quote(pod_id, safe='')}", ok=(200, 202, 204))
        except RunPodError as e:
            if e.status != 404:
                raise

    def billed_usd(self, pod_id: str, start: str, end: str) -> float | None:
        """What RunPod billed for one pod between two ISO times, when billing is visible to the key."""
        try:
            r = self._call("GET", "/v2/billing/pods", query={"podId": pod_id, "startTime": start,
                                                              "endTime": end, "bucketSize": "day"})
        except RunPodError:
            return None
        return float((r.get("metadata") or {}).get("totals", {}).get("totalAmount") or 0.0)
