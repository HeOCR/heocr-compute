"""A small RunPod client (REST API v2, plus GraphQL for the balance and account), standard library only.

The API key comes only from HeOCR's own key file, ``~/.config/heocr/runpod.toml`` (``apikey = "..."``),
or the file ``HEOCR_RUNPOD_CONFIG`` names. Nothing else is read: no other tool's config and no
``RUNPOD_API_KEY`` variable, since those may hold another project's key. The key is sent only in the
``Authorization`` header, never in a URL, and never printed: errors carry the method, path and status.

Before any run the runner checks that the key belongs to HeOCR's account. The expected account is in a
local file, ``~/.config/heocr/remote_account.toml`` (or ``HEOCR_REMOTE_ACCOUNT``), never committed::

    email = "..."      # the account's login address
    label = "heocr"    # how the ledger tags this account's runs
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
CONFIG = "~/.config/heocr/runpod.toml"
CONFIG_ENV = "HEOCR_RUNPOD_CONFIG"
ACCOUNT = "~/.config/heocr/remote_account.toml"
ACCOUNT_ENV = "HEOCR_REMOTE_ACCOUNT"
USER_AGENT = "heocr-remote/0.1"


class RunPodError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _config_path(env, var: str, default: str) -> Path:
    env = os.environ if env is None else env
    return Path(os.path.expanduser(env.get(var) or default))


def load_key(env=None) -> str:
    path = _config_path(env, CONFIG_ENV, CONFIG)
    if not path.is_file():
        raise RunPodError(f"no RunPod key file at {path} ({CONFIG_ENV} names another path)")
    key = tomllib.loads(path.read_text(encoding="utf-8")).get("apikey")
    if not key:
        raise RunPodError(f"{path} has no apikey")
    return key


def load_hf_token(env=None) -> str:
    """The optional Hub credential uses the same local secret file as the RunPod key."""
    path = _config_path(env, CONFIG_ENV, CONFIG)
    try:
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise RunPodError("the runner secret file must be private (mode 600) for HF publishing")
        token = tomllib.loads(path.read_text(encoding="utf-8")).get("hf_token")
    except (OSError, ValueError):
        raise RunPodError("could not read the runner secret file for HF publishing") from None
    if not isinstance(token, str) or not token.startswith("hf_") or not token[3:].isalnum():
        raise RunPodError("set hf_token in the runner secret file for HF publishing")
    return token


def load_account(env=None) -> dict:
    path = _config_path(env, ACCOUNT_ENV, ACCOUNT)
    if not path.is_file():
        raise RunPodError(f"no expected account at {path}: nothing runs until it names HeOCR's account")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    email = str(data.get("email") or "").strip().lower()
    if "@" not in email:
        raise RunPodError(f"{path} has no account email")
    return {"email": email, "label": str(data.get("label") or "heocr")}


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
                body_json = json.loads(raw)
                detail = body_json.get("message") or body_json.get("error") or json.dumps(body_json)[:300]
            except (ValueError, AttributeError, TypeError):
                detail = (raw or b"").decode("utf-8", "replace").strip()[:300]
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

    def account_email(self) -> str:
        r = self._call("POST", "/graphql", {"query": "query { myself { email } }"})
        try:
            return str(r["data"]["myself"]["email"] or "").strip().lower()
        except (KeyError, TypeError) as e:
            raise RunPodError("account: unexpected response") from e

    def check_account(self, expected: dict) -> str:
        """The account's ledger label, or RunPodError when the key belongs to any other account."""
        if self.account_email() != expected["email"]:
            raise RunPodError("the RunPod key is not for the configured account: refused")
        return expected["label"]

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
