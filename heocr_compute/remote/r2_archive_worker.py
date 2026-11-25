"""Pod-local supervisor for one private, verified tar return."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import urllib.error
import urllib.request

from . import r2_input

ARCHIVE = "output.tar"
RECEIPT = "manifest.json"
SCHEMA = "heocr_r2_archive.v1"
MAX_ARCHIVE = 2 << 30
MAX_RECEIPT = 8192
RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")
PUT_HEADERS = {"Content-Type": "application/octet-stream", "If-None-Match": "*"}
PUT_SIGNED = "content-type;host;if-none-match"
OPERATIONS = ("PUT", "HEAD", "GET")


def object_key(run_id, name):
    if not RUN_ID.fullmatch(run_id) or name not in (ARCHIVE, RECEIPT):
        raise ValueError("invalid archive run or object name")
    return f"runs/{run_id}/{name}"


def verify_urls(scope, run_id, urls, *, now=None, min_remaining=0):
    if not isinstance(urls, dict) or urls.keys() != {ARCHIVE, RECEIPT}:
        raise ValueError("archive secrets must identify exactly two output objects")
    now = now or dt.datetime.now(dt.timezone.utc)
    for name, methods in urls.items():
        if not isinstance(methods, dict) or methods.keys() != set(OPERATIONS):
            raise ValueError("archive secrets must include PUT, HEAD and GET")
        for method, url in methods.items():
            r2_input.check_url({"account": scope["account"], "path": object_key(run_id, name)}, url,
                               now=now, min_remaining=min_remaining, bucket=scope["bucket"], method=method,
                               signed_headers=PUT_SIGNED if method == "PUT" else "host")


def archive_size(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ValueError("output must be a regular tar file")
    size = path.stat().st_size
    if not 0 < size <= limit <= MAX_ARCHIVE:
        raise ValueError("archive is empty or exceeds the bounded size")
    seen, files = set(), 0
    try:
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (name.is_absolute() or ".." in name.parts or name in seen
                        or not (member.isfile() or member.isdir())):
                    raise ValueError("archive contains an unsafe or repeated member")
                seen.add(name)
                files += member.isfile()
    except tarfile.TarError:
        raise ValueError("output must be an uncompressed tar") from None
    if not files:
        raise ValueError("archive must contain at least one regular file")
    return size


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _call(opener, method, url, *, data=None, size=None):
    try:
        headers = {**PUT_HEADERS, "Content-Length": str(size)} if method == "PUT" else {}
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        return opener.open(request, timeout=30)
    except urllib.error.HTTPError as error:
        error.close()
        raise ValueError(f"private archive {method} failed") from None
    except (OSError, ValueError):
        raise ValueError(f"private archive {method} failed") from None


def upload_and_verify(opener, methods, data, size, expected):
    with _call(opener, "PUT", methods["PUT"], data=data, size=size) as response:
        if response.status != 200:
            raise ValueError("private archive PUT was unconfirmed")
    with _call(opener, "HEAD", methods["HEAD"]) as response:
        if response.status != 200 or response.headers.get("Content-Length") != str(size):
            raise ValueError("private archive HEAD size differs")
    digest, seen = hashlib.sha256(), 0
    with _call(opener, "GET", methods["GET"]) as response:
        if response.status != 200 or response.headers.get("Content-Length") != str(size):
            raise ValueError("private archive GET size differs")
        while chunk := response.read(1024 * 1024):
            seen += len(chunk)
            if seen > size:
                raise ValueError("private archive readback exceeds its size")
            digest.update(chunk)
    if seen != size or digest.hexdigest() != expected:
        raise ValueError("private archive readback SHA-256 differs")


def publish(root, settings, opener=None):
    scope, urls = settings["scope"], settings["urls"]
    run_id = settings["run_id"]
    verify_urls(scope, run_id, urls)
    opener = opener or urllib.request.build_opener(r2_input.NoRedirect(), urllib.request.ProxyHandler({}))
    path = root / "out" / ARCHIVE
    size = archive_size(path, scope["max_bytes"])
    sha = sha_file(path)
    with path.open("rb") as data:
        upload_and_verify(opener, urls[ARCHIVE], data, size, sha)
    receipt = {"schema": SCHEMA, "run_id": run_id, "pod_id": settings["pod_id"], "archive": ARCHIVE,
               "size": size, "sha256": sha, "status": "ok",
               "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    upload_and_verify(opener, urls[RECEIPT], raw, len(raw), hashlib.sha256(raw).hexdigest())


def supervise(root):
    secret = root / ".archive-secret"
    try:
        try:
            with secret.open("rb") as stream:
                settings = json.loads(stream.read(32769))
        finally:
            secret.unlink(missing_ok=True)
        verify_urls(settings["scope"], settings["run_id"], settings["urls"],
                    min_remaining=settings["seconds"])
        # The renderer receives only ordinary runtime configuration.
        allowed = ("PATH", "LANG", "LC_ALL", "HOME", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                   "NVIDIA_VISIBLE_DEVICES")
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        (root / "archive.ready").touch(exist_ok=False)
        with (root / "out" / "job.log").open("wb") as log:
            subprocess.run(["timeout", "--kill-after=10", str(settings["seconds"]), "bash", "-lc", settings["cmd"]],
                           cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=True)
        publish(root, settings)
    except Exception:
        # Signed URLs and provider exceptions never reach the job log or console.
        print("Autonomous archive failed before a verified receipt was published.", flush=True)
        return 1
    finally:
        subprocess.run(["sh", str(root / "deadman.sh"), "now"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(supervise(Path("/heocr")))
