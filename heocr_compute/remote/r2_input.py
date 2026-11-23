"""Read exact TRAIN archives from short-lived, object-scoped Cloudflare R2 GETs."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
from urllib.parse import parse_qsl, quote, urlsplit

BUCKET = "heocr-train-private"
ENV = "HEOCR_R2_URLS"
MAX_LIFETIME = 24 * 60 * 60


def check_url(pin, url, *, now, min_remaining, signed_headers="host", method="GET"):
    """Match the endpoint, object and lifetime independently of the signed URL."""
    try:
        if not isinstance(url, str) or not url.isascii() or len(url) > 4096 or not url.isprintable():
            raise ValueError
        parsed = urlsplit(url)
        pairs = parse_qsl(parsed.query, strict_parsing=True, max_num_fields=16)
        query = dict(pairs)
        start = dt.datetime.strptime(query["X-Amz-Date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
        ttl = int(query["X-Amz-Expires"])
        claims = query["X-Amz-Credential"].split("/")
        signature = query["X-Amz-Signature"]
        host = f"{pin['account']}.r2.cloudflarestorage.com"
        valid = (
            parsed.scheme == "https" and parsed.netloc == host and not parsed.fragment
            and parsed.path == f"/{BUCKET}/{quote(pin['path'], safe='/')}"
            and len(query) == len(pairs) and query["X-Amz-Algorithm"] == "AWS4-HMAC-SHA256"
            and query["X-Amz-SignedHeaders"] == signed_headers and len(signature) == 64
            and query.get("x-id", {"GET": "GetObject", "PUT": "PutObject"}[method])
            == {"GET": "GetObject", "PUT": "PutObject"}[method]
            and all(char in "0123456789abcdef" for char in signature)
            and len(claims) == 5 and claims[-4:] == [start.strftime("%Y%m%d"), "auto", "s3", "aws4_request"]
            and 0 < ttl <= MAX_LIFETIME and start <= now + dt.timedelta(minutes=1)
            and start + dt.timedelta(seconds=ttl) > now + dt.timedelta(seconds=min_remaining)
        )
    except (ValueError, TypeError, KeyError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"invalid, expired or mismatched presigned R2 {method}") from None


def read_urls(job, raw, *, now=None, min_remaining=0):
    pins = {item["target"]: item for item in job["inputs"] if item.get("transport") == "r2"}
    if not pins:
        return {}
    try:
        if not isinstance(raw, str) or len(raw) > 64 * 1024:
            raise ValueError
        urls = json.loads(raw)
        if not isinstance(urls, dict) or urls.keys() != pins.keys():
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("supply exactly the presigned R2 GETs for this job in the secret environment") from None
    now = now or dt.datetime.now(dt.timezone.utc)
    for target, pin in pins.items():
        check_url(pin, urls[target], now=now, min_remaining=min_remaining)
    return urls


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download(pin, url, path):
    """Bound size during download and expose the file only after its SHA-256 matches."""
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({}))
    temporary = None
    written, digest = 0, hashlib.sha256()
    try:
        with opener.open(urllib.request.Request(url, method="GET"), timeout=30) as response:
            if response.status != 200 or response.headers.get("Content-Length") != str(pin["size"]):
                raise ValueError
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > pin["size"]:
                        raise ValueError
                    digest.update(chunk)
                    output.write(chunk)
        if written != pin["size"] or digest.hexdigest() != pin["sha256"]:
            raise ValueError
        temporary.rename(path)
    except Exception:
        raise ValueError("could not download and verify the approved R2 archive") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
