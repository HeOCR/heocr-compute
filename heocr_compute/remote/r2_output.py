"""Publish bounded training artifacts to per-run, create-only R2 objects."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import urllib.error
import urllib.request

from . import hf_job, r2_input

ENV = "HEOCR_R2_PUT_URLS"
STARTUP = "startup.json"
TERMINAL = "receipt.json"
RECEIPT_LIMIT = 64 * 1024
URLS_LIMIT = 96 * 1024
HEADERS = {"Content-Type": "application/octet-stream", "If-None-Match": "*"}
SIGNED_HEADERS = "content-type;host;if-none-match"


def progress_name(number):
    return f"progress/{number:03d}.json"


def object_key(run_id, name):
    hf_job.require(hf_job.matches(hf_job.RUN_ID, run_id), "invalid R2 run ID")
    return f"training-runs/{run_id}/{hf_job.safe_path(name)}"


def object_names(job):
    return (STARTUP, TERMINAL, *(progress_name(i) for i in range(1, len(job["snapshots"]) + 1)),
            *hf_job.output_limits(job))


def check_urls(job, run_id, raw, *, now=None, min_remaining=0):
    try:
        if not isinstance(raw, str) or len(raw) > URLS_LIMIT:
            raise ValueError
        urls = json.loads(raw)
        if not isinstance(urls, dict) or urls.keys() != set(object_names(job)):
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("supply exactly the declared objects in the R2 PUT secret environment") from None
    now = now or dt.datetime.now(dt.timezone.utc)
    for name, url in urls.items():
        pin = {"account": job["output"]["account"], "path": object_key(run_id, name)}
        r2_input.check_url(pin, url, now=now, min_remaining=min_remaining,
                           signed_headers=SIGNED_HEADERS, method="PUT")
    return urls


class Publisher:
    """Upload a checkpoint's files before publishing its independently named receipt."""

    def __init__(self, urls, job):
        self.urls = urls
        self.checkpoints = {spec["name"] for spec in job["snapshots"]}
        self.opener = urllib.request.build_opener(r2_input.NoRedirect(), urllib.request.ProxyHandler({}))

    def put(self, name, data, size):
        for _ in range(2):
            if hasattr(data, "seek"):
                data.seek(0)
            request = urllib.request.Request(self.urls[name], method="PUT", data=data,
                                             headers={**HEADERS, "Content-Length": str(size)})
            try:
                with self.opener.open(request, timeout=10) as response:
                    if response.status == 200:
                        return
                    break
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                if code not in (408, 429) and not 500 <= code <= 599:
                    break
            except OSError:
                continue
            except ValueError:
                break
        raise hf_job.PublishError("private R2 output PUT could not be confirmed") from None

    def commit(self, receipt, artifacts=()):
        for path in artifacts:
            with path.open("rb") as stream:
                self.put(path.name, stream, os.fstat(stream.fileno()).st_size)
        if receipt["status"] != "running":
            name = TERMINAL
        else:
            number = sum(record["name"] in self.checkpoints for record in receipt["snapshots"])
            name = progress_name(number) if number else STARTUP
        raw = json.dumps(receipt, sort_keys=True).encode("utf-8")
        hf_job.require(len(raw) <= RECEIPT_LIMIT, "R2 receipt exceeds its declared limit")
        self.put(name, raw, len(raw))


def cloudflare_token(expected_account):
    """Resolve the account-bound API token from its existing local credential file."""
    path = Path.home() / ".config/heocr/cloudflare.env"
    try:
        if not path.is_file() or path.stat().st_mode & 0o777 != 0o600:
            raise ValueError
        config = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = shlex.split(line, comments=True)
            if fields[:1] == ["export"]:
                fields = fields[1:]
            if len(fields) == 1 and "=" in fields[0]:
                key, value = fields[0].split("=", 1)
                config[key] = value
        account, token = config["CLOUDFLARE_ACCOUNT_ID"], config["CLOUDFLARE_API_TOKEN"]
        if account != expected_account or not hf_job.matches(hf_job.R2_ACCOUNT, account) or not token:
            raise ValueError
    except (KeyError, OSError, ValueError):
        raise ValueError("expected the approved R2 account in the local mode-600 Cloudflare credential file") from None

    opener = urllib.request.build_opener(r2_input.NoRedirect(), urllib.request.ProxyHandler({}))
    for suffix in (f"/accounts/{account}/tokens/verify", "/user/tokens/verify"):
        request = urllib.request.Request("https://api.cloudflare.com/client/v4" + suffix,
                                         headers={"Authorization": f"Bearer {token}"})
        try:
            with opener.open(request, timeout=10) as response:
                result = json.loads(response.read(8192))["result"]
            if result["status"] == "active" and hf_job.matches(hf_job.R2_ACCOUNT, result["id"]):
                return result["id"], hashlib.sha256(token.encode()).hexdigest()
        except (KeyError, OSError, TypeError, ValueError):
            continue
    raise ValueError("could not verify the local Cloudflare token") from None


class LocalR2:
    """Sign upload URLs and read exact objects on the operator's machine."""

    def __init__(self, account):
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ValueError("install the local R2 client with pip install '.[r2]'") from None
        key, secret = cloudflare_token(account)
        try:
            self.client = boto3.client(
                "s3", endpoint_url=f"https://{account}.r2.cloudflarestorage.com", region_name="auto",
                aws_access_key_id=key, aws_secret_access_key=secret,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                              proxies={}, connect_timeout=10, read_timeout=20,
                              request_checksum_calculation="when_required",
                              response_checksum_validation="when_required"),
            )
        except Exception:
            raise ValueError("could not initialize the local R2 client") from None

    def put_urls(self, job, run_id, *, now, min_remaining):
        try:
            urls = {name: self.client.generate_presigned_url(
                "put_object", Params={"Bucket": r2_input.BUCKET, "Key": object_key(run_id, name),
                                      "ContentType": HEADERS["Content-Type"], "IfNoneMatch": "*"},
                ExpiresIn=r2_input.MAX_LIFETIME, HttpMethod="PUT",
            ) for name in object_names(job)}
        except Exception:
            raise ValueError("could not sign the approved R2 output keys") from None
        raw = json.dumps(urls, separators=(",", ":"))
        check_urls(job, run_id, raw, now=now, min_remaining=min_remaining)
        return raw

    def fetch(self, run_id, name, destination, limit, *, missing_ok=False):
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=r2_input.BUCKET, Key=object_key(run_id, name))
        except ClientError as exc:
            if missing_ok and exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise ValueError("could not read the private R2 output object") from None
        except Exception:
            raise ValueError("could not read the private R2 output object") from None
        try:
            with response["Body"] as stream:
                size = response["ContentLength"]
                hf_job.require(type(size) is int and 0 < size <= limit, "R2 output exceeds its declared limit")
                written = 0
                with destination.open("xb") as file:
                    while data := stream.read(1024 * 1024):
                        written += len(data)
                        hf_job.require(written <= size, "R2 output exceeds its declared length")
                        file.write(data)
                hf_job.require(written == size, "private R2 output is truncated")
        except Exception:
            raise ValueError("could not read a bounded private R2 output object") from None
        return destination
