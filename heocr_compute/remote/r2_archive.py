"""Sign only the two run-scoped R2 objects and verify their collected bytes."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import tempfile

from . import hf_job
from .r2_archive_worker import (ARCHIVE, MAX_ARCHIVE, MAX_RECEIPT, OPERATIONS, PUT_HEADERS, RECEIPT,
                                SCHEMA, archive_size, object_key, sha_file, verify_urls)


def _credentials(path):
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError
            config = {}
            for line in stream:
                parts = shlex.split(line, comments=True)
                if parts[:1] == ["export"]:
                    parts = parts[1:]
                if not parts:
                    continue
                if len(parts) != 1 or "=" not in parts[0]:
                    raise ValueError
                key, value = parts[0].split("=", 1)
                if key in config:
                    raise ValueError
                config[key] = value
        endpoint, bucket = config["R2_ENDPOINT"].rstrip("/"), config["R2_BUCKET"]
        identity = re.fullmatch(r"https://([0-9a-f]{32})\.r2\.cloudflarestorage\.com", endpoint)
        key, secret = config["R2_ACCESS_KEY_ID"], config["R2_SECRET_ACCESS_KEY"]
        if not (identity and hf_job.R2_BUCKET.fullmatch(bucket)
                and hf_job.R2_ACCOUNT.fullmatch(key) and hf_job.SHA.fullmatch(secret)):
            raise ValueError
        return endpoint, identity[1], bucket, key, secret
    except (KeyError, OSError, ValueError):
        raise ValueError("expected a mode-600 bucket-scoped R2 credentials file") from None


class ArchiveStore:
    def __init__(self, credentials, max_bytes):
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_ARCHIVE:
            raise ValueError("R2 archive limit must be at most 2 GiB")
        endpoint, account, bucket, key, secret = _credentials(Path(credentials).expanduser())
        self.scope = {"account": account, "bucket": bucket, "max_bytes": max_bytes}
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ValueError("install the local R2 client with pip install '.[r2]'") from None
        try:
            self.client = boto3.client(
                "s3", endpoint_url=endpoint, region_name="auto",
                aws_access_key_id=key, aws_secret_access_key=secret,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                              proxies={}, connect_timeout=10, read_timeout=30,
                              request_checksum_calculation="when_required",
                              response_checksum_validation="when_required"),
            )
        except Exception:
            raise ValueError("could not initialize the private archive client") from None

    def urls(self, run_id, lifetime, now):
        if not 0 < lifetime <= 24 * 3600:
            raise ValueError("archive URL lifetime must fit within 24 hours")
        operations = {"PUT": "put_object", "HEAD": "head_object", "GET": "get_object"}
        urls = {}
        try:
            for name in (ARCHIVE, RECEIPT):
                args = {"Bucket": self.scope["bucket"], "Key": object_key(run_id, name)}
                urls[name] = {}
                for method in OPERATIONS:
                    params = ({**args, "ContentType": PUT_HEADERS["Content-Type"], "IfNoneMatch": "*"}
                              if method == "PUT" else args)
                    urls[name][method] = self.client.generate_presigned_url(
                        operations[method], Params=params, ExpiresIn=lifetime, HttpMethod=method)
        except Exception:
            raise ValueError("could not sign the approved private archive objects") from None
        verify_urls(self.scope, run_id, urls, now=now, min_remaining=lifetime - 30)
        return urls

    def fetch(self, run_id, name, destination, limit, *, size=None, digest=None, missing_ok=False):
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.scope["bucket"], Key=object_key(run_id, name))
        except ClientError as error:
            if missing_ok and error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise ValueError("could not fetch the private archive object") from None
        except Exception:
            raise ValueError("could not fetch the private archive object") from None
        try:
            checked_size, seen, checksum = response["ContentLength"], 0, hashlib.sha256()
            with response["Body"] as stream:
                if not (type(checked_size) is int and 0 < checked_size <= limit
                        and (size is None or size == checked_size)):
                    raise ValueError
                with destination.open("xb") as output:
                    while chunk := stream.read(1024 * 1024):
                        seen += len(chunk)
                        if seen > checked_size:
                            raise ValueError
                        checksum.update(chunk)
                        output.write(chunk)
            if seen != checked_size or (digest is not None and checksum.hexdigest() != digest):
                raise ValueError
        except Exception:
            raise ValueError("private archive length or SHA-256 could not be verified") from None
        return True


def validated_receipt(path, run_id, pod_id, max_bytes, *, opened=None, deadline=None):
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw)
        if not isinstance(receipt, dict) or receipt.keys() != {
            "schema", "run_id", "pod_id", "archive", "size", "sha256", "status", "finished_utc"
        }:
            raise ValueError
        finished = dt.datetime.fromisoformat(receipt["finished_utc"])
        if not (finished.tzinfo is not None and len(raw) <= MAX_RECEIPT
                and (opened is None or dt.datetime.fromisoformat(opened) <= finished)
                and (deadline is None or finished <= dt.datetime.fromisoformat(deadline))
                and receipt["schema"] == SCHEMA and receipt["status"] == "ok"
                and receipt["run_id"] == run_id and receipt["pod_id"] == pod_id
                and receipt["archive"] == ARCHIVE and type(receipt["size"]) is int
                and 0 < receipt["size"] <= max_bytes and hf_job.matches(hf_job.SHA, receipt["sha256"])):
            raise ValueError
    except (KeyError, ValueError, TypeError):
        raise ValueError("private archive receipt does not match the reserved run") from None
    return receipt


def collect(api, ledger, run_id, credentials, output, *, out=print):
    from .hf_train import settle_collection

    object_key(run_id, ARCHIVE)
    run = ledger.runs().get(run_id, {})
    reserve = run.get("reserve", {})
    scope, pod_id = reserve.get("r2_archive"), run.get("pod", {}).get("pod_id")
    if not scope or not pod_id or reserve.get("account") != ledger.account:
        raise ValueError("no known private archive reservation for this account")
    pod = api.get_pod(pod_id)
    if pod is not None and (pod.get("id") != pod_id or pod.get("status") != "TERMINATED"):
        raise ValueError("collect the private archive after the exact pod is gone")
    store = ArchiveStore(credentials, scope["max_bytes"])
    if store.scope != scope:
        raise ValueError("R2 credential file differs from the reserved archive destination")

    output.mkdir(parents=True, exist_ok=True)
    dest = output / run_id
    with tempfile.TemporaryDirectory(prefix=".archive-collect-", dir=output) as temp:
        pending = Path(temp) / "verified"
        pending.mkdir()
        receipt_path = pending / RECEIPT
        if not store.fetch(run_id, RECEIPT, receipt_path, MAX_RECEIPT, missing_ok=True):
            settle_collection(api, ledger, run_id, run, pod_id, None, None)
            out("The pod is gone and no verified archive receipt was found.")
            return 1
        receipt = validated_receipt(receipt_path, run_id, pod_id, scope["max_bytes"],
                                    opened=reserve["date"], deadline=reserve["deadman_utc"])
        if shutil.disk_usage(pending).free < receipt["size"] + (1 << 30):
            raise ValueError("not enough local disk to collect the verified archive")
        archive = pending / ARCHIVE
        store.fetch(run_id, ARCHIVE, archive, scope["max_bytes"], size=receipt["size"], digest=receipt["sha256"])
        archive_size(archive, scope["max_bytes"])
        if dest.exists() or dest.is_symlink():
            if (dest.is_symlink() or not dest.is_dir() or {p.name for p in dest.iterdir()} != {ARCHIVE, RECEIPT}
                    or any((dest / name).is_symlink() or not (dest / name).is_file()
                           or sha_file(dest / name) != sha_file(pending / name) for name in (ARCHIVE, RECEIPT))):
                raise ValueError("local archive destination holds different data")
        else:
            pending.rename(dest)
    settle_collection(api, ledger, run_id, run, pod_id, object_key(run_id, RECEIPT), receipt)
    out(f"Collected the verified archive and SHA-256 manifest into {dest}.")
    return 0
