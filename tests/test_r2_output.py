"""Offline coverage for the private R2 output protocol."""
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error
from urllib.parse import quote, urlencode

from heocr_compute.remote import hf_job, hf_train, r2_input, r2_output


RUN = "20261123-001122-a123"
NOW = dt.datetime(2026, 11, 23, tzinfo=dt.timezone.utc)
ACCOUNT = "a" * 32
JOB = {
    "output": {"transport": "r2", "account": ACCOUNT},
    "snapshots": [{"name": "model.pt", "max_bytes": 100}],
    "telemetry": {"run_log_max_bytes": 100, "setup_log_max_bytes": 100,
                  "metrics_max_bytes": 100, "trainer_log_max_bytes": 100},
}


def create_urls(method="PutObject", seconds=3600, signed_headers=r2_output.SIGNED_HEADERS,
                bucket=r2_input.BUCKET):
    urls = {}
    for name in r2_output.object_names(JOB):
        query = urlencode({
            "X-Amz-Credential": "invented/20261123/auto/s3/aws4_request",
            "X-Amz-Date": NOW.strftime("%Y%m%dT%H%M%SZ"),
            "X-Amz-Expires": seconds,
            "X-Amz-SignedHeaders": signed_headers,
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Signature": "f" * 64,
            "x-id": method,
        })
        key = quote(r2_output.object_key(RUN, name))
        urls[name] = f"https://{ACCOUNT}.r2.cloudflarestorage.com/{bucket}/{key}?{query}"
    return urls


class R2OutputTest(unittest.TestCase):
    def test_signed_urls_bind_conditional_put_exact_keys_and_deadline(self):
        urls = create_urls()
        self.assertEqual(r2_output.check_urls(JOB, RUN, json.dumps(urls), now=NOW, min_remaining=3599), urls)
        for bad in (
            {**urls, r2_output.TERMINAL: urls[r2_output.STARTUP]},
            {**urls, "extra.tar": urls[r2_output.STARTUP]},
            create_urls(method="GetObject"),
            create_urls(seconds=86401),
            create_urls(signed_headers="host"),
        ):
            with self.assertRaises(ValueError) as caught:
                r2_output.check_urls(JOB, RUN, json.dumps(bad), now=NOW)
            self.assertNotIn("X-Amz", str(caught.exception))
        with self.assertRaisesRegex(ValueError, "expired"):
            r2_output.check_urls(JOB, RUN, json.dumps(urls), now=NOW, min_remaining=3600)

    def test_named_bucket_rejects_default_and_cross_bucket_signatures(self):
        bucket = "invented-output"
        job = {**JOB, "output": {**JOB["output"], "bucket": bucket}}
        urls = create_urls(bucket=bucket)
        self.assertEqual(r2_output.check_urls(job, RUN, json.dumps(urls), now=NOW), urls)
        for bad in (create_urls(), create_urls(bucket="other-output")):
            with self.assertRaisesRegex(ValueError, "mismatched presigned R2"):
                r2_output.check_urls(job, RUN, json.dumps(bad), now=NOW)

    def test_scoped_credentials_require_exact_scope_and_safe_file(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(Path, "home", return_value=Path(temp)):
            config_dir = Path(temp) / ".config/heocr"
            config_dir.mkdir(parents=True)
            path = config_dir / "r2-output.env"
            key, secret = "b" * 32, "c" * 64
            path.write_text(f"R2_ACCOUNT_ID={ACCOUNT}\nR2_BUCKET=invented-output\n"
                            f"R2_ACCESS_KEY_ID={key}\nR2_SECRET_ACCESS_KEY={secret}\n")
            path.chmod(0o600)
            self.assertEqual(r2_output.scoped_credentials(ACCOUNT, "invented-output"), (key, secret))
            for account, bucket in ((ACCOUNT, "other-output"), ("d" * 32, "invented-output")):
                with self.assertRaisesRegex(ValueError, "matching bucket-scoped"):
                    r2_output.scoped_credentials(account, bucket)
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                r2_output.scoped_credentials(ACCOUNT, "invented-output")
            path.chmod(0o600)
            original = path.with_suffix(".saved")
            path.rename(original)
            path.symlink_to(original)
            with self.assertRaises(ValueError):
                r2_output.scoped_credentials(ACCOUNT, "invented-output")

    def test_streams_create_only_put_and_redacts_provider_error(self):
        urls = create_urls()
        with patch.object(r2_output.urllib.request, "build_opener") as opener:
            publisher = r2_output.Publisher(urls, JOB)
        response = opener.return_value.open.return_value.__enter__.return_value
        response.status = 200
        stream = io.BytesIO(b"fresh")
        publisher.put("model.pt", stream, 5)
        req = opener.return_value.open.call_args.args[0]
        self.assertEqual(req.get_method(), "PUT")
        self.assertEqual({key.lower(): value for key, value in req.headers.items()}, {
            "content-type": "application/octet-stream", "content-length": "5", "if-none-match": "*"})
        self.assertIs(req.data, stream)
        fault = urllib.error.HTTPError(urls["model.pt"], 412, "invented-sensitive-url", {}, io.BytesIO())
        opener.return_value.open.reset_mock(side_effect=True)
        opener.return_value.open.side_effect = fault
        with self.assertRaisesRegex(hf_job.PublishError, "^private R2 output PUT could not be confirmed$"):
            publisher.put("model.pt", stream, 5)
        self.assertEqual(opener.return_value.open.call_count, 1)

    def test_orders_outputs_before_progress_and_keeps_startup_distinct(self):
        publisher = r2_output.Publisher(create_urls(), JOB)
        names = []
        publisher.put = lambda name, _data, _size: names.append(name)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.pt"
            model.write_bytes(b"fresh")
            publisher.commit({"status": "running", "snapshots": []})
            publisher.commit({"status": "running", "snapshots": [{"name": model.name}]}, [model])
            publisher.commit({"status": "ok", "snapshots": [{"name": model.name}]})
        self.assertEqual(names, ["startup.json", "model.pt", "progress/001.json", "receipt.json"])

    def test_collect_recovers_latest_progress_and_checks_hash(self):
        data = b"fresh"
        records = [{"name": "model.pt", "size": len(data), "sha256": hf_job.digest(data)}]
        metadata = {"status": "running", "snapshots": records}
        objects = {"progress/001.json": json.dumps(metadata).encode(), "model.pt": data}
        r2 = Mock()

        def fetch(_run, name, path, limit, *, missing_ok=False):
            blob = objects.get(name)
            if blob is None and missing_ok:
                return None
            self.assertLessEqual(len(blob), limit)
            path.write_bytes(blob)
            return path

        r2.fetch.side_effect = fetch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt, key = hf_train.read_r2_receipt(r2, RUN, JOB, root)
            self.assertEqual(key, r2_output.object_key(RUN, "progress/001.json"))
            self.assertEqual(receipt, metadata)

            def download(name, limit):
                return r2.fetch(RUN, name, root / name, limit)

            output = root / "whole"
            hf_train.materialize_from(download, receipt, output, root)
            self.assertEqual((output / "model.pt").read_bytes(), data)
            objects["model.pt"] = b"bogus"
            other = root / "bad"
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                hf_train.materialize_from(download, receipt, other, root)
            self.assertFalse(other.exists())
