import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from urllib.parse import urlencode, urlsplit
from unittest import mock

from heocr_compute.remote import r2_archive
from heocr_compute.remote import r2_archive_worker as worker
from heocr_compute.remote.ledger import Ledger

RUN_ID = "20261125-120000-abcd"
SCOPE = {"account": "1" * 32, "bucket": "invented-private-output", "max_bytes": 20_480}


def urls():
    now = dt.datetime.now(dt.timezone.utc)
    routes = {}
    for name in (worker.ARCHIVE, worker.RECEIPT):
        routes[name] = {}
        for method, operation in zip(worker.OPERATIONS, ("PutObject", "HeadObject", "GetObject")):
            query = {
                "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
                "X-Amz-Credential": f"{'2' * 32}/{now:%Y%m%d}/auto/s3/aws4_request",
                "X-Amz-Date": f"{now:%Y%m%dT%H%M%SZ}",
                "X-Amz-Expires": 3600,
                "X-Amz-SignedHeaders": worker.PUT_SIGNED if method == "PUT" else "host",
                "X-Amz-Signature": "3" * 64,
                "x-id": operation,
            }
            key = worker.object_key(RUN_ID, name)
            routes[name][method] = (
                f"https://{SCOPE['account']}.r2.cloudflarestorage.com/{SCOPE['bucket']}/{key}?{urlencode(query)}"
            )
    return routes


class FakeR2:
    def __init__(self, corrupt=False, wrong_head=False):
        self.calls, self.objects = [], {}
        self.corrupt = corrupt
        self.wrong_head = wrong_head

    def open(self, request, *, timeout):
        method, name = request.get_method(), urlsplit(request.full_url).path.rsplit("/", 1)[1]
        self.calls.append((method, name))
        if method == "PUT":
            data = request.data
            self.objects[name] = data if isinstance(data, bytes) else data.read()
            assert request.get_header("If-none-match") == "*"
            assert request.get_header("Content-length") == str(len(self.objects[name]))
        data = self.objects[name]
        length = len(data) + (self.wrong_head and method == "HEAD")
        if self.corrupt and method == "GET":
            data = b"!" + data[1:]
        response = io.BytesIO(data if method == "GET" else b"")
        response.status = 200
        response.headers = {"Content-Length": str(length)}
        return response


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "out").mkdir()
        self.archive = self.root / "out" / worker.ARCHIVE
        self.settings = {
            "scope": SCOPE, "urls": urls(), "run_id": RUN_ID, "pod_id": "invented-pod",
            "seconds": 60, "cmd": "bash in/render.sh",
        }
        with tarfile.open(self.archive, "w") as archive:
            member = tarfile.TarInfo("synthetic.txt")
            data = b"invented synthetic example"
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    def test_publish_checks_both_exact_objects_in_order(self):
        opener = FakeR2()
        worker.publish(self.root, self.settings, opener)
        self.assertEqual(opener.calls, [(method, name) for name in (worker.ARCHIVE, worker.RECEIPT)
                                        for method in worker.OPERATIONS])
        manifest = self.root / worker.RECEIPT
        manifest.write_bytes(opener.objects[worker.RECEIPT])
        receipt = r2_archive.validated_receipt(manifest, RUN_ID, "invented-pod", SCOPE["max_bytes"])
        self.assertEqual(receipt["size"], self.archive.stat().st_size)
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.archive.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "reserved run"):
            r2_archive.validated_receipt(manifest, RUN_ID, "another-pod", SCOPE["max_bytes"])

    def test_failed_head_or_same_size_corruption_does_not_publish_manifest(self):
        for opener in (FakeR2(wrong_head=True), FakeR2(corrupt=True)):
            with self.subTest(opener=opener):
                with self.assertRaisesRegex(ValueError, "HEAD size|SHA-256"):
                    worker.publish(self.root, self.settings, opener)
                self.assertEqual(set(opener.objects), {worker.ARCHIVE})

    def test_rejects_wrong_object_or_method_before_network(self):
        for replacement in (self.settings["urls"][worker.RECEIPT]["HEAD"],
                            self.settings["urls"][worker.ARCHIVE]["GET"]):
            with self.subTest():
                signed = urls()
                signed[worker.ARCHIVE]["HEAD"] = replacement
                opener = FakeR2()
                with self.assertRaisesRegex(ValueError, "mismatched"):
                    worker.publish(self.root, {**self.settings, "urls": signed}, opener)
                self.assertEqual(opener.calls, [])

    def test_rejects_traversal_empty_or_invalid_tar(self):
        with tarfile.open(self.archive, "w"):
            pass
        with self.assertRaisesRegex(ValueError, "at least one"):
            worker.archive_size(self.archive, SCOPE["max_bytes"])
        with tarfile.open(self.archive, "w") as archive:
            archive.addfile(tarfile.TarInfo("../escape"))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            worker.archive_size(self.archive, SCOPE["max_bytes"])
        self.archive.write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "uncompressed tar"):
            worker.archive_size(self.archive, SCOPE["max_bytes"])

    def test_collection_never_exposes_corrupt_bytes_and_is_idempotent(self):
        opener = FakeR2()
        worker.publish(self.root, self.settings, opener)
        objects = dict(opener.objects)
        ledger = Ledger(self.root / "ledger.jsonl", account="invented")
        now = dt.datetime.now(dt.timezone.utc)
        ledger.reserve(RUN_ID, pod_type="cpu", r2_archive=SCOPE, output={"transport": "r2"},
                       date=(now - dt.timedelta(minutes=1)).isoformat(),
                       deadman_utc=(now + dt.timedelta(minutes=1)).isoformat())
        ledger.pod(RUN_ID, pod_id="invented-pod")
        store = object.__new__(r2_archive.ArchiveStore)
        store.scope = SCOPE

        def get_object(*, Bucket, Key):
            name = Key.rsplit("/", 1)[1]
            self.assertEqual(Bucket, SCOPE["bucket"])
            self.assertEqual(Key, worker.object_key(RUN_ID, name))
            raw = objects[name]
            return {"ContentLength": len(raw), "Body": io.BytesIO(raw)}

        store.client = SimpleNamespace(get_object=get_object)
        api = SimpleNamespace(get_pod=mock.Mock(return_value={"id": "invented-pod", "status": "RUNNING"}),
                              billed_usd=mock.Mock(return_value=0.02))
        output = self.root / "results"
        dest = output / RUN_ID
        with mock.patch.object(r2_archive, "ArchiveStore", return_value=store), mock.patch.dict(
            "sys.modules", {"botocore.exceptions": SimpleNamespace(ClientError=type("ClientError", (Exception,), {}))}
        ):
            with self.assertRaisesRegex(ValueError, "after the exact pod is gone"):
                r2_archive.collect(api, ledger, RUN_ID, "invented", output)
            api.get_pod.return_value = None
            original = objects[worker.ARCHIVE]
            objects[worker.ARCHIVE] = b"!" + original[1:]
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                r2_archive.collect(api, ledger, RUN_ID, "invented", output)
            self.assertEqual(list(output.iterdir()), [])
            objects[worker.ARCHIVE] = original
            for _ in range(2):
                self.assertEqual(r2_archive.collect(api, ledger, RUN_ID, "invented", output, out=lambda _: None), 0)
            self.assertEqual((dest / worker.ARCHIVE).read_bytes(), original)
            self.assertEqual((dest / worker.RECEIPT).read_bytes(), objects[worker.RECEIPT])
            self.assertEqual(ledger.runs()[RUN_ID]["final"]["status"], "ok")
            self.assertEqual(api.billed_usd.call_count, 1)

    def test_supervisor_unlinks_secret_before_child_and_terminates_after_publish(self):
        secret = self.root / ".archive-secret"
        secret.write_text(json.dumps(self.settings))
        events = []

        def run(args, **kwargs):
            if args[0] == "timeout":
                self.assertFalse(secret.exists())
                self.assertTrue((self.root / "archive.ready").exists())
                self.assertNotIn("INVENTED_SECRET", kwargs["env"])
                events.append("render")
            else:
                self.assertEqual(args, ["sh", str(self.root / "deadman.sh"), "now"])
                self.assertFalse(kwargs["check"])
                events.append("terminate")

        with mock.patch.dict(worker.os.environ, {"INVENTED_SECRET": "dummy"}), (
            mock.patch.object(worker.subprocess, "run", side_effect=run)
        ), mock.patch.object(worker, "publish", side_effect=lambda *_: events.append("publish")):
            self.assertEqual(worker.supervise(self.root), 0)
        self.assertEqual(events, ["render", "publish", "terminate"])
