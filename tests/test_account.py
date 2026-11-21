"""The key source, the account check and per-account ledgers. All addresses and keys are invented."""
import builtins
import datetime as dt
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import heocr_compute
from heocr_compute.remote import api as api_mod
from heocr_compute.remote.api import RunPod, RunPodError, load_account, load_key
from heocr_compute.remote.ledger import Ledger


class KeySourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        other = self.home / ".runpod"
        other.mkdir()
        (other / "config.toml").write_text('apikey = "invented-other-project-key"\n')
        self.env = {"HOME": str(self.home), "RUNPOD_API_KEY": "invented-env-key"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_other_tools_config_and_env_are_ignored(self):
        with mock.patch.dict(os.environ, self.env, clear=False):
            with self.assertRaises(RunPodError):
                load_key()

    def test_heocr_key_file(self):
        cfg = self.home / ".config" / "heocr"
        cfg.mkdir(parents=True)
        (cfg / "runpod.toml").write_text('apikey = "invented-heocr-key"\n')
        with mock.patch.dict(os.environ, self.env, clear=False):
            self.assertEqual(load_key(), "invented-heocr-key")
        alt = self.home / "elsewhere.toml"
        alt.write_text('apikey = "invented-alt-key"\n')
        self.assertEqual(load_key({"HEOCR_RUNPOD_CONFIG": str(alt)}), "invented-alt-key")

    def test_old_path_is_never_read(self):
        cfg = self.home / ".config" / "heocr"
        cfg.mkdir(parents=True)
        (cfg / "runpod.toml").write_text('apikey = "invented-heocr-key"\n')
        (cfg / "remote_account.toml").write_text('email = "Runner@Example.org"\n')
        seen = []
        real_open, real_read = builtins.open, Path.read_text

        def spy_open(f, *a, **k):
            seen.append(str(f))
            return real_open(f, *a, **k)

        def spy_read(self, *a, **k):
            seen.append(str(self))
            return real_read(self, *a, **k)

        with mock.patch.dict(os.environ, self.env, clear=False), \
                mock.patch.object(builtins, "open", spy_open), mock.patch.object(Path, "read_text", spy_read):
            load_key()
            load_account()
        self.assertTrue(seen)
        self.assertFalse([p for p in seen if ".runpod" in p], seen)

    def test_package_never_names_the_old_path(self):
        root = Path(heocr_compute.__file__).parent
        for f in root.rglob("*.py"):
            self.assertNotIn(".runpod/", f.read_text(encoding="utf-8"), f)


class AccountCheckTest(unittest.TestCase):
    def client(self, email):
        def transport(method, url, headers, body):
            return 200, json.dumps({"data": {"myself": {"email": email}}}).encode()
        return RunPod("invented-key", transport=transport, sleep=lambda s: None)

    def test_matching_account(self):
        expected = {"email": "runner@example.org", "label": "heocr"}
        self.assertEqual(self.client("Runner@Example.org").check_account(expected), "heocr")

    def test_other_account_refused_without_naming_it(self):
        expected = {"email": "runner@example.org", "label": "heocr"}
        with self.assertRaises(RunPodError) as cm:
            self.client("someone-else@example.net").check_account(expected)
        self.assertNotIn("someone-else", str(cm.exception))

    def test_account_file_required(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RunPodError):
                load_account({"HEOCR_REMOTE_ACCOUNT": str(Path(d) / "none.toml")})
            p = Path(d) / "account.toml"
            p.write_text('label = "heocr"\n')
            with self.assertRaises(RunPodError):
                load_account({"HEOCR_REMOTE_ACCOUNT": str(p)})
            p.write_text('email = "Runner@Example.org"\nlabel = "heocr"\n')
            self.assertEqual(load_account({"HEOCR_REMOTE_ACCOUNT": str(p)}),
                             {"email": "runner@example.org", "label": "heocr"})


class ErrorDetailTest(unittest.TestCase):
    def test_body_without_message_is_shown(self):
        def transport(method, url, headers, body):
            return 400, b'{"errors": [{"field": "cpu", "reason": "invented: no capacity"}]}'
        with self.assertRaises(RunPodError) as cm:
            RunPod("invented-key", transport=transport, sleep=lambda s: None).create_pod({})
        self.assertIn("invented: no capacity", str(cm.exception))
        self.assertNotIn("invented-key", str(cm.exception))


class PodStartupTelemetryTest(unittest.TestCase):
    def test_selects_runtime_and_desired_status_from_same_snapshot(self):
        for status, runtime, expected in (
            ("RUNNING", None, "pending"),
            ("RUNNING", {"uptimeInSeconds": 0}, "running"),
            ("EXITED", {"uptimeInSeconds": 1234}, "stopped"),
        ):
            with self.subTest(status=status, runtime=runtime):
                def transport(method, url, headers, body, *, timeout):
                    self.assertTrue(url.endswith("/graphql"))
                    self.assertEqual(method, "POST")
                    self.assertEqual(timeout, 8)
                    request = json.loads(body)
                    self.assertEqual(request["variables"], {"podId": "example"})
                    self.assertIn("runtime { uptimeInSeconds }", request["query"])
                    return 200, json.dumps({"data": {"pod": {
                        "id": "example", "desiredStatus": status, "runtime": runtime,
                    }}}).encode()
                client = RunPod("invented-key", transport=transport, sleep=lambda _: None)
                self.assertEqual(client.pod_startup_state("example"), expected)

    def test_incomplete_telemetry_is_an_error_and_is_redacted(self):
        missing = {"data": {"pod": {"id": "example", "desiredStatus": "RUNNING"}}}
        partial = {"data": {"pod": {"id": "example", "desiredStatus": "RUNNING", "runtime": None}},
                   "errors": [{"message": "invented sensitive diagnostic"}]}
        for reply in (missing, partial):
            def transport(*args, **kwargs):
                return 200, json.dumps(reply).encode()
            with self.assertRaisesRegex(RunPodError, "^could not verify pod startup telemetry$"):
                RunPod("invented-key", transport=transport).pod_startup_state("example")

    def test_missing_graphql_pod_needs_rest_confirmation(self):
        for rest_status in (200, 404):
            def transport(method, url, headers, body, *, timeout):
                if method == "POST":
                    return 200, b'{"data": {"pod": null}}'
                self.assertTrue(url.endswith("/v2/pods/example"))
                return rest_status, b'{"id": "example"}'
            client = RunPod("invented-key", transport=transport)
            if rest_status == 404:
                self.assertEqual(client.pod_startup_state("example"), "gone")
            else:
                with self.assertRaises(RunPodError):
                    client.pod_startup_state("example")


class PodListTest(unittest.TestCase):
    def test_admission_listing_follows_pages_and_includes_cluster_pods(self):
        replies = [
            {"pods": [{"id": "example1"}], "pagination": {"hasNextPage": True, "nextCursor": "example-page"}},
            {"pods": [{"id": "example2"}], "pagination": {"hasNextPage": False, "nextCursor": None}},
        ]
        transport = mock.Mock(side_effect=[(200, json.dumps(reply).encode()) for reply in replies])
        pods = RunPod("invented-key", transport=transport).list_pods(include_cluster_pods=True)
        self.assertEqual([pod["id"] for pod in pods], ["example1", "example2"])
        urls = [call.args[1] for call in transport.call_args_list]
        self.assertTrue(all("includeClusterPods=true" in url for url in urls))
        self.assertIn("cursor=example-page", urls[1])

    def test_incomplete_or_looping_pod_list_is_an_error(self):
        for reply in ({"pagination": {"hasNextPage": False}}, {"pods": []},
                      {"pods": [], "pagination": {"hasNextPage": True, "nextCursor": "repeat"}}):
            with self.subTest(reply=reply):
                transport = mock.Mock(return_value=(200, json.dumps(reply).encode()))
                with self.assertRaisesRegex(RunPodError, "could not verify .*pod list"):
                    RunPod("invented-key", transport=transport).list_pods()


class PerAccountLedgerTest(unittest.TestCase):
    def test_queue_display_ignores_expired_dead_and_other_account_waits(self):
        now = dt.datetime(2026, 11, 18, 15, 0, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "spend.jsonl"
            led = Ledger(path, now=lambda: now, account="heocr")
            led.queue("waiting", vcpus=8, wait_seconds=600)
            led.queue("expired", vcpus=2, wait_seconds=60)
            with mock.patch.object(os, "getpid", return_value=999999):
                led.queue("dead", vcpus=4, wait_seconds=600)
            Ledger(path, now=lambda: now, account="other").queue("other", vcpus=16, wait_seconds=600)
            now += dt.timedelta(minutes=2)
            with mock.patch("heocr_compute.remote.ledger.pid_alive", side_effect=lambda pid: pid != 999999):
                self.assertEqual([e["run_id"] for e in led.waiting()], ["waiting"])
            self.assertEqual(led.month_to_date()["total"], 0)

    def test_only_this_accounts_runs_count(self):
        now = dt.datetime(2026, 11, 18, 15, 0, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "spend.jsonl"
            old = Ledger(path, now=lambda: now)                  # untagged history
            old.reserve("r1", pod_type="cpu3cx2", usd_est=0.05)
            old.final("r1", pod_type="cpu3cx2", hours=0.04, usd=0.0031, status="ok")
            other = Ledger(path, now=lambda: now, account="other")
            other.reserve("r2", pod_type="cpu3cx2", usd_est=9.0)  # open, on another account
            led = Ledger(path, now=lambda: now, account="heocr")
            led.reserve("r3", pod_type="cpu3cx2", usd_est=0.05)
            led.final("r3", pod_type="cpu3cx2", hours=0.04, usd=0.0031, status="ok")
            m = led.month_to_date()
            self.assertEqual([r["run_id"] for r in m["finals"]], ["r3"])
            self.assertAlmostEqual(m["total"], 0.0031)
            self.assertEqual(sorted(r["run_id"] for r in m["others"]), ["r1", "r2"])
            self.assertAlmostEqual(m["other_spent"], 0.0031)
            self.assertTrue(all(e.get("account") == "heocr" for e in led.events() if e["run_id"] == "r3"))


if __name__ == "__main__":
    unittest.main()
