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


class PerAccountLedgerTest(unittest.TestCase):
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
