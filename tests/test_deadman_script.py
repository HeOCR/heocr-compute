"""The pod-side dead-man's switch script, run for real with a stand-in runpodctl."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from heocr_compute.remote.runner import DEADMAN_SH


class DeadmanScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.script = d / "deadman.sh"
        self.script.write_text(DEADMAN_SH)
        self.bin = d / "bin"
        self.bin.mkdir()
        self.calls = d / "calls.txt"
        fake = self.bin / "runpodctl"
        fake.write_text(f'#!/bin/sh\necho "$@" >> {self.calls}\n')
        fake.chmod(0o755)
        self.log = d / "deadman.log"

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, arg, env):
        base = {"PATH": f"{self.bin}:/usr/bin:/bin", "HEOCR_DEADMAN_LOG": str(self.log)}
        p = subprocess.run(["sh", str(self.script), arg], env={**base, **env}, capture_output=True, text=True,
                           timeout=30)
        return p.stdout.strip()

    def test_reports_what_is_missing_without_values(self):
        out = self.run_script("60", {"RUNPOD_POD_ID": "invented-pod"})
        self.assertIn("unavailable", out)
        self.assertIn("pod-key", out)
        self.assertNotIn("invented-pod", out)
        self.assertFalse(self.calls.exists())

    def test_fires_now(self):
        out = self.run_script("now", {"RUNPOD_POD_ID": "invented-pod", "RUNPOD_API_KEY": "invented-key"})
        self.assertEqual(out, "fired")
        self.assertEqual(self.calls.read_text().split(), ["pod", "delete", "invented-pod"])
        self.assertNotIn("invented-key", out)

    def test_arms_in_background(self):
        out = self.run_script("1", {"RUNPOD_POD_ID": "invented-pod", "RUNPOD_API_KEY": "invented-key"})
        self.assertEqual(out, "armed")
        for _ in range(50):
            if self.calls.exists():
                break
            subprocess.run(["sleep", "0.1"])
        self.assertEqual(self.calls.read_text().split(), ["pod", "delete", "invented-pod"])


if __name__ == "__main__":
    unittest.main()
