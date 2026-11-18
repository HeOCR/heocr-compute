"""The job's start command returns at once, though the job runs on: run for real with a local shell."""
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from heocr_compute.remote import runner


class LocalShell:
    """Runs the pod-side command locally, rewriting /heocr to a temporary folder. Output is captured
    through pipes, which, like sshd's channel, stay open while any process holds them. A stand-in setsid
    (macOS has none) runs its command in place, as Linux's does when it need not fork."""

    def __init__(self, root: Path):
        self.root = str(root)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "setsid").write_text('#!/bin/sh\nexec "$@"\n')
        (bin_dir / "setsid").chmod(0o755)
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}

    def run(self, command, input=None, timeout=600, check=True):
        cmd = command.replace("/heocr", self.root)
        t0 = time.monotonic()
        p = subprocess.run(["bash", "-c", cmd], input=input, capture_output=True, text=True, timeout=timeout,
                           env=self.env)
        self.elapsed = time.monotonic() - t0
        return p


class DetachedStartTest(unittest.TestCase):
    def test_start_returns_while_the_job_runs(self):
        with tempfile.TemporaryDirectory() as d:
            shell = LocalShell(Path(d))
            r = runner.Runner(api=None, ledger=None, out=lambda *a: None, sleep=lambda s: None,
                              clock=iter([0.0] + [1.0] * 10 + [10**9] * 10).__next__, poll_seconds=0)
            spec = mock.Mock(cmd="sleep 20", hours=1)
            t0 = time.monotonic()
            status = r._run_job(shell, spec, deadline=10**8)
            self.assertLess(time.monotonic() - t0, 10, "the start call waited for the job")
            self.assertTrue((Path(d) / "job.started").exists())
            self.assertEqual(status, "timeout")  # the fake clock ran out while the 20 s job was still running


if __name__ == "__main__":
    unittest.main()
