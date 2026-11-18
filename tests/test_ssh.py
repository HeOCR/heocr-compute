import shlex
import tempfile
import unittest
from pathlib import Path

from heocr_compute.remote.ssh import Remote


class LocalRemote(Remote):
    """Runs the pod-side command locally, so the tar streams are exercised for real."""

    def argv(self, command):
        return ["sh", "-c", command]


class TarStreamTest(unittest.TestCase):
    def test_upload_and_download_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "src"
            (src / "lists").mkdir(parents=True)
            (src / "lists" / "train.txt").write_text("invented line one\ninvented line two\n")
            (src / "job.sh").write_text("echo invented\n")
            pod = d / "pod"
            r = LocalRemote("unused", 22)
            r.upload(src, ["lists/train.txt", "job.sh"], str(pod / "in"))
            self.assertEqual((pod / "in" / "lists" / "train.txt").read_text(), "invented line one\ninvented line two\n")
            out = pod / "out"
            (out / "best").mkdir(parents=True)
            (out / "best" / "heb.traineddata").write_bytes(b"\x00\x01" * 100)
            (out / "other.log").write_text("not asked for\n")
            got = r.download(str(out), ["best/heb.traineddata"], d / "results")
            self.assertEqual(got, ["best/heb.traineddata"])
            self.assertEqual((d / "results" / "best" / "heb.traineddata").read_bytes(), b"\x00\x01" * 100)
            self.assertFalse((d / "results" / "other.log").exists())

    def test_command_runs(self):
        p = LocalRemote("unused", 22).run("echo " + shlex.quote("hello"))
        self.assertEqual(p.stdout.strip(), "hello")


if __name__ == "__main__":
    unittest.main()
