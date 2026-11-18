"""The upload policy, on invented folders only."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from heocr_compute.remote import manifest
from heocr_compute.remote.policy import Policy, Refused


def make_policy(base: Path, roots: str, deny: str) -> Policy:
    (base / "open_roots.toml").write_text(roots)
    (base / "denylist.txt").write_text(deny)
    return Policy.load(str(base / "open_roots.toml"), str(base / "denylist.txt"))


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = d = Path(self.tmp.name).resolve()
        for sub in ("open/synth/lists", "open/text", "open/scans/run-a", "elsewhere", "secret-set",
                    "open/synth/reference-texts", "hashed"):
            (d / sub).mkdir(parents=True)
        (d / "open/synth/lists/train.txt").write_text("invented training line\n")
        (d / "open/synth/reference-texts/page.txt").write_text("invented reference line\n")
        (d / "open/text/corpus.jsonl").write_text('{"text": "invented"}\n')
        (d / "elsewhere/notes.txt").write_text("invented note\n")
        (d / "secret-set/doc.txt").write_text("invented stand-in for restricted material\n")
        self.restricted_body = b"invented stand-in for a hash-named restricted file\n"
        sha = hashlib.sha256(self.restricted_body).hexdigest()
        (d / "hashed" / f"{sha}.pdf").write_bytes(self.restricted_body)
        roots = (f'[roots]\nsynthetic = ["{d}/open/synth"]\nhtext = "{d}/open/text"\n'
                 f'pd_scans = ["{d}/open/scans"]\n')
        deny = (f"# invented rules\n{d}/secret-set\n*/reference-texts/*\n*/runs/*test*/*\n"
                f"sha256-names: {d}/hashed\n")
        self.policy = make_policy(d, roots, deny)

    def tearDown(self):
        self.tmp.cleanup()

    def test_open_roots_allowed(self):
        self.assertEqual(self.policy.check_root(self.d / "open/synth"), "synthetic")
        self.assertEqual(self.policy.check_root(self.d / "open/synth/lists"), "synthetic")
        self.assertEqual(self.policy.check_root(self.d / "open/text"), "htext")
        entries = manifest.build(self.d / "open/synth/lists", self.policy)
        self.assertEqual([e.path for e in entries], ["train.txt"])

    def test_everything_else_refused_by_default(self):
        for p in (self.d / "elsewhere", self.d, self.d / "open", Path("/")):
            with self.subTest(p=p), self.assertRaises(Refused):
                self.policy.check_root(p)

    def test_local_denylist_path_and_glob(self):
        with self.assertRaises(Refused):
            self.policy.check_root(self.d / "secret-set")
        self.assertIsNotNone(self.policy.deny_reason(self.d / "open/scans/runs/r2_test/pred.json"))
        self.assertIsNotNone(self.policy.deny_reason(self.d / "open/scans/RUNS/Heldout-TEST/x.json"))
        self.assertIsNone(self.policy.deny_reason(self.d / "open/scans/run-a/page.png"))
        # a denied folder inside an open root stops the manifest
        with self.assertRaises(Refused):
            manifest.build(self.d / "open/synth", self.policy)
        entries = manifest.build(self.d / "open/synth", self.policy, excludes=("reference-texts/*",))
        self.assertEqual([e.path for e in entries], ["lists/train.txt"])

    def test_denied_content_under_another_name(self):
        (self.d / "open/scans/run-a/harmless-name.pdf").write_bytes(self.restricted_body)
        with self.assertRaises(Refused):
            manifest.build(self.d / "open/scans", self.policy)

    def test_symlink_out_of_an_open_root(self):
        link = self.d / "open/text/notes.txt"
        link.symlink_to(self.d / "elsewhere/notes.txt")
        with self.assertRaises(Refused):
            manifest.build(self.d / "open/text", self.policy)

    def test_symlink_into_denied_folder(self):
        (self.d / "open/text/doc.txt").symlink_to(self.d / "secret-set/doc.txt")
        with self.assertRaises(Refused):
            manifest.build(self.d / "open/text", self.policy)

    def test_verify_local_rechecks_policy(self):
        entries = manifest.build(self.d / "open/text", self.policy)
        stricter = make_policy(self.d, f'[roots]\nsynthetic = ["{self.d}/open/synth"]\n', "*/text/*\n")
        with self.assertRaises(Refused):
            manifest.verify_local(entries, self.d / "open/text", stricter)

    def test_missing_or_empty_files_refuse_everything(self):
        with self.assertRaises(Refused):
            Policy.load(str(self.d / "none.toml"), str(self.d / "denylist.txt"))
        with self.assertRaises(Refused):
            Policy.load(str(self.d / "open_roots.toml"), str(self.d / "none.txt"))
        with self.assertRaises(Refused):
            make_policy(self.d, f'[roots]\nhtext = "{self.d}/open/text"\n', "# only a comment\n")
        with self.assertRaises(Refused):
            make_policy(self.d, "[roots]\n", "*/x/*\n")

    def test_unknown_category_refused(self):
        with self.assertRaises(Refused):
            make_policy(self.d, f'[roots]\nanything = "{self.d}/elsewhere"\n', "*/x/*\n")


if __name__ == "__main__":
    unittest.main()
