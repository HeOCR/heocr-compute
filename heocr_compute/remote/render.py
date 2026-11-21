"""Plan the bounded render and publish its verified bundle from the runner host."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path

from . import budget, manifest, profiles, render_job
from .api import load_hf_token
from .policy import Policy
from .runner import DEFAULT_CAPACITY_WAIT_HOURS, RunFailed, Runner, Spec

HF = "https://huggingface.co"
DATASET = "HeOCR/htext-pby"
REVISION = "fd5eba052cb5957f3a2c14f6613fb54f6176d7c2"  # pby-v3
FILE = "data/pby_lines_il_us_plain_v3.txt"
SOURCE = {"repo": DATASET, "revision": REVISION, "path": FILE, "license": "CC0-1.0",
          "sha256": "9ab46e476b3ec9d826334458ee5267444f90ff5585fbe03eaa4551f397a2e2de",
          "size": 17249017, "url": f"{HF}/datasets/{DATASET}/resolve/{REVISION}/{FILE}"}
CODE = "6632c6cfc7a33e93fe89ac9de8fc8bc181019a4f"
FONT_SHA = "55fb21128445b95ec6770204d5b84b88aaa52db48d4cbfd6a716a5becd6fca6f"
DEFAULT_FONT = "noto-sans-hebrew"
MAX_HOURS = 4.0
MAX_COST = 1.0
MAX_COUNT = 20_000
COMMAND = ("apt-get -qq update && apt-get -qq install -y --no-install-recommends "
           "ca-certificates git python3 python3-venv && python3 in/render_job.py in/job.json")

COMMIT = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
DESTINATION = re.compile(r"HeOCR/[A-Za-z0-9][A-Za-z0-9_-]{0,79}")


def settings(a) -> dict:
    if not 1 <= a.count <= MAX_COUNT or not 0 <= a.seed < 2**64:
        raise ValueError(f"render needs count 1–{MAX_COUNT} and a nonnegative 64-bit seed")
    if not math.isfinite(a.hours) or not 0.25 <= a.hours <= MAX_HOURS:
        raise ValueError(f"render --hours must be between 0.25 and {MAX_HOURS:g}")
    if not math.isfinite(a.run_cap) or not 0 < a.run_cap <= MAX_COST:
        raise ValueError(f"render --run-cap must be positive and at most ${MAX_COST:.2f}")
    if not math.isfinite(a.month_cap) or not 0 < a.month_cap <= budget.MONTH_CAP_USD:
        raise ValueError("render --month-cap must be positive and at most the runner monthly cap")
    if not COMMIT.fullmatch(a.code_revision) or not DIGEST.fullmatch(a.font_manifest_sha256):
        raise ValueError("render requires a full Git commit and font manifest SHA-256")
    fonts = a.font or [DEFAULT_FONT]
    if any(not IDENTIFIER.fullmatch(font) for font in fonts) or len(set(fonts)) != len(fonts):
        raise ValueError("render font IDs must be distinct identifiers")
    if a.mode is not None and not IDENTIFIER.fullmatch(a.mode):
        raise ValueError("render --mode must be an identifier")
    if a.mode is not None and a.code_revision == CODE:
        raise ValueError("the default renderer has no --mode option; pin the commit that implements it")
    if not DESTINATION.fullmatch(a.dataset):
        raise ValueError("render output must name an existing HeOCR dataset")
    return {"code_revision": a.code_revision, "font_manifest_sha256": a.font_manifest_sha256,
            "source": SOURCE, "fonts": fonts, "mode": a.mode, "count": a.count, "seed": a.seed}


def _read_public(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read(1 << 20)
    except OSError:
        raise RunFailed("could not verify a pinned public render source") from None


def check_sources(job: dict, policy: Policy) -> None:
    source = job["source"]
    policy.check_hash(source["path"], source["sha256"])
    policy.check_hash("font manifest", job["font_manifest_sha256"])
    card = json.loads(_read_public(f"{HF}/api/datasets/{DATASET}/revision/{REVISION}"))
    listing = json.loads(_read_public(f"{HF}/api/datasets/{DATASET}/tree/{REVISION}/data"))
    if card.get("private") is not False or card.get("gated") is not False or card.get("sha") != REVISION:
        raise RunFailed("pinned corpus is not an unrestricted public revision")
    if (card.get("cardData") or {}).get("license") != "cc0-1.0" or not any(
        f.get("path") == FILE and f.get("size") == source["size"]
        and (f.get("lfs") or {}).get("oid") == source["sha256"] for f in listing
    ):
        raise RunFailed("pinned corpus licence, size or SHA-256 did not match")
    base = f"https://raw.githubusercontent.com/HeOCR/hocrsyngen/{job['code_revision']}"
    fonts = _read_public(f"{base}/{render_job.FONT_MANIFEST}")
    if hashlib.sha256(fonts).hexdigest() != job["font_manifest_sha256"]:
        raise RunFailed("pinned renderer font manifest did not match its SHA-256")


class HubPublisher:
    """Use the local runner token only against the canonical Hub endpoint."""

    def __init__(self):
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
        except ImportError:
            raise RunFailed("install the render extra: pip install '.[render]'") from None
        self.api = HfApi(endpoint=HF, token=load_hf_token())
        self.add = CommitOperationAdd

    def check(self, dataset: str, *, revision: str | None = None) -> str:
        if not DESTINATION.fullmatch(dataset):
            raise RunFailed("render output must name an existing HeOCR dataset")
        try:
            info = self.api.dataset_info(dataset, revision=revision, timeout=30)
        except Exception:
            raise RunFailed("could not verify the private HF destination with the runner credential") from None
        if info.private is not True or info.id != dataset or not info.sha or not COMMIT.fullmatch(info.sha):
            raise RunFailed("HF destination did not resolve to the named private HeOCR dataset")
        return info.sha

    def preflight(self, dataset: str) -> None:
        self.check(dataset)
        try:
            identity = self.api.whoami()
        except Exception:
            raise RunFailed("could not verify the runner HF credential") from None
        if identity.get("auth", {}).get("accessToken", {}).get("role") not in ("write", "fineGrained"):
            raise RunFailed("HF publishing requires a write-capable runner token")

    def publish(self, dataset: str, run_id: str, root: Path, job: dict) -> str:
        archive = root / render_job.ARCHIVE
        sidecar = root / render_job.ARTIFACT_MANIFEST
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        if (info.get("schema") != render_job.ARTIFACT_SCHEMA or info.get("job") != job
                or info.get("archive") != {"path": archive.name, "size": archive.stat().st_size,
                                           "sha256": manifest.sha256_file(archive)}):
            raise RunFailed("verified render bundle does not match its artifact manifest")
        if archive.stat().st_size > render_job.MAX_ARCHIVE:
            raise RunFailed("render bundle exceeds the archive size limit")
        parent = self.check(dataset)
        prefix = f"renders/{run_id}"
        try:
            paths = {f"{prefix}/{f.name}": f for f in (archive, sidecar)}
            result = self.api.create_commit(dataset, repo_type="dataset", parent_commit=parent,
                                            commit_message=f"Add render {run_id}",
                                            operations=[self.add(path_in_repo=k, path_or_fileobj=v)
                                                        for k, v in paths.items()])
            revision = result.oid
            uploaded = self.api.get_paths_info(dataset, list(paths), repo_type="dataset", revision=revision)
            if {entry.path for entry in uploaded} != set(paths):
                raise ValueError("missing uploaded artifact")
            for entry in uploaded:
                local = paths[entry.path]
                if entry.size != local.stat().st_size:
                    raise ValueError("uploaded artifact size mismatch")
                if entry.lfs is not None:
                    if entry.lfs.sha256 != manifest.sha256_file(local):
                        raise ValueError("uploaded artifact SHA-256 mismatch")
                else:
                    # Regular Hub files are Git blobs. Include the Git size header in the digest.
                    git_hash = hashlib.sha1(f"blob {entry.size}\0".encode(), usedforsecurity=False)
                    with local.open("rb") as f:
                        while chunk := f.read(1 << 20):
                            git_hash.update(chunk)
                    if entry.blob_id != git_hash.hexdigest():
                        raise ValueError("uploaded manifest Git blob mismatch")
        except Exception:
            raise RunFailed("HF commit or readback failed; keep the verified local bundle") from None
        if self.check(dataset, revision=revision) != revision:
            raise RunFailed("could not verify the private HF commit; keep the local bundle")
        return f"{HF}/datasets/{dataset}/tree/{revision}/{prefix}"


def launch(a, connect) -> dict:
    job = settings(a)
    policy = Policy.load()
    check_sources(job, policy)
    publisher = HubPublisher()
    publisher.preflight(a.dataset)
    profile = profiles.PROFILES["cpu"].sized(vcpus=4)
    result_base = Path(a.results or "~/heocr-remote/results").expanduser()
    existing = next(p for p in (result_base, *result_base.parents) if p.exists())
    if not existing.is_dir() or shutil.disk_usage(existing).free < 3 << 30:
        raise RunFailed("render needs at least 3 GiB free at the local results destination")
    source = job["source"]
    print(f"render: {job['count']} lines, seed {job['seed']}, mode {job['mode'] or '(renderer default)'}")
    print(f"renderer: {job['code_revision']}, fonts: {', '.join(job['fonts'])}")
    print(f"font manifest sha256: {job['font_manifest_sha256']}")
    print(f"public input: {source['repo']}@{source['revision']}/{source['path']} ({source['license']})")
    print(f"input sha256: {source['sha256']}")
    print("private HeOCR destination and HF credential checked (write permission checked at commit)")
    print(f"render wall limit: {a.hours:g} h including provisioning and bundle pull")
    print("output: render.tar and artifact-manifest.json under renders/<run-id>/ after teardown")
    with tempfile.TemporaryDirectory(prefix="heocr-render-", dir=result_base if result_base.is_dir() else existing) as temp:
        root = Path(temp) / "in"
        root.mkdir()
        (root / "render_job.py").write_bytes(Path(render_job.__file__).read_bytes())
        (root / "job.json").write_text(json.dumps(job) + "\n", encoding="utf-8")
        # Only runner-generated recipe files are added, like the smoke recipe.
        policy = policy.with_root("synthetic", root)
        listing = Path(temp) / "inputs.json"
        manifest.dump(manifest.build(root, policy), listing)
        spec = Spec(profile, listing, root, COMMAND, a.hours, run_cap=a.run_cap,
                    month_cap=a.month_cap, results=result_base, dry_run=a.dry_run, yes=a.yes,
                    policy=policy, render_bundle=True,
                    capacity_wait_hours=getattr(a, "capacity_wait_hours", DEFAULT_CAPACITY_WAIT_HOURS))
        result = Runner(*connect()).run(spec)
    if result["status"] == "ok":
        dest = result_base / result["run_id"]
        url = publisher.publish(a.dataset, result["run_id"], dest, job)
        print(f"verified private HF commit: {url}")
        (dest / "hf-upload.json").write_text(json.dumps({"commit_url": url}) + "\n", encoding="utf-8")
        (dest / render_job.ARCHIVE).unlink()
    return result
