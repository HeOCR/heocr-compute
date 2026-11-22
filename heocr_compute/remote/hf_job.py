"""The autonomous training job contract and canonical private Hub access."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import manifest

SCHEMA = "heocr_hf_training.v1"
RECEIPT = "heocr_hf_training_receipt.v1"
TELEMETRY_SCHEMA = "heocr_hf_training.v2"
TELEMETRY_RECEIPT = "heocr_hf_training_receipt.v2"
MAX_SPEC = 24 * 1024
HUB = "https://huggingface.co"
SHA = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")
REPO = re.compile(r"HeOCR/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
RUN_ID = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{4}")
POD_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
IMAGE = re.compile(r"[\w./:-]+@sha256:[0-9a-f]{64}", re.ASCII)
SECRET = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
ARTIFACT_SUFFIXES = (".traineddata", ".checkpoint", "_checkpoint", ".ckpt", ".safetensors", ".pt", ".pth")


class PublishError(ValueError):
    """Publication could not be confirmed. Keep the latest visible receipt intact."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matches(pattern, value):
    return isinstance(value, str) and bool(pattern.fullmatch(value))


def fields(value, expected):
    require(isinstance(value, dict) and value.keys() == set(expected.split()), "unexpected job or receipt fields")


def safe_path(value):
    require(isinstance(value, str) and len(value) <= 200
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*", value)
            and all(not part.startswith(".") for part in value.split("/"))
            and manifest.safe_rel(value) == value, "invalid relative path")
    return value


def location(record):
    require(matches(REPO, record["repo"]) and record["type"] in ("dataset", "model"),
            "expected a HeOCR dataset or model repository")


def size_ok(value):
    return type(value) is int and value > 0


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checkpoint_groups(job):
    """Map each checkpoint to the files committed with it."""
    telemetry = job.get("telemetry")
    for spec in job["snapshots"]:
        files = {spec["name"]: spec["max_bytes"]}
        if telemetry is not None:
            stem = Path(spec["name"]).stem
            files[f"{stem}.training-metrics.json"] = telemetry["metrics_max_bytes"]
            files[f"{stem}.trainer.log"] = telemetry["trainer_log_max_bytes"]
        yield files


def final_logs(job):
    telemetry = job.get("telemetry")
    if telemetry is None:
        return {}
    return {"training.log": telemetry["run_log_max_bytes"],
            "setup.log": telemetry["setup_log_max_bytes"],
            "partial.trainer.log": telemetry["trainer_log_max_bytes"]}


def output_limits(job):
    outputs = final_logs(job)
    for group in checkpoint_groups(job):
        outputs.update(group)
    return outputs


def parse(raw, approved):
    require(len(raw) <= MAX_SPEC and matches(SHA, approved) and digest(raw) == approved,
            "job exceeds size limit or differs from the approved SHA-256")
    require(not re.search(rb"\bhf_[a-zA-Z0-9]{20,}\b", raw), "job must not contain a Hub token")
    job = json.loads(raw.decode("utf-8"))
    require(isinstance(job, dict) and job.get("schema") in (SCHEMA, TELEMETRY_SCHEMA),
            "expected an autonomous training job schema")
    telemetry_enabled = job["schema"] == TELEMETRY_SCHEMA
    fields(job, "schema clean_sha256 image inputs entrypoint args snapshots output"
           + (" telemetry" if telemetry_enabled else ""))
    require(matches(SHA, job["clean_sha256"]),
            "expected the training schema and exact CLEAN receipt SHA-256")
    if telemetry_enabled:
        fields(job["telemetry"], "metrics_max_bytes trainer_log_max_bytes run_log_max_bytes setup_log_max_bytes")
        require(all(size_ok(limit) for limit in job["telemetry"].values()), "invalid telemetry size limit")
    require(matches(IMAGE, job["image"]), "expected a digest-pinned worker image")
    fields(job["output"], "repo type")
    location(job["output"])
    require(isinstance(job["inputs"], list) and job["inputs"], "job has no pinned inputs")
    targets = set()
    for item in job["inputs"]:
        fields(item, "repo type revision path target size sha256")
        location(item)
        safe_path(item["path"])
        target = safe_path(item["target"])
        require(matches(REVISION, item["revision"]) and matches(SHA, item["sha256"])
                and size_ok(item["size"]), "input requires a full revision, byte size and SHA-256")
        require(target not in targets and all(not target.startswith(t + "/") and not t.startswith(target + "/")
                                              for t in targets), "input destinations overlap")
        targets.add(target)
    entrypoint = safe_path(job["entrypoint"])
    require(entrypoint.endswith(".sh") and entrypoint in targets, "entrypoint must be a pinned shell input")
    require(isinstance(job["args"], list) and all(isinstance(a, str) and a.isprintable()
                                                for a in job["args"]), "invalid entrypoint arguments")
    require(isinstance(job["snapshots"], list) and job["snapshots"], "declare at least one snapshot")
    seen = set()
    for spec in job["snapshots"]:
        fields(spec, "name max_bytes")
        name = safe_path(spec["name"])
        require("/" not in name and name not in seen and name.endswith(ARTIFACT_SUFFIXES)
                and size_ok(spec["max_bytes"]), "invalid or duplicate snapshot")
        seen.add(name)
    if telemetry_enabled:
        names = [*final_logs(job), *(name for group in checkpoint_groups(job) for name in group)]
        require(len(names) == len(set(names)), "telemetry output names collide")
        require(len(names) <= 128, "too many files for a bounded run receipt")
        for name in names:
            safe_path(name)
    return job


def check_repo_scope(job, repo):
    """Require every Hub operation in a job to use one operator-approved dataset."""
    require(matches(REPO, repo), "training worker requires one approved HeOCR dataset")
    require(all(record["repo"] == repo and record["type"] == "dataset"
                for record in (*job["inputs"], job["output"])),
            "job is outside its approved Hub dataset scope")


def verify_file(path, size, sha256):
    require(path.is_file() and path.stat().st_size == size
            and manifest.sha256_file(path) == sha256, "artifact size or SHA-256 mismatch")


class Hub:
    def __init__(self, token, cache):
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

        self.api = HfApi(endpoint=HUB, token=token)
        self.add = CommitOperationAdd
        self.get = hf_hub_download
        self.token = token
        self.cache = cache

    def private_revision(self, source, revision=None):
        try:
            info = self.api.repo_info(source["repo"], repo_type=source["type"], revision=revision, timeout=30)
        except Exception:
            raise ValueError("could not verify the named private Hub repository") from None
        require(info.private is True and info.id == source["repo"] and matches(REVISION, info.sha)
                and (revision is None or info.sha == revision), "Hub repository is not private or revision mismatched")
        return info.sha

    def download(self, source, name, revision, max_size):
        try:
            entries = self.api.get_paths_info(source["repo"], [name], repo_type=source["type"], revision=revision)
            expected = [e.size for e in entries if e.path == name and hasattr(e, "size")]
            require(len(expected) == 1 and 0 < expected[0] <= max_size, "missing or oversized Hub file")
            return Path(self.get(source["repo"], name, repo_type=source["type"], revision=revision,
                                 token=self.token, endpoint=HUB, cache_dir=self.cache))
        except Exception:
            raise ValueError("could not download a bounded file from the pinned Hub revision") from None

    def commit(self, output, prefix, receipt, artifacts=()):
        # Optimistic parent pins protect concurrent runs writing to the same private repository.
        for _ in range(3):
            try:
                parent = self.private_revision(output)
            except ValueError:
                break
            ops = [self.add(path_in_repo=f"{prefix}/receipt.json",
                            path_or_fileobj=json.dumps(receipt, sort_keys=True).encode("utf-8"))]
            ops.extend(self.add(path_in_repo=f"{prefix}/{path.name}", path_or_fileobj=path) for path in artifacts)
            try:
                result = self.api.create_commit(output["repo"], repo_type=output["type"], parent_commit=parent,
                                                commit_message=f"Training snapshots for {receipt['run_id']}",
                                                operations=ops)
                require(matches(REVISION, result.oid), "Hub did not confirm an immutable snapshot revision")
                return
            except KeyboardInterrupt:
                raise PublishError("private Hub commit was interrupted during publication") from None
            except Exception as exc:
                if getattr(getattr(exc, "response", None), "status_code", None) not in (409, 412):
                    break
        raise PublishError("private Hub snapshot commit could not be confirmed") from None
