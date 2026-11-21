# Autonomous training from private Hub inputs

Use this path for explicitly approved private training runs. Runpod starts the supervisor from a digest-pinned worker image. The Mac sends a job specification of at most 24 KiB with the pod create request. The pod downloads the exact files directly from private Hugging Face repositories. The legacy open-data runner and its local input policy still serve existing workloads.

## Prepare the approved inputs

Use existing private HeOCR repositories. For each training data shard, replay file, starter checkpoint and job script, record its repository type and name, full 40-character revision, exact Hub path, intended relative input path, byte size and SHA-256. Pin the reviewed worker image by OCI digest. Stage any missing derivatives in the approved private repository before launch. Keep held-out material and unapproved inputs out of the specification.

The lead approves the SHA-256 of the exact UTF-8 specification after checking the input and CLEAN receipts, entrypoint, worker digest, output destination, time cap and spending quote. Input SHA-256 values and paths also pass the local denylist before launch. A compact example with placeholder pins follows.

```json
{
  "schema": "heocr_hf_training.v1",
  "clean_sha256": "<source CLEAN receipt SHA-256>",
  "image": "<reviewed worker image>@sha256:<64 hex digits>",
  "inputs": [
    {
      "repo": "HeOCR/example-private-input",
      "type": "dataset",
      "revision": "<40 hex digits>",
      "path": "jobs/train.sh",
      "target": "train.sh",
      "size": 128,
      "sha256": "<64 hex digits>"
    }
  ],
  "entrypoint": "train.sh",
  "args": [],
  "snapshots": [{"name": "iter-10000.traineddata", "max_bytes": 16000000}],
  "output": {"repo": "HeOCR/example-private-output", "type": "model"}
}
```

The operator configures a Runpod managed secret containing a fine-grained Hub token with read access to the named input repos and write access to the one private output repo. Use the secret's **name** for launch. Runpod injects the value into the pod environment. Keep token values out of job files and flags. The supervisor omits credentials from the child job environment. The training entrypoint is reviewed and runs as trusted code inside the dedicated pod. Keep the output repository private through collection. [Runpod managed secrets](https://docs.runpod.io/pods/templates/secrets)

## Worker and launch

Build the Linux AMD64 image with the repository Dockerfile. It defaults to a pinned Runpod base and installs the checksum-pinned official `runpodctl` binary. An approved replacement base can be passed by digest. The service supplies the pod-scoped credentials at startup. Use the resulting deployed digest in the job specification. For a trainer with additional dependencies, bake those into the reviewed worker before pinning it.

```sh
docker build --platform linux/amd64 -f docker/hf-train.Dockerfile -t "$WORKER_TAG" .
heocr-remote hf-train --profile cpu --job "$JOB" --approved-sha256 "$JOB_SHA" \
  --hf-secret "$RUNPOD_HF_SECRET_NAME" --hours "$HOURS" --run-cap "$USD_CAP" --dry-run
```

The dry run checks the contract, exact job SHA, disk declarations, local denylist and live quote. Remove `--dry-run` after the launch decision. The launcher records the reservation before create and returns the run ID and pod ID. The Secure Cloud pod exposes no SSH port. Keep the launch and any monitoring outside the local power scheduler. After creation, the Mac can sleep or disconnect while the pod works. An ambiguous create keeps the reservation open and must be reconciled before another launch. [Runpod pod create API](https://docs.runpod.io/api-reference-v2/pods/create-a-pod)

The pod arms its deadline guard before making Hub requests. It checks that the pinned source repositories are private, downloads only declared files, verifies byte sizes and SHA-256, and starts the pinned job under `/heocr` with inputs in `in/`. Write each expected snapshot to a temporary name and atomically rename it to the declared filename in `out/` once complete. Each declared snapshot is immutable after rename and has its own size limit. The supervisor commits snapshots as they become available, each alongside a checksum receipt under `runs/<run-id>/`. The final receipt records success only when the trainer exits cleanly and all snapshots are committed. Trainer logs and training data stay off the output repository.

The deadline uses the absolute launch cutoff. The container timeout stops the worker shortly before cutoff, leaving time for shutdown, and the guard independently attempts pod deletion at cutoff. The pod also attempts deletion after success or failure. A provisioning failure before the entrypoint starts or a Runpod deletion outage can require the account sweeper or console. Before approving a real-data launch, use a separately approved invented-input smoke run to verify managed-secret injection, snapshot publication and self-termination on the actual image and account.

## Collect and reconcile

Install the Hub client with `pip install '.[render]'` and configure the local runner Hub token for read access to the output repo. Run the collector after pod teardown.

```sh
heocr-remote collect-hf "$RUN_ID" --results "$PRIVATE_RESULTS"
```

Collection pins the current private output revision once, checks the run and job IDs against the ledger, and verifies every returned snapshot's size and SHA-256 before exposing the result directory. Earlier committed snapshots can be collected after an incomplete run. If billing is delayed, collect again to reconcile it. The original estimate remains reserved until provider billing is available. The regular sweeper preserves autonomous pods before their recorded deadline even if the launcher is gone. It also preserves autonomous pods belonging to another launcher's ledger, unless explicitly invoked with `--all`.
