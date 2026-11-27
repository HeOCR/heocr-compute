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

### Approved public Hub TRAIN archives

Use public TRAIN archives in v2 jobs after independent publication review. Add
`transport: "hf-public"` and `release_sha256` to each archive's Hub input record.
Both the source and target paths must end in `.tar`. Bind the exact release
receipt, immutable dataset revision, byte size and SHA-256 into the approved job.
Supply `--hf-public-train-repo HeOCR/example-open-data` for each public source
and `--hf-repo-scope` for the single private dataset that holds the entrypoint,
other private inputs and all outputs. The worker verifies public visibility and
downloads these archives anonymously. Use a worker image built with this support.

### Approved R2 TRAIN archives

A v2 job can declare an approved TRAIN archive from the private `heocr-train-private` bucket. Its input record has `transport: "r2"`, the 32-hex Cloudflare `account`, the exact object key in `path`, a relative archive `target`, and the expected `size` and `sha256`. Both paths end in `.tar`. Bind these fields into the job approval. Supply `--hf-repo-scope` for the private dataset that holds the entrypoint, other Hub inputs and all outputs.

Generate an object-scoped [presigned GET](https://developers.cloudflare.com/r2/api/s3/presigned-urls/) from local credentials with a lifetime of at most 24 hours. Pass `HEOCR_R2_URLS` as an in-memory secret environment variable containing a JSON mapping from each R2 target to its URL. Keep it out of job specs, command arguments, shell traces and ledgers. Refresh a URL if it expires while a job is queued. The supervisor consumes the variable before starting the trainer. It refuses redirects and checks byte size and SHA-256 before making the archive visible to the trainer.

A v2 job can also stage an approved Python trainer in that bucket. Add
`"kind":"trainer"` to the R2 input record, use a `.py` object key, and set
`"target":"train.py"`. The file can be at most 1 MiB, with the same exact
size, SHA-256 and presigned GET checks as other R2 inputs. Review the trainer
as executable code when approving its job SHA. The shell entrypoint remains
a pinned private Hub input, and explicitly invokes the trainer if needed.

### Private R2 output fallback

For a v2 job, set `output` to
`{"transport":"r2","account":"<32 hex digits>"}` and approve the resulting
job SHA-256. Use a worker image built with this support and add
`--hf-repo-scope` and `--require-worker-receipt`. The approved inputs and
private Hub read credential still work the same way. The worker writes run
receipts, declared checkpoints, metrics and logs to private R2.

Install `heocr-compute[r2]` on the launching and collecting machine. For a
dedicated output bucket, add `"bucket":"<bucket-name>"` to the approved
`output` object. Configure an R2 Object Read & Write credential scoped to
that single bucket. In the operator's mode-600 `r2-output.env` credential
file, set `R2_ACCOUNT_ID`, `R2_BUCKET`, `R2_ACCESS_KEY_ID` and
`R2_SECRET_ACCESS_KEY`. The account and bucket must both match the job and
are carried through startup verification and collection. Verify the
bucket's public access settings and test an authenticated write and
size/SHA-256 readback before using it. Legacy output objects without a
bucket use the existing Cloudflare credential file.

After capacity admission, the signer creates an exact PUT for each declared
object. The URLs expire within 24 hours and go only into the pod supervisor's
secret environment. The local credential stays on the operator's machine.
The dry run checks signing and job admission without creating an object or pod.
[Cloudflare R2 presigned URLs](https://developers.cloudflare.com/r2/api/s3/presigned-urls/)

Checkpoints and logs are uploaded under the unique run ID. PUTs are create-only.
The supervisor writes a new receipt only after uploading its artifact files.
Startup, checkpoint progress and terminal receipts have separate keys. The
launcher checks the startup receipt against its reservation. After teardown,
`heocr-remote collect-hf RUN_ID` checks the terminal receipt, or recovers the
last complete checkpoint if the terminal receipt is absent. It verifies
each file's size and SHA-256 before exposing local results. A PUT that returns
an ambiguous result does not count as a confirmed upload. Keep the raw
snapshots available through model selection. After independent readback,
retain the selected checkpoints, final artifacts and their manifest. Prune
only the other exact run-owned object keys after the selection is recorded.
[Cloudflare R2 conditional writes](https://developers.cloudflare.com/r2/api/s3/conditional-operations/)

## Tesseract native writer

For a v2 job whose bootstrap installs Tesseract into `/heocr/work/tess`, set
`"tesseract_writer_parent":"parent.traineddata"` to use that already pinned Hub
input to initialize `tesseract … lstm.train`. The worker creates the
`eng.traineddata` alias before the bootstrap installs the writer and its configs.
Source builds install the configs but do not install a default language model.
The alias supplies the writer's initialization without changing the boxes,
labels, or the model the trainer explicitly loads. Use a worker image built with
this option and replay a sample with the same initialization before launch.

## Worker and launch

Build the Linux AMD64 image with the repository Dockerfile. It defaults to a pinned Runpod base and installs the checksum-pinned official `runpodctl` binary. An approved replacement base can be passed by digest. The service supplies the pod-scoped credentials at startup. Use the resulting deployed digest in the job specification. For a trainer with additional dependencies, bake those into the reviewed worker before pinning it.

```sh
docker build --platform linux/amd64 -f docker/hf-train.Dockerfile -t "$WORKER_TAG" .
heocr-remote hf-train --profile cpu --job "$JOB" --approved-sha256 "$JOB_SHA" \
  --hf-secret "$RUNPOD_HF_SECRET_NAME" --hours "$HOURS" --run-cap "$USD_CAP" --dry-run
```

For a private worker image, add `--registry-auth-id` with the ID of a Runpod-managed
registry credential that has pull access.

The dry run checks the contract, exact job SHA, disk declarations, local denylist and live quote. Remove `--dry-run` after the launch decision. The launcher records the reservation before create, prints the run ID and pod ID, and watches for container startup. The Secure Cloud pod exposes no SSH port. Keep the launch and any monitoring outside the local power scheduler. After startup is confirmed, the Mac can sleep or disconnect while the pod works. An ambiguous create keeps the reservation open and must be reconciled before another launch. [Runpod pod create API](https://docs.runpod.io/api-reference-v2/pods/create-a-pod)

For an approved unattended CPU launch, pass `--yes` to `hf-train`. The launcher queues by default when the account is full, with a six-hour limit that `--max-wait HOURS` can override. Keep the launching process alive while it waits. It logs the shortage and rechecks about every five minutes. The local ledger records the wait. The billable wall limit, spend reservation and startup watchdog begin only after admission. Capacity wait expiry and errors in account verification, budget, or create return a failure to the supervisor.

The account-side startup watchdog polls Runpod GraphQL for runtime and desired status in the same response. A desired status of `RUNNING` with null runtime remains in startup. Set `--startup-minutes` to a shorter window than the default 15 minutes when the image supports it. By the startup deadline the runner requires confirmed runtime or a confirmed departure from the account. It terminates the exact pod and confirms teardown if a fresh response still shows no running container. Missing or invalid telemetry produces an error and keeps the reservation available for another check. [Runpod GraphQL pod telemetry](https://github.com/runpod/docs/blob/main/sdks/graphql/manage-pods.mdx) · [Runpod CLI runtime semantics](https://github.com/runpod/runpodctl/blob/main/AGENTS.md)

For a v2 job on a worker built with startup receipt support, add `--require-worker-receipt` and `--hf-repo-scope`. The local Hub token must be able to read that private dataset. The worker publishes an empty running receipt before it downloads inputs. The launcher checks the receipt against the reserved pod, job and CLEAN digests before confirming startup. If the container runs without a verified receipt through the startup deadline, the launcher requests deletion and checks teardown. In this mode, R2 GETs must remain valid through the reserved wall deadline when admitted. Use a newly built image for this option. Published older images use the container-only startup check.

The reservation also stores the startup deadline. If the launcher is interrupted during provisioning, an account-checked `heocr-remote sweep` can repeat the startup check for the recorded pod. The sweeper preserves pods with confirmed startup until their normal wall deadline. The startup failure is recorded after confirmed termination. The reservation stays open while billing is pending, and a later sweep records the provider charge when available.

A training incident had a provider reservation and null runtime through the wall deadline. An operator terminated the pod, and the private output contained no artifacts. The in-container guard could arm only after the entrypoint began. The input fetch runs after that setup. The remaining provider evidence does not distinguish host provisioning, image pull, or a container that exited before startup could be observed.

The pod arms its deadline guard before making Hub requests. It checks that each pinned source repository matches its declared visibility, downloads only declared files, verifies byte sizes and SHA-256, and starts the pinned job under `/heocr` with inputs in `in/`. Write each expected snapshot to a temporary name and atomically rename it to the declared filename in `out/` once complete. Each declared snapshot is immutable after rename and has its own size limit. The supervisor commits snapshots as they become available, each alongside a checksum receipt under `runs/<run-id>/`. A v1 receipt records success after the trainer exits cleanly and every snapshot is committed. Trainer logs remain pod-local for v1 jobs.

The deadline uses the absolute launch cutoff. The container timeout stops the worker shortly before cutoff, leaving time for shutdown, and the guard independently attempts pod deletion at cutoff. The pod also attempts deletion after success or failure. A provisioning failure before the entrypoint starts or a Runpod deletion outage can require the account sweeper or console. Before approving a real-data launch, use a separately approved invented-input smoke run to verify managed-secret injection, snapshot publication and self-termination on the actual image and account.

### Checkpoint telemetry and private training logs

Set `schema` to `heocr_hf_training.v2` and include `telemetry` in the exact job spec to enable checkpoint metrics and log publication. Explicit byte limits apply to each artifact, including the runner's full log. Use a private output repository approved to retain trainer logs.

```json
"telemetry": {
  "metrics_max_bytes": 64000,
  "trainer_log_max_bytes": 8000000,
  "run_log_max_bytes": 8000000,
  "setup_log_max_bytes": 8000000
}
```

For each declared snapshot, atomically rename both `<stem>.training-metrics.json` and `<stem>.trainer.log` into `out/` after closing them. `<stem>` is the snapshot filename with its final extension removed. Close and publish the snapshot the same way. The supervisor waits for all three files and commits them atomically with the running receipt. The trainer can also close and write `out/setup.log` and, on failure, `out/partial.trainer.log`. The supervisor adds those logs when present and publishes its own closed `training.log` with the final receipt. All published artifacts must be nonempty. Avoid logging prompt or environment data that is outside the approved run scope. The supervisor gives v2 runs an earlier shutdown signal to leave room for bounded log publication before the absolute deadline.

If the trainer logs on a different cadence from its checkpoints, a completed interval may have no progress lines. Before publishing its metrics and snapshot, close the interval log with the worker's helper:

```sh
/opt/heocr-venv/bin/python -m heocr_compute.remote.trainer_phase_log \
    work/phase.log "out/$stem.trainer.log" "$previous_visit" "$completed_visit"
```

The helper preserves existing progress lines, appends the completed visit window and atomically moves the closed log. It creates a nonempty window record for a quiet interval. Use one writer and keep the phase log and destination on the same filesystem. This avoids tying a checkpoint's durability to whether an optimizer step happens to land on a progress-log boundary.

The v2 receipt lists the checksum and size of every committed file. A successful receipt requires all checkpoint groups and the full `training.log`. A failed receipt may contain completed groups and partial failure logs. The supervisor starts `training.log` before input staging, using fixed phase messages so input failures still have private diagnostics. With startup receipts enabled, the same phase messages also appear in the container log. If a Hub commit cannot be confirmed, the supervisor preserves the latest visible receipt and emits a fixed error with the HTTP status when available. The collector checks group completeness, filenames, sizes and digests against the reservation.

Tesseract trainers can apply `heocr_compute.remote.tesseract_telemetry.instrument` to `src/training/lstmtraining.cpp` at the pinned `SOURCE_REVISION`. This changes the CLI stopping counter to sample attempts and emits one `HEOCR_VISIT` record per main-loop attempt. `summarize` joins the exact attempt window to the ordered lane ledger. The training loss is native activation RMS, and BCER is native bag-of-characters error over recoder outputs. Both are fractions from the same training forward pass. Skipped attempts get no loss or BCER denominator. A missing or noncontiguous trace fails the window.

## Collect and reconcile

Install the Hub client with `pip install '.[render]'` and configure the local runner Hub token for read access to the output repo. Run the collector after pod teardown.

```sh
heocr-remote collect-hf "$RUN_ID" --results "$PRIVATE_RESULTS"
```

Collection pins the current private output revision once, checks the run and job IDs against the ledger, and verifies every returned snapshot's size and SHA-256 before exposing the result directory. Earlier committed snapshots can be collected after an incomplete run. When a known pod is gone and that private revision has no receipt, collection returns a failure and records any positive provider charge as `missing_receipt`. The hours field in this case is an observation or deadline bound, and the provider supplies the charge. A later valid receipt can update the record. If billing is delayed, collect again to reconcile it. The original estimate remains reserved until provider billing is available. The regular sweeper preserves autonomous pods before their recorded deadline even if the launcher is gone. It also preserves autonomous pods belonging to another launcher's ledger, unless explicitly invoked with `--all`.

## Published worker

The [Publish training worker workflow](../.github/workflows/publish-training-image.yml) builds the Linux AMD64 worker from main and publishes it as `ghcr.io/heocr/heocr-compute-train:sha-<commit>`. Its job output and summary give the image reference by digest for the job spec. The build context contains only the committed launcher code, package metadata and worker build files.

GitHub gives the first package version private visibility. After the first push, a HeOCR package administrator sets `heocr-compute-train` to Public. The workflow uses the repository's automatic token for the push, then verifies an anonymous pull of the digest with a clean Docker configuration. Repeat that check after the one-time visibility change. [GitHub Container registry visibility](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)

### Kraken CPU worker

The [Kraken training image workflow](../.github/workflows/kraken-train-image.yml) builds the Linux AMD64 image and runs an offline one-epoch smoke on invented lines. After review, dispatch the workflow on `main` to publish the checked image and obtain its digest. The worker uses the same pinned Kraken 7.2 and CPU PyTorch dependencies as the Linux comparator, with the autonomous supervisor as its entrypoint.

Pin the published digest in the job specification. Use `--profile cpu` and pass `--hf-repo-scope` with the single private dataset approved for the job. All input pins and the snapshot destination must be in that dataset. Use a managed Hub token scoped to read and write that exact dataset. The launcher checks the scope before reserving budget, and the worker checks it again before any Hub access. For a private image, pass the approved Runpod registry credential ID as described above.

### Kraken CUDA worker

Select `kraken-train-cuda` in the same workflow to publish a Linux AMD64 worker with CUDA 13.0 and a hash-pinned Torch 2.14 runtime. Use the published digest with the GPU profile and the same private repository scope. CI checks the CUDA build and runs the invented-line smoke on CPU. On the intended pod, verify that `torch.cuda.is_available()` is true and that the same invented-line smoke completes with `ketos --device cuda:0` before staging a real fit.
