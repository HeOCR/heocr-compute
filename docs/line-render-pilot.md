# Bounded CPU line-render pilot

Use `heocr-remote render-lines` for an open synthetic training batch. It selects up to 20,000 seeded lines from the version of the [CC0 US-and-Israel plain-text corpus](https://huggingface.co/datasets/HeOCR/htext-pby/tree/fd5eba052cb5957f3a2c14f6613fb54f6176d7c2) published as `pby-v3`. The source card, byte size and SHA-256 are checked in preflight. The pod checks the downloaded bytes again. Selection uses 20–48 character Hebrew lines within a small shared font repertoire. It keeps source line IDs and produces the same selection for the same source, count and seed.

Install the optional publisher with `python -m pip install '.[render]'`. In the runner's existing private RunPod TOML credential file, add `hf_token` for a fine-grained write token scoped to the one existing private HeOCR output dataset. Use file mode 600. The token stays in the local Hub client's memory and authenticates to the canonical Hub endpoint. The upload uses the normal HF commit path and requires the dataset to already exist. Keep its visibility private while the upload runs and after it finishes. Dry run checks the authenticated repository and token class. Effective write permission is checked by the actual commit.

Use an approved private dataset name for `DATASET`. Start with this preflight for the default renderer and font:

```sh
heocr-remote render-lines --dataset "$DATASET" --count 20000 --seed 19 --dry-run
```

The default hocrsyngen code pin is `6632c6cfc7a33e93fe89ac9de8fc8bc181019a4f`, with Noto Sans Hebrew and its governed OFL font manifest. For handwriting, the lead supplies a reviewed public commit that implements the new mode, the SHA-256 of that commit's governed font manifest, and the chosen font IDs. The same recipe calls `hocrsyngen render-lines` with the requested mode and uses its normal manifest, summary, image and label contract:

```sh
heocr-remote render-lines --dataset "$DATASET" --count 20000 --seed 19 \
  --code-revision "$RENDERER_COMMIT" --font-manifest-sha256 "$FONT_MANIFEST_SHA256" \
  --font "$FONT_ID" --mode "$MODE" --dry-run
```

The CPU template uses four `cpu3c` vCPUs and 40 GB of ephemeral disk. The operator can lower the four-hour wall limit and $1 per-run cap. The live quote reserves startup headroom and counts open reservations against the monthly cap. The wall limit covers provisioning, the job, and the bundle pull. Rendering can start only after the pod has armed its own termination timer. The runner also terminates and verifies the pod on success, failure, timeout or interruption. If RunPod cannot confirm termination, the runner reports an error and leaves the spend reservation open for the sweeper. No publication starts until teardown is confirmed.

On success the pod produces `render.tar` and `artifact-manifest.json`. The manifest records the source, code, fonts, mode, seed, count, and SHA-256 and size of every archive member, including each PNG and its label. A partial render, a refused line, a missing image or label, a hash mismatch, or a bundle exceeding the transfer limit fails the run. The Mac needs 3 GiB of free workspace to receive and verify the bundle. It relays the verified files to `renders/<run-id>/` in a single private HF commit, then verifies the stored bytes by immutable commit metadata and removes the local archive. If publication or readback fails, the local archive is retained for recovery. The Mac performs the existing restricted decontamination review before any training use.

The dry run checks source and destination metadata, the generated input manifest and policy, eligible account, quote, balance and caps. It performs only read-only remote requests and shows the proposed pod and job. The renderer, archive transfer, and HF write are exercised only by a separately approved job. A full batch in the new mode has yet to establish its runtime and artifact size.
