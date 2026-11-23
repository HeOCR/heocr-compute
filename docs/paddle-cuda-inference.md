# Paddle CUDA inference

Run **Publish Paddle CUDA comparator** from main. Use the AMD64 image digest reported by the workflow. The image keeps the CPU comparator's Paddle wrapper, pinned upstream model assets, word detector, cascade and reading order. It fetches the existing asset pins at runtime and verifies their hashes before inference. The CUDA adapter checks all four ONNX sessions and disables runtime provider fallback.

The operator pins a JSON job with these fields.

| Field | Value |
| --- | --- |
| `schema` | `heocr.paddle_gpu_predictions.v1` |
| `split` | `DEV` or `SEALED` |
| `image` | `ghcr.io/heocr/heocr-compute-comparator-paddle-cuda@sha256:<digest>` |
| `authority_sha256` | SHA-256 of the approved execution record |
| `dev_receipt_sha256` | `null` for DEV, or the verified CUDA DEV receipt SHA-256 for sealed inference |
| `manifest_sha256`, `count` | SHA-256 of `manifest.json` and its exact row count |
| `input` | `account`, `bucket`, `path`, `size`, `sha256` for one uncompressed raster tar |
| `output` | `account`, `bucket`, `path`, `max_bytes` for the predictions object |

The raster tar contains `manifest.json` with an ordered array of `{id,path,sha256}` rows and exactly the listed PNG or JPEG files under `rasters/`. Paths are relative to the manifest. Use an isolated private inference bucket and `inference-runs/` object paths. The output key ends with `/predictions.jsonl`. The authorized operator supplies one object-scoped GET and one object-scoped PUT in `HEOCR_PADDLE_R2_GET` and `HEOCR_PADDLE_R2_PUT` through the secret environment. Sign the PUT with `Content-Type: application/octet-stream` and `If-None-Match: *`. Both URLs expire within 24 hours and must cover the full quoted horizon.

```sh
python -m heocr_compute.remote.paddle_eval --job job.json \
  --approved-sha256 "$approved_job_sha256" --run-cap "$approved_cap"
```

The default mode checks the job digest, live account quote and any supplied URL scope. Pass `--launch` after approval for the exact job and spend. The worker publishes one ordered JSONL of `{id,text}` records after all predictions validate and shuts down on completion or deadline. Record GPU model, image digest, job and output hashes, provider checks, elapsed time and DEV comparison in the operator's CUDA DEV receipt. The operator verifies publication, teardown and final billing from the approved account.
