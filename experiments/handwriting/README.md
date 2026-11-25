# TRAIN-only Kraken handwriting pilot

Fine-tune the Apache-2.0 [Kraken PP-OCRv6 Medium](https://zenodo.org/records/21788410)
recognizer. `train.py` pins the original weights, Kraken 7.2, CPU, seed, RTL
settings, normalization and a three-epoch AdamW schedule. It publishes one private
safetensors file via the existing autonomous supervisor.

## Prepare the inputs

Stage `train.sh`, `train.py`, `manifest.json`, `lines.tar` and
`starter.safetensors` as exact inputs of an approved autonomous training job.
Set `entrypoint` to `train.sh`, and declare `handwriting-medium-pilot.safetensors`
as the only snapshot. Pin the reviewed Kraken CPU image and exact input file
sizes and digests. Bind the source receipts, cloud scope and manifest/archive
digests in the job's CLEAN receipt. Use the existing launcher and its dry run
with a per-run cap of $3. [Autonomous training contract](../../docs/autonomous-hf-training.md)

The data handoff uses `heocr.handwriting_train.v1` with `split` equal to
`train` and `archive_sha256` set to the tar SHA-256. `sources` lists the opaque
source `id`, its exact `clean_sha256`, `runpod_allowed: true` and `partner: false`.
The source custodian supplies the permission flags from the signed records.
`rows` gives the ordered line `id`, `source`, connected writer/collection `group`,
`split: "train"`, `role` (`fit` or `monitor`), `image_sha256` and `label_sha256`.
The tar contains exactly `lines/<id>.png` and `lines/<id>.gt.txt` for each row.
Use the same writer or linked collection key across all derivatives and sources.
For deterministic staging, sort paths and set tar owner, group and timestamps to
zero. The trainer accepts only declared file members and verifies every pair.

The `monitor` rows form a writer-disjoint internal subset of the admitted TRAIN
data. They select the checkpoint on the worker and remain TRAIN. For the
independent DEV comparison, keep the entire cohort and its references off the pod.

## Compare the checkpoint locally

Prepare a separately cleared `heocr.handwriting_dev.v1` manifest with `split: "dev"`.
It lists `rows` with `id`, `group`, `split`, `image_sha256` and `label_sha256`.
The manifest directory contains the same `lines/<id>.png` and `.gt.txt`
file convention. Pin the manifest, checkpoint and Tesseract executable and
use `tessdata_best` 4.1.0 `heb` at commit
`e2aad9b983032bb1beff9133104a67cdbb87ca4d`. Install the appropriate
[Kraken comparison lock](../../comparators/KRAKEN.md) for the host.

```sh
python compare.py --manifest "$DEV_MANIFEST" --manifest-sha256 "$DEV_SHA" \
  --candidate "$CHECKPOINT" --candidate-sha256 "$CHECKPOINT_SHA" \
  --starter "$STARTER" --tesseract "$TESSERACT" \
  --tesseract-sha256 "$TESSERACT_SHA" --tessdata "$TESSDATA" --output "$RECEIPT"
```

Both readers receive identical crop bytes. Kraken uses full-image line mode
with RTL reordering. Stock Hebrew uses OEM 1 and PSM 13. The scorer preserves
literal characters under NFC and whitespace collapse, counts valid blank
outputs as deletions and fails on reader errors. It writes per-line counts and
a writer-cluster paired interval. `improved` requires a strictly negative
upper paired bound across more than one independent writer group. The
optional starter score separates the gain from adaptation from the gain
already present in the base recognizer.
