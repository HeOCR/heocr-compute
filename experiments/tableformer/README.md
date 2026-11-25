# Hebrew TableFormer pilot

This offline harness starts from Docling TableFormer ACCURATE,
`docling-ibm-models==4.0.4` at source `a14d633b`, with the exact
`fc0f2d45` pretrained assets. See [asset pins](assets.py) and the
[upstream release](https://github.com/docling-project/docling-ibm-models/tree/v4.0.4).
The candidate is a normal TableFormer safetensors state dict. The
initial recipe updates only the OTSL token classifier. The encoder,
decoder representations and box decoder keep the stock parameters.
The loss is causal, teacher-forced next-token cross entropy.

## Local setup

Use Python 3.12 and `requirements-dev.txt` for the CPU reference replay
(Mac arm64). The Linux x86_64 training dependency lock uses wheel
hashes. Install it with
`python -m pip install --require-hashes --only-binary=:all: -r requirements-train-linux.txt`.
Choose a CUDA GPU and container driver compatible with that Torch
build before quoting a pilot. Keep the reviewed worker image and
Python runtime pinned by digest in the job. [Train overlay pins](requirements-train.in)

Supply the existing local Docling snapshots in this form.

```text
models/docling-project--docling-models/model_artifacts/tableformer/accurate/tm_config.json
models/docling-project--docling-models/model_artifacts/tableformer/accurate/tableformer_accurate.safetensors
models/docling-project--docling-layout-heron/
```

Run the invented-input smoke to check weight loading, a real optimizer
step, and agreement between the cached upstream decoder and the causal
teacher-forced states.

```sh
python train.py --models models --smoke --output smoke-a
python train.py --models models --smoke --output smoke-b
cmp smoke-a/receipt.json smoke-b/receipt.json
cmp smoke-a/tableformer_accurate.safetensors smoke-b/tableformer_accurate.safetensors
```

Byte repeatability is checked on a fixed runtime and device. Run receipts
record the upstream and harness hashes, runtime, seed, source and output
pins and losses. The smoke cannot load a data manifest.

## DEV baseline

Set `PYTHONPATH` to pinned `heocr/src` and `heocr-bench/src` checkouts
and make the local OCR models available through `HEOCR_MODELS`.
The baseline takes a SHA-pinned `heocr.tableformer_dev.v1` manifest
with `split: "dev"` and `pages` entries. Each entry includes `id`,
document `cluster`, `capture` (`issuer-render`, `scanner`, or `phone`),
`direction`, a `source: {path, sha256}` and a list `gold` of
`{path, sha256}` table JSON assets. `page` is a 1-based PDF page or
null for a standalone image. Paths are relative to the explicit
packet root.

```sh
python baseline.py --root dev --manifest dev/manifest.json --manifest-sha256 "$DEV_SHA" \
  --models models --tesseract "$TESS" --heocr-commit "$HEOCR_COMMIT" \
  --bench-commit "$BENCH_COMMIT" --output baseline.json
```

The two lanes are raw Docling structure (TEDS-Struct, structure-based
matching, physical columns mapped to logical order) and HeOCR +
TableFormer (filled-cell TEDS and TEDS-Struct under the scorer's
content-based match). HeOCR's current table exports preserve explicit
spans and omit leading-header metadata. The scorer sees that limitation.
A declared scan forces OCR to exclude hidden text layers. Each manifest
page is isolated before inference. Unmatched gold scores zero, and
unmatched or invalid predictions are counted. The receipt contains no
predicted or reference text.

Supplying both `--candidate FILE --candidate-sha256 SHA` changes the
TableFormer pin for that local replay. Model/config hashes, page inputs,
HeOCR, the scorer, and the layout model stay paired. This is an
evaluation-only session and leaves package defaults and released
configuration unchanged.

## CLEAN TRAIN and bounded job

Input is a separate `heocr.tableformer_train.v1` manifest with
`split: "train"` and `examples` entries. Each entry has a unique
`id`, `source_id`, `split: "train"`, a pre-cropped `image` and
`gold` (`{path, sha256}` each). Use one visible physical fragment per
crop, preserving its printed header copies. Gold uses `heocr_table.v1`,
explicit empty cells and exact spans. The converter mirrors RTL anchors
once and generates OTSL span continuations. Split logical cross-page and
side-by-side tables at their independently verified physical boundaries
before signing the derived manifest. Exclude clipped or masked fragments.

The independently signed TRAIN CLEAN receipt must bind the exact
fragment manifest in its `files` map. Stage the signed receipt, source
manifest, all pinned assets and the entrypoint using the existing
[autonomous private-Hub job](../../docs/autonomous-hf-training.md).
Stage only the approved TRAIN packet and stock model files on the pod.
Install the hash lock in the reviewed image rather than during the job.
Run using one epoch, seed 32 and LR 2e-5 for the first pilot.

```sh
python train.py --root in/train --manifest in/train/manifest.json \
  --manifest-sha256 "$TRAIN_SHA" --clean in/clean.json --clean-sha256 "$CLEAN_SHA" \
  --models in/models --device cuda --output candidate
```

The supervisor can publish the completed safetensors and receipt as two
immutable snapshots. Set snapshot sizes from their known byte counts and
use the existing job wall-time and spend limits. Select locally using a
paired DEV replay with the exact same manifest and scorer plus a paired,
local calibration field witness check. Require increased filled-cell
TEDS on the generated DEV set and zero lost calibration field witnesses
before promoting a candidate. Keep authored DEV references, calibration
sources and results out of training and the remote runtime.

The [P3 recipe](P3.md) trains the structure and box decoders after phone rectification.
