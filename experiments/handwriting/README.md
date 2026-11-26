# Kraken handwriting pilot

Fine-tune the Apache-2.0 [Kraken PP-OCRv6 Medium](https://zenodo.org/records/21788410)
recognizer. `train.py` pins the original weights, Kraken 7.2, CPU, seed, RTL
settings, normalization and a three-epoch AdamW schedule. The TRAIN monitor
supplies checkpoint selection. Keep that monitor writer-disjoint from fit.

## Prepare Mac-local TRAIN data

The bootstrap renders a pinned TRAIN CLEAN text pool.
It verifies the CLEAN receipt and its units and keep ledgers, and splits two of the
source works into the TRAIN monitor before rendering. Pango
shapes the original Hebrew in RTL order with font fallback disabled. Each
supported line/font pair receives two seeded variants of size, angle, blur,
brightness, and noise at 300 dpi. The generator skips a combination if a
font lacks a character. It leaves the supplied label bytes intact. The three
font faces and their SIL OFL 1.1 licenses are pinned at the
[Google Fonts revision](https://github.com/google/fonts/tree/6ebe737e44fd64ee736bbc6feed6136a6dbfd222/ofl)
in `bootstrap.py`. Keep the license texts next to the fonts. The file names
for the licenses are `gveretlevin-OFL.txt`, `playpensanshebrew-OFL.txt`, and
`amaticsc-OFL.txt`. Use Pango 1.58.2, Pillow 13.0.0 and NumPy 2.4.6 to
reproduce the original rendering, and record the executable and runtime pins.

```sh
python bootstrap.py --clean-root "$TEXT_CLEAN_ROOT" --clean-sha256 "$TEXT_CLEAN_SHA" \
  --source-id "$TEXT_SOURCE_ID" --fonts "$PINNED_FONTS" --output "$JOB/in"
mkdir "$JOB/out"
cp "$STARTER" "$JOB/in/starter.safetensors"
TRAIN_SCRIPT="$PWD/train.py"
(cd "$JOB" && PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python "$TRAIN_SCRIPT" --local)
```

The source receipt applies to the input text. heocr-clean must
review the resulting exact TRAIN payload against the registered DEV slice
before DEV scoring. Generated crops and their checkpoints stay local. The
bootstrap reports TRAIN counts, and the one-shot benchmark uses real handwriting.

When heocr-clean releases paired local TRAIN batches, pass each batch with
`--local-train-batch CLEAN_RECEIPT SHA256` in the same staging command.
Obtain that SHA independently from heocr-clean. The receipt handoff format is
`heocr.handwriting_train_batch.v1` with `split: "train"`, `verdict: "TRAIN CLEAN"`,
`batch`, `runpod_allowed: false`, `partner: false`, `rows_sha256` and
`archive_sha256`. The signed hashes identify `rows.jsonl` and `lines.tar`
adjacent to the receipt. The rows use the common pair format below, omitting
`source`, and must use the same opaque `group` and TRAIN role for each writer
across batches. Staging checks both signed files and every member, joins the
batches in command-line order and refuses a writer or image across both roles.
Only the separately supplied TRAIN batch receipts enter this command.

For routine intake, have the CLEAN publisher hand off a pinned local index
and call `compose_local.py` from its batch-release step. This reuses the
existing base TRAIN pair bundle, so adding a batch does not rerender earlier
crops. Use a fresh output directory for each index. The new manifest binds
both the base and index hashes, preserves row tags including
`label_unreviewed`, and keeps Mac-only restrictions. `local-train.json` uses
`schema: "heocr.handwriting_train_index.v1"`, `split: "train"` and `batches`
with one `id` and `clean_sha256` per approved batch. Each batch supplies
`<id>/clean.json`, `<id>/rows.jsonl` and `<id>/lines.tar` alongside the index.
The receipt's `batch` must match its indexed `id`. A receipt with
`row_tags_sha256` also binds `row-tags.jsonl`, which lists each admitted row's
`id`, `group` and `tags`. The importer joins these tags to the same writer
and row and retains provenance already present in the pair list. The index digest comes
from the same trusted release handoff as the receipts.

```sh
python compose_local.py --base "$BASE_DIR" --base-sha256 "$BASE_MANIFEST_SHA" \
  --index "$CLEAN_INDEX" --index-sha256 "$CLEAN_INDEX_SHA" --output "$STAGED_IN"
```

The composer processes only indexed CLEAN batches and publishes the new
bundle after all batches validate. The staging step does not launch training.
Use the Mac-local trainer for the composed bundle after binding the exact
input manifest and archive to the intended run.

## Common staging format and cloud lane

The data handoff uses `heocr.handwriting_train.v1` with `split: "train"` and
`archive_sha256`. `sources` lists `id`, `clean_sha256`, `runpod_allowed` and
`partner`. `rows` gives the ordered line `id`, `source`, linked writer or
collection `group`, `split: "train"`, `role` (`fit` or `monitor`), `image_sha256`
and `label_sha256`. The tar contains exactly `lines/<id>.png` and
`lines/<id>.gt.txt` for each row. The trainer checks every member.

For a separately approved cloud job, stage `train.sh`, `train.py`,
`manifest.json`, `lines.tar` and `starter.safetensors`. Use `train.sh` as the
entrypoint and `handwriting-medium-pilot.safetensors` as the single snapshot.
Pin the Kraken CPU image, exact input sizes and digests, and signed transfer
permissions. The cloud entrypoint requires `runpod_allowed: true` and
`partner: false` for all sources. Follow the [autonomous training contract](../../docs/autonomous-hf-training.md)
and its dry run with a per-run cap of $3.

## Registered DEV one-shot

Run locally after eval signs actual use. Give the runner the actual-use
receipt and its independently supplied hash. The runner checks its binding
and relies on eval custody for the signature. The machine receipt contains
the integer `registration`, `actual_use: "signed"`, `manifest_sha256`,
`candidate_sha256`, `terminal_sha256` and `train_clean_sha256`. The latter
identifies the CLEAN for the actual TRAIN payload, including the DEV slice
text check. The scorer does not need the TRAIN text or its overlap index.

The eval-admitted `heocr.handwriting_dev.v1` has `split: "dev"`. Each row
has `id`, opaque writer `group`, `page`, `line` (1–29), `split`, `layout`
(`printed_sheet` or `separate_page`) and the joint `retained` boolean after
writer/text masking. Include metadata for all 29 lines of each page. Supply
`image_sha256` and `label_sha256` and the adjacent `lines/<id>.png` and
`.gt.txt` pair for each retained row. The same bytes go to both readers.
Install the appropriate [Kraken 7.2 lock](../../comparators/KRAKEN.md) for the
host, and use `tessdata_best` Hebrew at revision
`e2aad9b983032bb1beff9133104a67cdbb87ca4d`.

```sh
python compare.py --registration "$REGISTRATION" --actual-use "$EVAL_PERMIT" --actual-use-sha256 "$PERMIT_SHA" \
  --manifest "$DEV_MANIFEST" --manifest-sha256 "$DEV_SHA" \
  --candidate "$CHECKPOINT" --candidate-sha256 "$CHECKPOINT_SHA" \
  --fold "$LINE_FOLD_V1" --tesseract "$TESSERACT" \
  --tesseract-sha256 "$TESSERACT_SHA" --tessdata "$TESSDATA" --output "$RECEIPT"
```

Kraken treats the full 300-dpi crop as a line with RTL reordering, loading
once per batch of up to 32. Stock Hebrew uses OEM 1, PSM 7 and DPI 300. The
runner pins the `line_fold_v1` profile and counts empty folded output as
line deletions. It issues HOLD without opening pairs when the retained
support is below four writers or 100 lines. The paired micro-CER difference
uses 10,000 writer-cluster draws with seed 20261125, nearest-rank 95% bounds
and percentage points. The upper bound must be below zero and candidate
empty share must not exceed stock empty share. The receipt has pooled and
layout metrics, with layout scores withheld when there is only one writer.
It contains no writer or line rows. If an authorized attempt is interrupted
or fails before publishing a receipt, rerun the same pinned comparison. The
runner publishes the receipt atomically and preserves a completed result.
