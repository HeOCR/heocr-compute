# Raw scorer output

Use batch mode for a frozen comparator run. Give it a UTF-8 JSON array in the
registered input order. Each entry contains the exact `id`, a raster image
`path` relative to the manifest (or absolute), and the image's lowercase
hex `sha256`. IDs must be unique. The manifest contains only input identities
and image pins. Keep reference text separate.

```json
[
  {"id": "invented-page", "path": "invented.png",
   "sha256": "0000000000000000000000000000000000000000000000000000000000000000"}
]
```

```sh
python comparators/runner.py run --model best --assets "$ASSETS" --mode page \
  --manifest "$MANIFEST" --output-dir "$RESULT_DIR" \
  --runtime-sha256 "$RUNTIME_SHA" --config-sha256 "$CONFIG_SHA"
```

For Kraken or BiblIA use `python comparators/kraken_runner.py` and omit
`run`. For supplied crops use `--mode line`. The output directory must be new
and its parent must exist. Use the image digest (without the `sha256:` prefix)
or the SHA-256 of a pinned local runtime receipt for `RUNTIME_SHA`. Use the
digest of eval's frozen run configuration for `CONFIG_SHA`. The launcher must
verify these external pins.

| Output | Contents |
|---|---|
| `pages/00000000.txt`, etc. | One raw UTF-8 file per page or frame in manifest order. The sidecar maps IDs to files. |
| `lines.jsonl` | One LF-delimited JSON record per supplied crop, with `id` and `text`. JSON escapes embedded line breaks without changing the decoded text. |
| `run.json` | Schema, manifest and run digests, runtime and config pins, adapter digests, tool and asset versions or hashes, settings, and one status and text hash per ID. |

Only CRLF and CR transport endings are converted to LF. Keep the reader's
logical Unicode order and every emitted mark, punctuation character, space,
form feed and bidi control. Page reader line order is preserved. A crop gets
one record. An unexpected internal line break is preserved and flagged in the
sidecar, so extra recognitions remain visible to the scorer.

Per-input failures produce empty text with `error` status. A successful
reader result with only whitespace or no text has `blank` status and retains
its emitted bytes. Keep all expected IDs in the scorer input. Invalid manifests
or unavailable or invalid pinned runtimes and model assets stop the batch.
An interrupted batch emits no complete result directory.

Stock Tesseract page defaults here are `--psm 3 -l heb`. Supplied line inputs
use `--psm 7 -l heb` for stock best. Rivok Tesseract uses the registered model's
language name and PSM 3 for pages. Rivok Paddle uses the pinned word detector,
cascade and upstream assembly. Kraken, BiblIA and S44 recognize supplied
crops directly. Admit any page route only after eval fixes its exact
segmentation, reading order and versions. Use the frozen package defaults and
registered model option for HeOCR runs. Matched HeOCR arms use the same
executable and settings.

Eval applies the registered profile symmetrically, including any NFC,
line whitespace or punctuation folding, bidi-control removal and form-feed
folding. Page CER and line assignment use their registered whitespace and
line-break rules. Preserve niqqud in transport. Eval derives stripped views.
