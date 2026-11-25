# Table comparison runners

Both CPU lanes accept the ordered, hash-pinned 200 DPI page raster roster.
The runtime contains pinned Tesseract 5.5.3 and the registered tessdata_best
Hebrew weights. Its cell path uses PSM 6 on each predicted cell box.
The runtimes and package locks target Linux AMD64 and Python 3.12.

- **PP-StructureV3** uses PaddleOCR 3.7.0, PaddleX 3.7.0 and PaddlePaddle
  3.3.1, with the default structure and layout branches. The release's
  PP-Structure language map has no Hebrew. The adapter uses the native
  HTML grid and cell boxes with external Hebrew cell text. When the native
  box count differs, the adapter retains the valid grid and records a
  geometry issue with empty cell text.
  [Upstream usage](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html)
- **Table Transformer** uses Transformers 5.24.1, timm 1.0.31, the
  detection and v1.1-all checkpoints, and Microsoft's postprocessor
  pinned to `16d124f616109746b7785f03085100f1f6247575`. The
  postprocessor uses PyMuPDF, distributed under AGPL. Detection and
  structure threshold 0.5 and detection crop margin 10 px follow
  Microsoft's inference defaults.
  [Upstream model and inference implementation](https://github.com/microsoft/table-transformer/tree/16d124f616109746b7785f03085100f1f6247575)

The inventories `paddle-models.json` and `tatr-models.json` give
the exact upstream revisions, file sizes and SHA-256 digests. The
image build checks the bytes after each first download. Inference
reads only the image's verified, local models.

## Standalone entry

Run this inside the corresponding `tables-paddle` or `tables-tatr`
image using `--entrypoint /opt/tables/bin/python` for a local call:

```sh
/opt/tables/bin/python /opt/table-comparator/worker.py \
  --lane tatr --manifest /input/pages.json --manifest-sha256 "$MANIFEST_SHA" \
  --models /opt/models --tessdata /opt/tessdata \
  --output /output/rows.jsonl --issues /output/issues.json
```

The raster manifest is `{"schema":"heocr.table_comparator_pages.v1","dpi":200,
"pages":[{"id":"document-id","page":1,"path":"page.png","sha256":"..."}]}`.
Paths are relative to the manifest. Each input document appears once in
`rows.jsonl`, in first-input occurrence order, as `{"id":"document-id",
"tables":[...]}`. Tables contain physical page bounds, logical RTL anchors
and spans in `heocr_table.v1`. A document with no table has an empty list.
An invalid model grid is recorded in the issues output.

For an autonomous private DEV job use the supervisor image entrypoint and
`run.sh` as the job script with lane and manifest SHA arguments. The
runner takes a pinned `dev-pages.tar` with only the page manifest and
referenced rasters. Publish `rows.jsonl` and `issues.json` as the
declared outputs.

No reference texts, fixture images or DEV predictions ship with the
runtime. CI uses an invented raster to exercise the offline path.
