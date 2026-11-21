# Open comparator runners (partial)

This directory supplies stock and Rivok Tesseract, plus Rivok's ONNX flagship.
It is **not a complete or frozen launch roster**. Admission, reference-source
exclusions, settings and input/output normalization must be fixed by the evaluator
before any held-out run. Never use invented pages to report model performance.

## Setup

These CPU recipes target Python 3.12 and Tesseract 5.5.3 on macOS or Linux AMD64.
Install Tesseract 5.5.3 from a verified distribution or its [source release](https://github.com/tesseract-ocr/tesseract/releases/tag/5.5.3),
including its image-library dependencies. The runner checks its reported version,
not the binary or linked-library hashes; record those and the OS in the run record.
See [Linux AMD64 images](IMAGES.md) for digest-pinned container builds. For a
local Rivok Paddle installation, install Git and:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r comparators/requirements-paddle.txt
python comparators/runner.py fetch --model paddle --assets /path/to/assets
```

The requirements pin the tested direct and transitive package versions and the
upstream source commit; package archives, system libraries, Python build and
platform are not hash locked. Re-resolve only as a new run configuration. The
macOS ARM smoke was run; Linux AMD64 and macOS Intel remain unverified. Keep
model assets outside the checkout. Downloads are verified against SHA-256 before
atomic placement and again on every invocation. For each Tesseract model replace
`paddle` with `best`, `fast`, `rivok-full`, or `rivok-lite` in the fetch command.
See [assets.json](assets.json) for exact revisions and digests. Assets and code
are Apache-2.0 as declared by the [stock models](https://github.com/tesseract-ocr/tessdata_best),
[Rivok Tesseract](https://huggingface.co/Rivok/tesseract-heb-rivok), and
[Rivok Paddle](https://github.com/RivoksLab/paddleocr-hebrew/tree/3baae91ad6479ef0473f901b4ed96366ae6c1670).

```sh
python comparators/runner.py run --model rivok-full --assets /path/to/assets \
  --mode page --input /path/to/page.png --output /path/to/page.txt
python comparators/runner.py run --model rivok-full --assets /path/to/assets \
  --mode line --input /path/to/line.png --output /path/to/line.txt
```

Use a separate output file for every image, including blank ones. Tesseract uses
OEM 1 and PSM 3 for pages, PSM 7 for lines. Its UTF-8 output is passed through;
no reordering or normalization is applied. Rivok Paddle uses its upstream
word-detector and full-precision CTC/NRTR cascade for both modes, concatenating
the upstream reading-order lines with newline separators. `line` for this model
still detects words; it is not a separate line recognizer. The pipeline emits
logical Unicode order; do not use a visual bidi transform. It cannot emit niqqud,
so retain this limitation when assessing pointed categories; do not silently strip
reference marks. PDF page selection, multi-column order, bidi controls, blank
outputs and final newline conventions require evaluation-owner agreement and
end-to-end validation against the scorer before a frozen comparison.

## Category roster

Categories: 1 modern unpointed print; 2 old unpointed print; 3 modern pointed
print; 4 old pointed print; 5 modern handwriting; 6 old handwriting. A check
marks registration, not proven suitability or run readiness. All blocked entries
remain in the record.

| Reader | 1 | 2 | 3 | 4 | 5 | 6 | State |
|---|---|---|---|---|---|---|---|
| Tesseract best | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Adapter supplied; freeze runtime and scorer contract |
| Tesseract fast | ✓ | ✓ | ✓ | ✓ | — | — | Adapter supplied |
| Rivok Tesseract full and lite | ✓ | ✓ | ✓ | ✓ | — | — | Separate adapters supplied |
| Rivok Paddle flagship | ✓ | ✓ | ✓ | ✓ | — | — | Adapter supplied; no niqqud |
| Kraken PP-OCRv6 | — | ✓ | — | ✓ | ✓ | ✓ | Pinned candidate runner supplied. Evaluation configuration and exposure review pending. |
| cyttic base and large | — | — | — | — | ✓ | — | Blocked: decoder/tokenizer and training lineage review |
| BiblIA general and script-specific | — | — | — | — | — | ✓ | All four releases admitted for comparison. Eval records which scripts apply to each set. |

[Kraken's small record](https://zenodo.org/records/21788405) declares Apache-2.0
and lists `small.safetensors`; its title resolves a prior small/medium naming
ambiguity, but this does not select the model or segmentation for a run. See [candidate Kraken and BiblIA runners](KRAKEN.md).
See [Hebrew handwriting candidates](HTR.md) for the line-only TrOCR-style runner
and additional discovery work.

Additional language families require a specific eligible pretrained Hebrew reader,
not just framework support. Stock [PaddleOCR multilingual recognition](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
does not list Hebrew in the cited model list; Rivok's model is a separate reader.
[EasyOCR](https://github.com/JaidedAI/EasyOCR/issues/363) Hebrew support is not
established. [docTR](https://github.com/mindee/doctr/blob/v1.1.0/doctr/datasets/vocabs.py)
has a Hebrew vocabulary, but a Hebrew-capable pretrained recognition checkpoint
has not been identified; vocabulary alone is not evidence of a runnable model.
Do not silently remove these discovery items from future consideration. Permission-
gated readers are outside this directory.

— roster[openai/codex#shay-macbook-m4]@heocr/opensource
