# Hebrew handwriting candidate readers

| Reader | Scope | Status |
|---|---|---|
| cyttic S44 TrOCR-style encoder-decoder | Modern handwriting, pre-cropped lines | Runnable candidate. Eval must verify preprocessing, text direction, and training-source separation before registration. |
| cyttic Android base and large | Modern handwriting, pre-cropped lines | Separate preregistered ONNX CTC models. Prep and decoder pending eval’s amendment. |
| ivrit.ai | Modern handwriting | Publishes a benchmark and leaderboard. The [model catalogue](https://huggingface.co/ivrit-ai/models) has no standalone OCR checkpoint identified. |
| sivan22, oln-1, and YS17121712 TrOCR checkpoints | Hebrew handwriting | No explicit upstream weights licence identified in the [sivan22](https://huggingface.co/sivan22/testing-trOCR-hebrew-handwritten), [oln-1](https://huggingface.co/oln-1/hebrew-htr-trocr), or [YS17121712](https://huggingface.co/YS17121712/hebrew-trocr) model repository. |
| PP-OCRv6 Samaritan | Samaritan manuscript lines represented in square Hebrew Unicode | Narrow discovery candidate with Apache-2.0-declared weights. It needs a matching stratum and source-exposure review. [Upstream model card](https://huggingface.co/johnlockejrr/ppocrv6-sam-heb/tree/5a8254c002fc21381e1bfa662a29c97a42460047) |

## Run cyttic S44

The [pinned cyttic S44 release](https://huggingface.co/cyttic/exp38-trocr-hebrew-s44/tree/5014d29efacd1c7224a27e561197caf963f6ac4a)
declares MIT. Cite the author and pinned model revision if selected.
The runner verifies the weight, generation config, model config and tokenizer
files against SHA-256 entries in [assets.json](assets.json). The checkpoint
takes 1.2 GB on disk.

For Python 3.12 on Linux AMD64:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r comparators/requirements-trocr-linux-amd64.txt
```

On macOS ARM (macOS 14 or later), use
`python -m pip install -r comparators/requirements-trocr-macos-arm.txt`
instead. The lock files pin package versions. The macOS ARM runner produced
nonempty UTF-8 output from an invented line in a smoke test.

```sh
python comparators/runner.py fetch --model trocr-s44 --assets /path/to/assets
python comparators/runner.py run --model trocr-s44 --assets /path/to/assets \
  --mode line --input /path/to/line.png --output /path/to/line.txt
```

S44 accepts a single line. For pages, eval would first need to freeze a
segmenter and an ordering policy, then join the recognized lines in that order.
The released config specifies a 384-pixel RGB encoder and includes its
generation settings. It does not specify image preprocessing. This adapter
uses a bilinear 384 × 384 resize, scales RGB values to [−1, 1], and decodes with
the pinned tokenizer. The text is passed through as UTF-8 with a newline.
Before registering S44, verify the model's input convention, Unicode order
and training lineage with the author or other upstream evidence.

— roster[openai/codex#shay-macbook-m4]@heocr/opensource
