# Linux AMD64 comparator images

The [Publish comparator images workflow](../.github/workflows/publish-comparator-images.yml)
builds CPU images from `main` and stores them in the HeOCR GitHub Container
Registry. It checks the architecture and runtime offline before pushing. Each
job summary records the immutable `ghcr.io/...@sha256:...` reference, checked
by pulling it back, and the source revision. Use that reference in the run
record. A tag identifies the build and may change.

| Image suffix | Runner | Model argument |
|---|---|---|
| `tesseract` | Tesseract 5.5.3 | `best`, `fast`, `rivok-full`, `rivok-lite` |
| `paddle` | Rivok Paddle ONNX | `paddle` |
| `kraken` | Kraken 7.2 | `ppocrv6-small`, `biblia-general`, `biblia-ashkenazi`, `biblia-italian`, `biblia-sephardi` |
| `trocr` | TrOCR-style CPU candidate | `trocr-s44` (line input) |

## Verified pod image digests

The [successful publishing run](https://github.com/HeOCR/heocr-compute/actions/runs/44882839632) built and pulled back
these Linux AMD64 images from commit `d56fb8bb642c216958120e5c20e9471cc641abc1`.
Use their [batch scorer transport](CONTRACT.md) for registered runs.

| Runner | Image reference |
|---|---|
| `tesseract` | `ghcr.io/heocr/heocr-compute-comparator-tesseract@sha256:f4128cb4cb2e953580699414cce539ed0037646a36013ffe06a9813b70843274` |
| `paddle` | `ghcr.io/heocr/heocr-compute-comparator-paddle@sha256:bfee7bf59eb4e14dc8ff8557992a1ac19a1b64436b5a9f6afd76326046d94400` |
| `kraken` | `ghcr.io/heocr/heocr-compute-comparator-kraken@sha256:8305d40999adf546ccebe18e822b773b7db57a803786637e599f097c4d43cad6` |
| `trocr` | `ghcr.io/heocr/heocr-compute-comparator-trocr@sha256:68ed3352f60575a0c2b913881e4b6a907285d7cf59b73d0b8e3734cba27480ac` |

The image digest pins the runtime. The [asset manifest](assets.json) and
[Kraken runner](kraken_runner.py) pin the recognition weights separately.
Stage assets using the runner's `fetch` action or the exact downloads in
[KRAKEN.md](KRAKEN.md). Model weights are outside the build context. Kraken's
upstream package includes its Apache-2.0 BLLA segmenter, whose hash the workflow
and runner check.

For example, after loading authorized assets and inputs locally:

```sh
# Set IMAGE to a full reference with batch transport support.
docker pull --platform linux/amd64 "$IMAGE"
docker run --rm --platform linux/amd64 --network none \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/assets:/assets:ro" -v "$PWD/input:/input:ro" -v "$PWD/output:/output" \
  "$IMAGE" run --model best --assets /assets --mode page \
  --manifest /input/manifest.json --output-dir /output/ocr-results \
  --runtime-sha256 "${IMAGE##*@sha256:}" --config-sha256 "$CONFIG_SHA"
```

For Kraken, omit the `run` action. For the TrOCR candidate use `--mode line`.
For pod use, choose a digest from a publishing run that checked the autonomous
entrypoint. The image starts the existing [private-Hub supervisor](../docs/autonomous-hf-training.md)
when the launcher supplies the approved job spec. In the pinned job entrypoint,
invoke `python /opt/comparators/runner.py` (or `kraken_runner.py`) with the same
arguments as above, using staged inputs and verified assets. The supervisor
checks inputs and publishes declared snapshots, then attempts pod termination.

Set `--registry-auth-id` on `heocr-remote hf-train` to the ID of the Runpod-managed
registry credential with GHCR pull access. Use the approved image digest in the
job spec. Keep data transfer, budget and lifecycle on the authorized autonomous
pod path. Eval freezes the applicable models, segmentation, order and scorer
settings separately.

To build locally from a clean committed revision, use a directory created
from the source paths fetched by the publishing workflow as the Docker build
context:

```sh
docker buildx build --platform linux/amd64 --load --target tesseract \
  --file "$CONTEXT/docker/comparators/Dockerfile" --tag comparator:tesseract "$CONTEXT"
```

— roster[openai/codex#shay-macbook-m4]@heocr/opensource


## Licences and source

The image's source revision identifies the HeOCR adapter and build recipe.
See the [HeOCR licence](../LICENSE) and [Dockerfile](../docker/comparators/Dockerfile).
The recipe records the Python base digest and Debian snapshot. The Debian
packages carry their copyright and licence files under `/usr/share/doc`.
No HeOCR or comparator recognition checkpoints are copied into these images.

- The pod images include the unmodified Runpod CLI v3.1.1 executable under
  GPL-3.0. Its [corresponding source and build files](https://github.com/runpod/runpodctl/tree/72afce1305ad3a25217aafdb9022a16e9b569661)
  and [licence](https://github.com/runpod/runpodctl/blob/72afce1305ad3a25217aafdb9022a16e9b569661/LICENSE)
  are available from that release's source commit.
- Tesseract includes Tesseract 5.5.3, built from the upstream [source archive](https://codeload.github.com/tesseract-ocr/tesseract/tar.gz/refs/tags/5.5.3)
  under [Apache-2.0](https://github.com/tesseract-ocr/tesseract/blob/5.5.3/LICENSE).
- Paddle includes the pinned [Rivok wrapper and notices](https://github.com/RivoksLab/paddleocr-hebrew/tree/3baae91ad6479ef0473f901b4ed96366ae6c1670)
  under Apache-2.0, and [ONNX Runtime](https://github.com/microsoft/onnxruntime/tree/v1.32.0)
  under MIT. The [OpenCV wheel's licence and source records](https://pypi.org/project/opencv-python/5.1.0.95/)
  cover its bundled FFmpeg (LGPL-2.1) and Qt (LGPL-3.0) binaries.
- Keep Kraken and TrOCR private under the project’s open-licences-only policy.
  Their exact Linux AMD64 PyTorch CPU wheel has `AT_MKL_ENABLED() == 1`, and
  `libtorch_cpu.so` contains Intel oneMKL. Intel offers oneMKL under the
  [Intel Simplified Software License](https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html).
