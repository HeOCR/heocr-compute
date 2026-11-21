# Linux AMD64 comparator images

The [Publish comparator images workflow](../.github/workflows/publish-comparator-images.yml)
builds CPU images from `main` and stores them in the HeOCR GitHub Container
Registry. It checks the architecture and runtime offline before pushing. Each
job summary records the immutable `ghcr.io/...@sha256:...` reference, checked
by pulling it back, and the source revision. Use that reference in the run
record. A tag identifies the build and may change.

| Image suffix | Runner | Model argument | Redistribution notice |
|---|---|---|---|
| `tesseract` | Tesseract 5.5.3 | `best`, `fast`, `rivok-full`, `rivok-lite` | — |
| `paddle` | Rivok Paddle ONNX | `paddle` | — |
| `kraken` | Kraken 7.2 | `ppocrv6-small`, `biblia-general`, `biblia-ashkenazi`, `biblia-italian`, `biblia-sephardi` | Contains non-open-source but redistributable components |
| `trocr` | TrOCR-style CPU candidate | `trocr-s44` (line input) | Contains non-open-source but redistributable components |

## Verified pod image digests

The [successful publishing run](https://github.com/HeOCR/heocr-compute/actions/runs/44883953606) built and pulled back
these Linux AMD64 images from commit `9b98a09a428934b807778e95154754b8e8921b5f`.
All four include the in-image NOTICE and support the [batch scorer transport](CONTRACT.md).

| Runner | Image reference |
|---|---|
| `tesseract` | `ghcr.io/heocr/heocr-compute-comparator-tesseract@sha256:a2db81807526bb41631f52a9ba4f8fe3d2a98d2ea9941c2ca124d8e3a9194e7c` |
| `paddle` | `ghcr.io/heocr/heocr-compute-comparator-paddle@sha256:6f79d33371caefc4c22036946c80b1bff695cac5ce670690da99ba18eadd3574` |
| `kraken` | `ghcr.io/heocr/heocr-compute-comparator-kraken@sha256:991f1181ac65ba54e32d2133d6baec2b8ba49be341fe2e6cbe5ab8a0b301be49` |
| `trocr` | `ghcr.io/heocr/heocr-compute-comparator-trocr@sha256:7857201325b3148f4d4d2ba6249d9fcea5db7f833f5249a661157fda45e8514d` |

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

For a private registry image, set `--registry-auth-id` on `heocr-remote hf-train`
to the ID of a Runpod-managed registry credential with pull access. Use the approved image digest in the
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
Builds that include the [NOTICE](../docker/comparators/NOTICE) install it at
`/usr/share/doc/heocr-comparator/NOTICE`. The Kraken and TrOCR builds also
install a copy of Intel's ISSL in that directory.

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
- Kraken and TrOCR **contain non-open-source but redistributable components**.
  Intel oneMKL ships in the Linux AMD64 PyTorch 2.14.0+cpu wheel under the
  [Intel Simplified Software License (ISSL)](https://www.intel.com/content/dam/develop/external/us/en/documents/pdf/intel-simplified-software-license.pdf).
  Intel [permits redistribution in public containers](https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html).
