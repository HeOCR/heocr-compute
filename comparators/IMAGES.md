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

The image digest pins the runtime. The [asset manifest](assets.json) and
[Kraken runner](kraken_runner.py) pin the recognition weights separately.
Stage assets using the runner's `fetch` action or the exact downloads in
[KRAKEN.md](KRAKEN.md). Model weights are outside the build context. Kraken's
upstream package includes its Apache-2.0 BLLA segmenter, whose hash the workflow
and runner check.

For example, after loading authorized assets and inputs locally:

```sh
# Set IMAGE to the full reference copied from the publishing job.
docker pull --platform linux/amd64 "$IMAGE"
docker run --rm --platform linux/amd64 --network none \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/assets:/assets:ro" -v "$PWD/input:/input:ro" -v "$PWD/output:/output" \
  "$IMAGE" run --model best --assets /assets --mode page \
  --input /input/page.png --output /output/page.txt
```

For Kraken, omit the `run` action. For the TrOCR candidate use `--mode line`.
Anyone pulling a private package needs GHCR read access. For pod use, the
approved compute job must be able to pull the image by digest and stage the
same assets. Keep its data transfer, budget and lifecycle on the project's
authorized autonomous pod path. Eval selects the applicable models and
freezes segmentation, order and scorer settings separately.

To build locally from a clean committed revision, use a directory created
from `git archive HEAD comparators docker/comparators/Dockerfile` as the
Docker build context:

```sh
docker buildx build --platform linux/amd64 --load --target tesseract \
  --file "$CONTEXT/docker/comparators/Dockerfile" --tag comparator:tesseract "$CONTEXT"
```

— roster[openai/codex#shay-macbook-m4]@heocr/opensource
