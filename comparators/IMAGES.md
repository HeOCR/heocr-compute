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

The [successful publishing run](https://github.com/HeOCR/heocr-compute/actions/runs/44879291570) built and pulled back
these Linux AMD64 images from commit `924efa744d6d1b766965b74bfe3bb1f9f3c754d8`.
Use their [batch scorer transport](CONTRACT.md) for registered runs.

| Runner | Image reference |
|---|---|
| `tesseract` | `ghcr.io/heocr/heocr-compute-comparator-tesseract@sha256:a324f263d689dc3178b4bc842ac0225f4f4d6181807f98eafb99e7defefc19ea` |
| `paddle` | `ghcr.io/heocr/heocr-compute-comparator-paddle@sha256:027e5f5d65e95392b0113c38098a5fa0540abe5273fa34f505354b24c305fae8` |
| `kraken` | `ghcr.io/heocr/heocr-compute-comparator-kraken@sha256:e88a73be6857958df0b12785eaa419cc027b6a056ae68a3c75e271756afd28e5` |
| `trocr` | `ghcr.io/heocr/heocr-compute-comparator-trocr@sha256:8f0c77fbc41d29f254ac4e52e18d29456e65df877d50359d4cbedf22caefba7d` |

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
