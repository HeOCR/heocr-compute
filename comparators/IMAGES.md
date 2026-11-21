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

The [successful publishing run](https://github.com/HeOCR/heocr-compute/actions/runs/44878288865) built and pulled back
these Linux AMD64 images from commit `86aa84d2b52a77e0077951d89e2f2667f78bf685`.

| Runner | Image reference |
|---|---|
| `tesseract` | `ghcr.io/heocr/heocr-compute-comparator-tesseract@sha256:701d58c9a17b1d937d09a89254f5002722a294d68c0b0c1d18269c608745a83b` |
| `paddle` | `ghcr.io/heocr/heocr-compute-comparator-paddle@sha256:6a5d87ca16e87f6f0fdbc05c15118a1cd2300f0c6421580e58eef80b56f410f3` |
| `kraken` | `ghcr.io/heocr/heocr-compute-comparator-kraken@sha256:86eedd8354462eb189b8cea5ec29d52a7957fc4eec18a8a91fba686456c9118c` |
| `trocr` | `ghcr.io/heocr/heocr-compute-comparator-trocr@sha256:31371c54617df5cff893c1c52ebbd039753acb92abe96dc2965eb0d546519b05` |

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
