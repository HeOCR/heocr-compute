#!/bin/bash
# Build the reviewed commit and exercise Kraken inside the Linux image offline.
set -euo pipefail

image="${1:?supply a local image tag}"
target="${2:-kraken-train}"
case "$target" in
  kraken-train) cuda="" ;;
  kraken-train-cuda) cuda="13.2" ;;
  *) echo "Unknown Kraken training target: $target" >&2; exit 2 ;;
esac
revision="$(git rev-parse HEAD)"
context="$(mktemp -d)"
trap 'rm -rf -- "$context"' EXIT
git archive "$revision" pyproject.toml heocr_compute docker/hf-train-entrypoint.sh \
  docker/comparators/Dockerfile docker/comparators/NOTICE \
  docker/comparators/licenses/Intel-Simplified-Software-License.pdf \
  comparators/requirements-kraken-linux-amd64.txt \
  experiments/tableformer/requirements-train-linux.txt | tar -x -C "$context"
docker buildx build --platform linux/amd64 --file "$context/docker/comparators/Dockerfile" \
  --target "$target" --load --provenance=false --sbom=false --tag "$image" \
  --label org.opencontainers.image.source=https://github.com/HeOCR/heocr-compute \
  --label "org.opencontainers.image.revision=$revision" "$context"
test "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")" = linux/amd64
test "$(docker image inspect --format '{{json .Config.Entrypoint}}' "$image")" = '["/bin/sh","/opt/heocr-hf-entrypoint.sh"]'
docker run --rm --network none --entrypoint /opt/heocr-venv/bin/python "$image" \
  -c 'import heocr_compute.remote.hf_train_pod, huggingface_hub'

docker run --rm -i --network none --env "EXPECTED_CUDA=$cuda" --entrypoint /bin/bash "$image" -s <<'BASH'
set -euo pipefail
cd "$(mktemp -d)"
test "$HEOCR_HF_REQUIRE_REPO_SCOPE" = 1
python - <<'PY'
from importlib.metadata import version
import os
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

assert version("kraken") == "7.2" and torch.version.cuda == (os.environ["EXPECTED_CUDA"] or None)
assert torchvision.extension._has_ops()
font = ImageFont.load_default(size=24)
for split, labels in (("train", ["AB 12", "BA 21"]), ("dev", ["A1 B2"])):
    files = []
    for i, label in enumerate(labels):
        path = Path.cwd() / f"{split}-{i}.png"
        image = Image.new("L", (200, 48), 255)
        ImageDraw.Draw(image).text((8, 8), label, font=font, fill=0)
        image.save(path)
        path.with_suffix(".gt.txt").write_text(label + "\n", encoding="utf-8")
        files.append(str(path))
    Path(f"{split}.lst").write_text("\n".join(files) + "\n", encoding="utf-8")
PY
ketos --device cpu --seed 7 --workers 0 --threads 2 train \
  --arch vgsl --format-type path \
  --spec '[1,48,0,1 Cr3,3,16 Mp2,2 Cr3,3,32 Mp2,2 S1(1x0)1,3 Lbx64 Lbx64]' \
  --training-data train.lst --evaluation-data dev.lst \
  --normalization NFC --normalize-whitespace --reorder --base-dir R \
  --batch-size 2 --quit fixed --epochs 1 --output checkpoints --weights-format safetensors > smoke.log 2>&1 || {
    echo 'Offline Kraken training smoke failed.' >&2
    sed -n '/Traceback (most recent call last):/,$p' smoke.log | tail -40 >&2
    exit 1
  }
python - <<'PY'
from pathlib import Path
from safetensors import safe_open

weights, = Path("checkpoints").glob("best_*.safetensors")
with safe_open(weights, framework="pt") as file:
    assert file.keys()
PY
printf '%s\n' 'Offline Kraken CPU training smoke passed.'
BASH
