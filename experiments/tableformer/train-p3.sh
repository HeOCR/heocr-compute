#!/bin/bash
set -euo pipefail
umask 077
cd /heocr
test "$#" -eq 3
manifest=$1
manifest_sha=$2
clean_sha=$3
python=/opt/heocr-tableformer/bin/python
weights=tableformer_accurate.safetensors
stem=${weights%.*}
mkdir -p work
log=work/trainer.log
publish_failure() {
    if [ -s "$log" ]; then
        mv "$log" out/partial.trainer.log
    fi
}
trap publish_failure EXIT
{
    "$python" -c 'import torch; assert torch.cuda.is_available(); torch.empty(1, device="cuda")'
    "$python" in/transport.py unpack --archive in/train.tar --out work/train \
        --manifest-name "$manifest" --manifest-sha256 "$manifest_sha" --clean-sha256 "$clean_sha"
    "$python" -B in/train.py --recipe p3 --root work/train --manifest "work/train/$manifest" \
        --manifest-sha256 "$manifest_sha" --clean work/train/clean.json --clean-sha256 "$clean_sha" \
        --models in/models --device cuda --seed 32 --epochs 3 --lr 1e-5 --output work/candidate
} > "$log" 2>&1
mv work/candidate/receipt.json "out/$stem.training-metrics.json"
mv "$log" "out/$stem.trainer.log"
mv "work/candidate/$weights" "out/$weights"
trap - EXIT
