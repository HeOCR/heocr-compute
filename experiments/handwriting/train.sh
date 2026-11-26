#!/bin/bash
set -euo pipefail
umask 077
export PYTHONUTF8=1 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for arg in "$@"; do
    if [ "$arg" = "--local" ]; then
        echo 'Local-only TRAIN data cannot use the cloud entrypoint.' >&2
        exit 2
    fi
done
exec python in/train.py "$@"
