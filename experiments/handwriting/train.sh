#!/bin/bash
set -euo pipefail
umask 077
export PYTHONUTF8=1 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python in/train.py
