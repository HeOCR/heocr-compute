#!/bin/bash
set -euo pipefail
umask 077
cd /heocr
test "$#" -eq 2
case "$1" in
    paddle|tatr) ;;
    *) exit 2 ;;
esac
test "$(cat /opt/table-comparator/lane)" = "$1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
mkdir -p work
tar -xf in/dev-pages.tar -C work --no-same-owner --no-same-permissions
/opt/tables/bin/python /opt/table-comparator/worker.py \
    --lane "$1" --manifest work/pages.json --manifest-sha256 "$2" \
    --models /opt/models --tessdata /opt/tessdata \
    --output out/rows.jsonl --issues out/issues.json
