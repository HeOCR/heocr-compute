#!/bin/sh
set -eu

if [ "${HEOCR_HF_JOB+x}" = x ]; then
    exec /bin/sh /opt/heocr-hf-entrypoint.sh
fi

runner="$1"
shift
exec python "$runner" "$@"
