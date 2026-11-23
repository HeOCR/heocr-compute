#!/bin/sh
set -eu
umask 077

if [ "${HEOCR_PADDLE_JOB+x}" != x ]; then
    exec /opt/paddle/bin/python -B /opt/comparators/paddle_cuda.py "$@"
fi

case "${HEOCR_HF_DEADLINE:-}" in
    ''|*[!0-9]*) echo 'Missing valid Paddle pod deadline' >&2; exit 1 ;;
esac
if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo 'Missing pod-scoped termination credentials' >&2
    exit 1
fi

terminate_pod() {
    for delay in 1 2 4 8 15; do
        if timeout -k 2s 10s runpodctl pod delete "$RUNPOD_POD_ID" >/dev/null 2>&1 ||
           timeout -k 2s 10s runpodctl remove pod "$RUNPOD_POD_ID" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$delay"
    done
    echo 'Paddle pod teardown remains unconfirmed' >&2
    return 1
}

remaining=$((HEOCR_HF_DEADLINE - $(date +%s)))
if [ "$remaining" -le 10 ]; then
    terminate_pod
    exit 1
fi
(sleep "$remaining"; terminate_pod) </dev/null >/dev/null 2>&1 &
watchdog=$!
finish() {
    if terminate_pod; then
        kill "$watchdog" 2>/dev/null || true
    else
        wait "$watchdog"
    fi
}
trap finish EXIT
trap 'exit 1' INT TERM
timeout -s TERM -k 5s "$((remaining - 10))" \
    /opt/heocr-venv/bin/python -B -m heocr_compute.remote.paddle_eval --worker
