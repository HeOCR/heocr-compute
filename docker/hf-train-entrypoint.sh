#!/bin/sh
set -eu
umask 077

case "$HEOCR_HF_DEADLINE" in
  ''|*[!0-9]*) echo 'Missing valid pod deadline' >&2; exit 1 ;;
esac
if [ -z "$RUNPOD_POD_ID" ] || [ -z "$RUNPOD_API_KEY" ]; then
    echo 'Missing pod-scoped termination credentials' >&2
    exit 1
fi

terminate_pod() {
    # Provider stderr can contain environment settings. Keep it out of the logs.
    for delay in 1 2 4 8 15; do
        if timeout -k 2s 10s runpodctl pod delete "$RUNPOD_POD_ID" >/dev/null 2>&1 ||
           timeout -k 2s 10s runpodctl remove pod "$RUNPOD_POD_ID" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$delay"
    done
    echo 'Pod could not confirm self-termination' >&2
    return 1
}

remaining=$((HEOCR_HF_DEADLINE - $(date +%s)))
if [ "$remaining" -le 0 ]; then
    terminate_pod
    exit 1
fi

# Arm the independent deadline guard before any Hub or trainer operation.
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
worker_seconds=$((remaining - 10))
[ "$worker_seconds" -gt 0 ] || exit 1
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
timeout -s TERM -k 5s "$worker_seconds" /opt/heocr-venv/bin/python -m heocr_compute.remote.hf_train_pod
