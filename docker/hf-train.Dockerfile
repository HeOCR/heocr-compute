ARG BASE_IMAGE=docker.io/runpod/base@sha256:e93bf61149910d35407fc865078ceb9b2887e986befd2b7c0eee322abe47ac59
FROM $BASE_IMAGE

ADD --checksum=sha256:ae0a2e5ff723c8fceb609e7120dfcaa4d681bf3015adc74141812a21f2bb9486 https://github.com/runpod/runpodctl/releases/download/v3.1.1/runpodctl-linux-amd64 /tmp/runpodctl
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && install -m 0755 /tmp/runpodctl /usr/local/bin/runpodctl \
    && rm /tmp/runpodctl \
    && runpodctl pod delete --help >/dev/null && command -v timeout

COPY pyproject.toml /build/
COPY heocr_compute /build/heocr_compute
RUN python3 -m venv /opt/heocr-venv \
    && /opt/heocr-venv/bin/pip install --no-cache-dir /build 'huggingface_hub==0.36.2'

COPY docker/hf-train-entrypoint.sh /opt/heocr-hf-entrypoint.sh
ENTRYPOINT ["/bin/sh", "/opt/heocr-hf-entrypoint.sh"]
