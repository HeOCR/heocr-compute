ARG BASE_IMAGE
FROM $BASE_IMAGE

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && command -v runpodctl && command -v timeout

COPY pyproject.toml /build/
COPY heocr_compute /build/heocr_compute
RUN python3 -m venv /opt/heocr-venv \
    && /opt/heocr-venv/bin/pip install --no-cache-dir /build 'huggingface_hub==0.36.2'

COPY docker/hf-train-entrypoint.sh /opt/heocr-hf-entrypoint.sh
ENTRYPOINT ["/bin/sh", "/opt/heocr-hf-entrypoint.sh"]
