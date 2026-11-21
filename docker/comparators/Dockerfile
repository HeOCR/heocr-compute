# Python 3.12.16 slim on Debian trixie. Build all targets for linux/amd64.
FROM docker.io/library/python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS base

ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /opt/comparators
RUN printf '%s\n' 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20261115T000000Z/ trixie main' \
    > /etc/apt/sources.list.d/debian.sources.list \
    && rm /etc/apt/sources.list.d/debian.sources

FROM base AS pod-tools
ADD --checksum=sha256:ae0a2e5ff723c8fceb609e7120dfcaa4d681bf3015adc74141812a21f2bb9486 \
    https://github.com/runpod/runpodctl/releases/download/v3.1.1/runpodctl-linux-amd64 /tmp/runpodctl
RUN install -m 0755 /tmp/runpodctl /usr/local/bin/runpodctl \
    && runpodctl pod delete --help >/dev/null && command -v timeout

COPY pyproject.toml /build/
COPY heocr_compute /build/heocr_compute
RUN python -m venv /opt/heocr-venv \
    && /opt/heocr-venv/bin/pip install /build 'huggingface_hub==0.36.2' \
    && /opt/heocr-venv/bin/pip check

FROM base AS pod-runtime
COPY --from=pod-tools /usr/local/bin/runpodctl /usr/local/bin/runpodctl
COPY --from=pod-tools /opt/heocr-venv /opt/heocr-venv
COPY docker/hf-train-entrypoint.sh /opt/heocr-hf-entrypoint.sh
COPY docker/comparators/NOTICE /usr/share/doc/heocr-comparator/NOTICE

FROM pod-runtime AS runtime
COPY docker/comparators/entrypoint.sh /opt/comparators/entrypoint.sh

FROM base AS tesseract-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libleptonica-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:9218e62793116d42a9f6d14cd9348518b27f382096eea3d0f2d1a24616bb5884 \
    https://codeload.github.com/tesseract-ocr/tesseract/tar.gz/refs/tags/5.5.3 /tmp/tesseract.tar.gz
RUN mkdir /tmp/tesseract && tar -xf /tmp/tesseract.tar.gz -C /tmp/tesseract --strip-components=1 \
    && cmake -S /tmp/tesseract -B /tmp/tesseract-build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/tesseract -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TRAINING_TOOLS=OFF -DDISABLE_CURL=ON -DDISABLE_ARCHIVE=ON \
        -DGRAPHICS_DISABLED=ON -DENABLE_NATIVE=OFF \
    && cmake --build /tmp/tesseract-build --parallel 2 \
    && cmake --install /tmp/tesseract-build --strip

FROM runtime AS adapter
COPY comparators/runner.py comparators/raw_output.py comparators/assets.json ./
ENTRYPOINT ["/bin/sh", "/opt/comparators/entrypoint.sh", "/opt/comparators/runner.py"]

FROM adapter AS tesseract
RUN apt-get update && apt-get install -y --no-install-recommends libleptonica6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=tesseract-build /opt/tesseract/bin/tesseract /usr/local/bin/tesseract

FROM base AS paddle-build
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY comparators/requirements-paddle.txt ./
RUN python -m venv /opt/paddle \
    && /opt/paddle/bin/python -m pip install setuptools==84.2.1 \
    && /opt/paddle/bin/python -m pip install --no-build-isolation -r requirements-paddle.txt \
    && /opt/paddle/bin/python -m pip check

FROM adapter AS paddle
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=paddle-build /opt/paddle /opt/paddle
ENV PATH="/opt/paddle/bin:$PATH"

FROM pod-runtime AS kraken-runtime
COPY docker/comparators/licenses/Intel-Simplified-Software-License.pdf /usr/share/doc/heocr-comparator/
COPY comparators/requirements-kraken-linux-amd64.txt ./
RUN python -m pip install --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-kraken-linux-amd64.txt && python -m pip check

FROM kraken-runtime AS kraken
LABEL org.opencontainers.image.description="Kraken comparator runtime. Contains non-open-source but redistributable components."
COPY docker/comparators/entrypoint.sh /opt/comparators/entrypoint.sh
COPY comparators/kraken_runner.py comparators/raw_output.py ./
ENTRYPOINT ["/bin/sh", "/opt/comparators/entrypoint.sh", "/opt/comparators/kraken_runner.py"]

FROM kraken-runtime AS kraken-train
LABEL org.opencontainers.image.description="Autonomous Kraken CPU training worker. Contains redistributable oneMKL."
ENV HEOCR_HF_REQUIRE_REPO_SCOPE=1
ENTRYPOINT ["/bin/sh", "/opt/heocr-hf-entrypoint.sh"]

FROM adapter AS trocr
LABEL org.opencontainers.image.description="TrOCR comparator runtime. Contains non-open-source but redistributable components."
COPY docker/comparators/licenses/Intel-Simplified-Software-License.pdf /usr/share/doc/heocr-comparator/
ENV TOKENIZERS_PARALLELISM=false
COPY comparators/trocr.py comparators/requirements-trocr-linux-amd64.txt ./
RUN python -m pip install --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-trocr-linux-amd64.txt && python -m pip check
