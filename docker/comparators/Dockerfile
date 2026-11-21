# Python 3.12.16 slim on Debian trixie. Build all targets for linux/amd64.
FROM docker.io/library/python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS runtime

ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /opt/comparators
RUN printf '%s\n' 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20261115T000000Z/ trixie main' \
    > /etc/apt/sources.list.d/debian.sources.list \
    && rm /etc/apt/sources.list.d/debian.sources

FROM runtime AS tesseract-build
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
COPY comparators/runner.py comparators/assets.json ./
ENTRYPOINT ["python", "/opt/comparators/runner.py"]

FROM adapter AS tesseract
RUN apt-get update && apt-get install -y --no-install-recommends libleptonica6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=tesseract-build /opt/tesseract/bin/tesseract /usr/local/bin/tesseract

FROM runtime AS paddle-build
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

FROM runtime AS kraken
COPY comparators/kraken_runner.py comparators/requirements-kraken-linux-amd64.txt ./
RUN python -m pip install --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-kraken-linux-amd64.txt && python -m pip check
ENTRYPOINT ["python", "/opt/comparators/kraken_runner.py"]

FROM adapter AS trocr
ENV HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
COPY comparators/trocr.py comparators/requirements-trocr-linux-amd64.txt ./
RUN python -m pip install --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-trocr-linux-amd64.txt && python -m pip check
