# Arm ATT (attention line recognizer): PaddleOCR SVTRv2 + NRTR trainer for Recipe A.
# The base image provides the autonomous HF training supervisor and entrypoint. The job's own
# trainer script, recovered init weights and data arrive as pinned job inputs.
FROM ghcr.io/heocr/heocr-compute-train@sha256:78414e4554ad4d1c55574b66e4b3fa3012c0bbb89b1725592d44e5a18e737bc8

COPY experiments/att/requirements-paddle-linux.txt /opt/att-paddle-requirements.txt
# PaddleOCR v3.7.0 at b03f46425e8ff4442b268ce449e3eef758146cd4 (Apache-2.0; LICENSE kept in /opt/paddleocr).
ADD --checksum=sha256:af62d37268bc342e45d1f08dcf90fc418637cf8fb335d9db2b38f4e84c594d91 \
    https://github.com/PaddlePaddle/PaddleOCR/archive/b03f46425e8ff4442b268ce449e3eef758146cd4.tar.gz /opt/paddleocr.tar.gz
RUN python3 -m venv /opt/heocr-att-paddle \
    && /opt/heocr-att-paddle/bin/python -m pip install \
         --no-cache-dir --only-binary=:all: --require-hashes \
         --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
         -r /opt/att-paddle-requirements.txt \
    && /opt/heocr-att-paddle/bin/python -m pip check \
    && mkdir /opt/paddleocr && tar -xzf /opt/paddleocr.tar.gz -C /opt/paddleocr --strip-components=1 \
         PaddleOCR-b03f46425e8ff4442b268ce449e3eef758146cd4/ppocr PaddleOCR-b03f46425e8ff4442b268ce449e3eef758146cd4/configs \
         PaddleOCR-b03f46425e8ff4442b268ce449e3eef758146cd4/tools PaddleOCR-b03f46425e8ff4442b268ce449e3eef758146cd4/LICENSE \
    && rm /opt/paddleocr.tar.gz && test -s /opt/paddleocr/LICENSE \
    && /opt/heocr-att-paddle/bin/python -c 'import sys, pathlib, setuptools; from importlib.metadata import version; assert sys.version_info[:2] == (3, 12); assert version("paddlepaddle-gpu") == "3.2.2"; assert pathlib.Path("/opt/paddleocr/ppocr/modeling/architectures/__init__.py").is_file()'
# paddle-gpu 3.x needs the driver's libcuda.so.1 at import, which only a GPU host provides. The import, CUDA device and
# SVTRv2 build checks therefore run in the pod's start-up preflight (boot.sh) before any work.
ENV HEOCR_ATT_PYTHON=/opt/heocr-att-paddle/bin/python HEOCR_ATT_PADDLEOCR=/opt/paddleocr
