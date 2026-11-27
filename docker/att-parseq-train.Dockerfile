# Arm ATT (attention line recognizer): PARSeq trainer for Recipe B and control C.
# The base image provides the autonomous HF training supervisor and entrypoint. The job's own
# trainer script and data arrive as pinned job inputs; this image only fixes the runtime.
FROM ghcr.io/heocr/heocr-compute-train@sha256:78414e4554ad4d1c55574b66e4b3fa3012c0bbb89b1725592d44e5a18e737bc8

COPY experiments/att/requirements-parseq-linux.txt /opt/att-parseq-requirements.txt
# baudm/parseq at 1902db043c029a7e03a3818c616c06600af574be (Apache-2.0; LICENSE and NOTICE kept in /opt/parseq).
ADD --checksum=sha256:f57289cb1fc7fe41f488a52ea48f8a7f114252ef405b0b64efc1d528c29b48ef \
    https://github.com/baudm/parseq/archive/1902db043c029a7e03a3818c616c06600af574be.tar.gz /opt/parseq.tar.gz
RUN python3 -m venv /opt/heocr-att-parseq \
    && /opt/heocr-att-parseq/bin/python -m pip install \
         --no-cache-dir --only-binary=:all: --require-hashes \
         -r /opt/att-parseq-requirements.txt \
    && /opt/heocr-att-parseq/bin/python -m pip check \
    && mkdir /opt/parseq && tar -xzf /opt/parseq.tar.gz -C /opt/parseq --strip-components=1 && rm /opt/parseq.tar.gz \
    && test -s /opt/parseq/LICENSE && test -s /opt/parseq/NOTICE \
    && PYTHONPATH=/opt/parseq /opt/heocr-att-parseq/bin/python -c 'import sys, torch, timm; from strhub.models.parseq.model import PARSeq; assert sys.version_info[:2] == (3, 12); assert torch.version.cuda == "13.2"'
ENV HEOCR_ATT_PYTHON=/opt/heocr-att-parseq/bin/python HEOCR_ATT_PARSEQ=/opt/parseq
