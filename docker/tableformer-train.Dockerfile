FROM ghcr.io/heocr/heocr-compute-train@sha256:78414e4554ad4d1c55574b66e4b3fa3012c0bbb89b1725592d44e5a18e737bc8

COPY experiments/tableformer/requirements-train-linux.txt /opt/tableformer-requirements.txt
RUN python3 -m venv /opt/heocr-tableformer \
    && /opt/heocr-tableformer/bin/python -m pip install \
         --no-cache-dir --only-binary=:all: --require-hashes \
         -r /opt/tableformer-requirements.txt \
    && /opt/heocr-tableformer/bin/python -m pip check \
    && /opt/heocr-tableformer/bin/python -c 'import sys, torch; from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor; assert sys.version_info[:2] == (3, 12); assert torch.version.cuda == "13.2"'
