# Linux AMD64. Keep the published CPU comparator userspace and change only ORT and its provider runtime.
FROM docker.io/nvidia/cuda@sha256:14d94b039cb94bbd5da559f303b46bc4b0d5d6c24ab1a9d7b186e566ed3400dc AS cuda
FROM ghcr.io/heocr/heocr-compute-comparator-paddle@sha256:6f79d33371caefc4c22036946c80b1bff695cac5ce670690da99ba18eadd3574

COPY --from=cuda /usr/local/cuda-13.0/targets/x86_64-linux/lib/ /opt/paddle-cuda/lib/
COPY --from=cuda /usr/lib/x86_64-linux-gnu/libcudnn*.so* /opt/paddle-cuda/lib/
ENV LD_LIBRARY_PATH=/opt/paddle-cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_REQUIRE_CUDA="cuda>=13.0"

ADD --checksum=sha256:09637f3e38a1addc308dec5ef825a36eeb786e9615f45ec76aa29909df5cdd36 \
    https://files.pythonhosted.org/packages/ce/8b/c48e90baf9b9389c4c709dae202938e03a01c7ffc792888bafcf60e9a6b6/onnxruntime_gpu-1.32.0-cp312-cp312-manylinux_2_28_x86_64.whl \
    /tmp/onnxruntime_gpu-1.32.0-cp312-cp312-manylinux_2_28_x86_64.whl
RUN /opt/paddle/bin/python -m pip uninstall -y onnxruntime \
    && /opt/paddle/bin/python -m pip install --no-deps /tmp/onnxruntime_gpu-1.32.0-cp312-cp312-manylinux_2_28_x86_64.whl \
    && rm /tmp/onnxruntime_gpu-1.32.0-cp312-cp312-manylinux_2_28_x86_64.whl \
    && /opt/paddle/bin/python -c 'import onnxruntime as ort; assert ort.__version__ == "1.32.0" and "CUDAExecutionProvider" in ort.get_available_providers()'

COPY comparators/runner.py comparators/paddle_cuda.py /opt/comparators/
COPY heocr_compute /opt/heocr-code/heocr_compute
COPY docker/comparators/paddle-cuda-entrypoint.sh /opt/paddle-cuda-entrypoint.sh
ENV PYTHONPATH=/opt/heocr-code
ENTRYPOINT ["/bin/sh", "/opt/paddle-cuda-entrypoint.sh"]
