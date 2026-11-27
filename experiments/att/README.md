# Arm ATT training workers

Two CUDA training images for arm ATT, the attention-decoder line recognizer. The plan is in the private `heocr-agent-wiki` at `records/attention-recognizer/`. Both images extend the pinned `heocr-compute-train` base, so pods run the existing autonomous `hf_train_pod` supervisor: pinned private Hub inputs, snapshots pushed to private HF, self-termination at the deadline. The trainer script, init weights and data arrive as pinned job inputs. These images only fix the runtime.

| Target | Venv | Source tree | CUDA |
| --- | --- | --- | --- |
| `att-parseq-train-cuda` | `/opt/heocr-att-parseq` from [requirements-parseq-linux.txt](requirements-parseq-linux.txt) (torch 2.15.0, timm 1.0.31) | baudm/parseq `1902db04` (Apache-2.0) at `/opt/parseq` | 13.2 (driver ≥ 580) |
| `att-paddle-train-cuda` | `/opt/heocr-att-paddle` from [requirements-paddle-linux.txt](requirements-paddle-linux.txt) (paddlepaddle-gpu 3.2.2 cu126) | PaddleOCR v3.7.0 `b03f4642` (Apache-2.0): `ppocr`, `configs`, `tools` at `/opt/paddleocr` | 12.6 (driver ≥ 560) |

The Paddle install also needs Paddle's cu126 wheel index (`paddlepaddle-gpu` isn't on PyPI); the hashes still pin the exact wheels. Both requirement files are hash-locked for Linux x86_64 / CPython 3.12. Regenerate them from the `.in` files:

```sh
uv pip compile requirements-parseq.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header -o requirements-parseq-linux.txt
uv pip compile requirements-paddle.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header \
  --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --index-strategy unsafe-best-match -o requirements-paddle-linux.txt
```

Source tarballs are pinned with `ADD --checksum`. Each tarball was hashed twice on 2026-11-27 with identical results. To publish, dispatch [att-train-images.yml](../../.github/workflows/att-train-images.yml) on `main` once per target. The workflow builds, runs an offline CPU smoke (a PARSeq forward pass, or an SVTRv2 MultiHead forward pass), pushes, and checks an anonymous pull. Use the digest it prints in job records.
