#!/usr/bin/env bash
set -euo pipefail

# Run this inside your RunPod container after activating the Python environment.
# Example:
#   cd /workspace/GreenAI-Compression/backend
#   chmod +x setup_tensorrt_runpod.sh
#   ./setup_tensorrt_runpod.sh

export PYTHONUNBUFFERED=1

echo "[1/5] Inspect runtime"
python - <<'PY'
import torch
import platform
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

echo "[2/5] Upgrade packaging tools"
python -m pip install --upgrade pip setuptools wheel packaging

echo "[3/5] Install TensorRT and Torch-TensorRT"
python -m pip install --upgrade --extra-index-url https://pypi.nvidia.com \
  tensorrt-cu12 \
  tensorrt-cu12-bindings \
  tensorrt-cu12-libs \
  torch-tensorrt

echo "[4/5] Install modelopt (required for TensorRT INT8)"
python -m pip install --upgrade --extra-index-url https://pypi.nvidia.com nvidia-modelopt

echo "[5/5] Verify TensorRT compile (INT8 -> FP16 -> FP32 fallback)"
python - <<'PY'
import importlib.util
import time
import torch

try:
    import torch_tensorrt
except Exception as exc:
    raise SystemExit(f"torch_tensorrt import failed: {exc}")

has_modelopt = importlib.util.find_spec("modelopt") is not None
print(f"modelopt installed: {has_modelopt}")

class TinyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(16, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this container.")

device = torch.device("cuda")
model = TinyCNN().eval().to(device)
x = torch.randn(1, 3, 224, 224, device=device)

compile_modes = []
if has_modelopt:
    compile_modes.append(("int8", {torch.int8}))
else:
    print("Skipping INT8 check because modelopt is not installed.")
compile_modes.append(("fp16", {torch.float16}))
compile_modes.append(("fp32", {torch.float32}))

compiled = None
compiled_mode = None
for mode, precisions in compile_modes:
    try:
        compiled = torch_tensorrt.compile(
            model,
            ir="torch_compile",
            inputs=[torch_tensorrt.Input(shape=(1, 3, 224, 224), dtype=torch.float32)],
            enabled_precisions=precisions,
        )
        compiled_mode = mode
        print(f"TensorRT compile success in mode={mode}")
        break
    except Exception as exc:
        print(f"TensorRT compile failed in mode={mode}: {type(exc).__name__}: {exc}")

if compiled is None:
    raise SystemExit("TensorRT verification failed for all precision modes.")

with torch.no_grad():
    _ = compiled(x)

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = compiled(x)
torch.cuda.synchronize()
lat_ms = (time.time() - start) / 100.0 * 1000.0
print(f"TensorRT runtime mode: {compiled_mode}")
print(f"TensorRT micro-benchmark latency: {lat_ms:.3f} ms")
print("TensorRT verification complete.")
PY

echo "Done. You can rerun your compression command now."
