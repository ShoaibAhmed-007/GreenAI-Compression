#!/usr/bin/env bash
set -euo pipefail

# Run this inside your RunPod container after activating the Python environment.
# Example:
#   cd /workspace/GreenAI-Compression/backend
#   chmod +x setup_tensorrt_runpod.sh
#   ./setup_tensorrt_runpod.sh

export PYTHONUNBUFFERED=1
TRT_VER="10.15.1.29"
TORCH_TRT_VER="2.11.0+cu128"

echo "[1/6] Inspect runtime"
python - <<'PY'
import torch
import platform
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

echo "[2/6] Upgrade packaging tools"
python -m pip install --upgrade pip setuptools wheel packaging

echo "[3/6] Install TensorRT and Torch-TensorRT (pinned compatible versions)"
python -m pip uninstall -y \
    torch-tensorrt \
    tensorrt \
    tensorrt-cu12 \
    tensorrt-cu12-bindings \
    tensorrt-cu12-libs \
    tensorrt-cu13 \
    tensorrt-cu13-bindings \
    tensorrt-cu13-libs >/dev/null 2>&1 || true

python -m pip install --upgrade --extra-index-url https://pypi.nvidia.com \
    "tensorrt-cu12==${TRT_VER}" \
    "tensorrt-cu12-bindings==${TRT_VER}" \
    "tensorrt-cu12-libs==${TRT_VER}" \
    "torch-tensorrt==${TORCH_TRT_VER}"

echo "[4/6] Install modelopt (required for TensorRT INT8)"
python -m pip install --upgrade --extra-index-url https://pypi.nvidia.com nvidia-modelopt

echo "[5/6] Export runtime library paths for torch_tensorrt"
export LD_LIBRARY_PATH="$(python - <<'PY'
import os
import site

paths = []
for root in site.getsitepackages():
    for rel in (
        "tensorrt_libs",
        os.path.join("tensorrt_libs", "lib"),
        os.path.join("torch_tensorrt", "lib"),
        os.path.join("nvidia", "cudnn", "lib"),
        os.path.join("nvidia", "cublas", "lib"),
        os.path.join("nvidia", "cuda_runtime", "lib"),
        os.path.join("nvidia", "cusolver", "lib"),
        os.path.join("nvidia", "cusparse", "lib"),
        os.path.join("nvidia", "cufft", "lib"),
        os.path.join("nvidia", "curand", "lib"),
        os.path.join("nvidia", "nvjitlink", "lib"),
        os.path.join("nvidia", "nvtx", "lib"),
    ):
        candidate = os.path.join(root, rel)
        if os.path.isdir(candidate):
            paths.append(candidate)

seen = set()
ordered_paths = []
for path in paths:
    if path not in seen:
        seen.add(path)
        ordered_paths.append(path)

print(":".join(ordered_paths))
PY
):${LD_LIBRARY_PATH:-}"

echo "[6/6] Verify TensorRT compile (INT8 -> FP16 -> FP32 fallback)"
python - <<'PY'
import importlib.util
import time
import torch
import ctypes
import glob
import os
import site

matches = []
for root in site.getsitepackages():
    matches.extend(glob.glob(os.path.join(root, "torch_tensorrt", "lib", "libtorchtrt.so")))

if not matches:
    raise SystemExit("Could not find libtorchtrt.so under site-packages.")

try:
    ctypes.CDLL(matches[0])
except OSError as exc:
    raise SystemExit(f"libtorchtrt.so could not be loaded ({matches[0]}): {exc}")

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
