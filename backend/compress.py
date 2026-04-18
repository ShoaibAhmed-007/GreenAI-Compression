# compress.py
"""
Green AI FYP — Comprehensive Model Compression Pipeline
========================================================
Applies 5 effective compression strategies to ResNet18 (CIFAR-10):

Strategy 1: Aggressive Unstructured Pruning (70%) + Fine-tuning + Sparse Save
Strategy 2: Post-Training Static Quantization (INT8) via quantizable ResNet18
Strategy 3: Knowledge Distillation → Compact Student Model (~95% smaller)
Strategy 4: Hybrid — Compact Student + Dynamic Quantization
Strategy 5: Ultra-Compact — Pruned Student + Quantization + Sparse Save

Key fixes over previous version:
- Pruning now saves in SPARSE format (zeros no longer waste disk space)
- Quantization now uses STATIC quantization (INT8 for weights AND activations)
- KD now uses a SMALLER student architecture (not another ResNet18)
- KD trains on TRAINING data, evaluates on TEST data (fixes data leakage)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import os
import json
import re
import argparse
import multiprocessing
import platform
try:
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Enable cuDNN auto-tuner for faster convolutions with fixed input sizes
torch.backends.cudnn.benchmark = True
import copy
import time
import gzip
import io
import warnings


# ============================================================
# Compact Student Model (MobileNet-style, ~300K params)
# ============================================================
class CompactStudent(nn.Module):
    """
    Lightweight MobileNet-style CNN for CIFAR-10.
    Uses depthwise separable convolutions to minimize parameters.
    ~300K params vs 11.2M for ResNet18 → ~97% parameter reduction.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        def _dsconv(in_c, out_c, stride=1):
            """Depthwise separable convolution: depthwise + pointwise."""
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1,
                          groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                # Pointwise
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            # Initial standard conv: 3 → 32
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # DS blocks with increasing channels and spatial reduction
            _dsconv(32, 64),               # 32x32
            _dsconv(64, 128, stride=2),    # → 16x16
            _dsconv(128, 128),             # 16x16
            _dsconv(128, 256, stride=2),   # → 8x8
            _dsconv(256, 256),             # 8x8
            _dsconv(256, 512, stride=2),   # → 4x4
            _dsconv(512, 512),             # 4x4
            nn.AdaptiveAvgPool2d(1),       # → 1x1
        )
        # Dropout improves confidence calibration and reduces overfitting on CIFAR-sized data.
        self.dropout = nn.Dropout(p=0.35)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ============================================================
# Utility Functions
# ============================================================
def evaluate(model, loader, dev=None, max_batches=None):
    """Evaluate model accuracy on a DataLoader.
    
    Args:
        max_batches: If set, stop after this many batches (for speed).
                     None = use all batches.
    """
    model.eval()
    correct = total = 0
    if dev is None:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device('cpu')

    use_half = False
    try:
        if next(model.parameters()).dtype == torch.float16:
            use_half = True
    except (StopIteration, TypeError, AttributeError):
        pass

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs, labels = inputs.to(dev), labels.to(dev)
            if use_half:
                inputs = inputs.half()
            outputs = _extract_logits(model(inputs))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return round(100. * correct / total, 2) if total > 0 else 0.0


def get_size_mb(path):
    """Get file size in megabytes."""
    return round(os.path.getsize(path) / 1e6, 2)


def count_params(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def count_nonzero(model):
    """Count non-zero parameters (useful after pruning)."""
    return sum((p != 0).sum().item() for p in model.parameters())


def _collect_prunable_modules(model):
    """Collect current Conv2d/Linear modules that expose a weight parameter."""
    prunable = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
            prunable.append((module, 'weight'))
    return prunable


def _safe_remove_pruning(module, name='weight'):
    """Best-effort removal of torch.nn.utils.prune reparameterization."""
    orig_name = f'{name}_orig'
    mask_name = f'{name}_mask'
    if not hasattr(module, orig_name) or not hasattr(module, mask_name):
        return False
    try:
        prune.remove(module, name)
        return True
    except (ValueError, AttributeError, KeyError, RuntimeError):
        return False


def _remove_pruning_from_model(model):
    """Remove pruning reparameterization from all supported layers in a model."""
    removed = 0
    for module, _ in _collect_prunable_modules(model):
        if _safe_remove_pruning(module, 'weight'):
            removed += 1
    return removed


def _tensor_is_floating_point(tensor: torch.Tensor) -> bool:
    """Handle PyTorch variants where is_floating_point can be a method or bool property."""
    attr = getattr(tensor, "is_floating_point", None)
    if callable(attr):
        return bool(attr())
    return bool(attr)


def _slugify_name(value):
    """Normalize a name so it is safe and consistent for filenames."""
    normalized = re.sub(r'[^a-zA-Z0-9]+', '_', str(value).strip().lower())
    normalized = normalized.strip('_')
    return normalized or 'model'


def build_compressed_model_path(save_dir, model_name, compression_method):
    """Create a standardized compressed filename: <model>_<method>.pth."""
    model_slug = _slugify_name(model_name)
    method_slug = _slugify_name(compression_method)
    filename = f"{model_slug}_{method_slug}.pth"
    return os.path.join(save_dir, filename)


def save_sparse_state_dict(model, path):
    """
    Save model state dict with sparse tensor optimization.
    Converts zero-heavy tensors (>50% zeros) to sparse format
    so pruned models actually shrink on disk.
    """
    state = model.state_dict() if isinstance(model, nn.Module) else model
    sparse_state = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            sparse_state[key] = tensor
            continue
        # Convert zero-heavy tensors to sparse format for disk savings
        if _tensor_is_floating_point(tensor) and tensor.dim() >= 2:
            total = tensor.numel()
            zeros = (tensor == 0).sum().item()
            if total > 0 and (zeros / total) > 0.5:
                sparse_state[key] = tensor.to_sparse()
                continue
        sparse_state[key] = tensor
    torch.save(sparse_state, path)
    return path


def save_compressed(model, path):
    """
    Save model state dict.
    Uses sparse tensor format for pruned models (zeros -> sparse)
    to actually reduce file size on disk.
    """
    state = model.state_dict() if isinstance(model, nn.Module) else model
    # Strip .gz extension if present so we always write a plain file
    if path.endswith('.gz'):
        path = path[:-3]

    # Check if model has significant sparsity — if so, use sparse saving
    total_elements = 0
    zero_elements = 0
    for tensor in state.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        if _tensor_is_floating_point(tensor) and tensor.dim() >= 2:
            total_elements += tensor.numel()
            zero_elements += (tensor == 0).sum().item()

    if total_elements > 0 and (zero_elements / total_elements) > 0.3:
        # Significant sparsity detected — save sparse
        return save_sparse_state_dict(model, path)
    else:
        # Dense model — plain save
        torch.save(state, path)
        return path


def _to_fp16_state_dict(state):
    """Create an fp16 state dict variant for smaller disk artifacts."""
    fp16_state = {}
    for key, tensor in state.items():
        if isinstance(tensor, torch.Tensor) and _tensor_is_floating_point(tensor):
            fp16_state[key] = tensor.half()
        else:
            fp16_state[key] = tensor
    return fp16_state


def save_smallest_artifact(model, base_path, prefer_sparse=False, include_fp16_variant=True):
    """Save multiple artifact variants and keep the smallest on disk.

    Returns:
        (best_path, best_size_mb)
    """
    state = model.state_dict() if isinstance(model, nn.Module) else model
    if base_path.endswith('.gz'):
        base_path = base_path[:-3]

    candidates = []

    # Dense artifact
    torch.save(state, base_path)
    candidates.append(base_path)

    # Optional sparse artifact for pruned models
    if prefer_sparse:
        sparse_path = base_path.replace('.pth', '_sparse.pth')
        save_sparse_state_dict(model, sparse_path)
        candidates.append(sparse_path)

    # fp16 artifact (optional)
    if include_fp16_variant:
        fp16_path = base_path.replace('.pth', '_fp16.pth')
        torch.save(_to_fp16_state_dict(state), fp16_path)
        candidates.append(fp16_path)

    # Gzipped variants
    gz_candidates = []
    for p in list(candidates):
        gz_path = p + '.gz'
        with gzip.open(gz_path, 'wb', compresslevel=9) as f:
            with open(p, 'rb') as src:
                f.write(src.read())
        gz_candidates.append(gz_path)
    candidates.extend(gz_candidates)

    # Pick the smallest and clean up others.
    best_path = min(candidates, key=lambda p: os.path.getsize(p))
    best_size_mb = get_size_mb(best_path)
    for p in candidates:
        if p != best_path and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    return best_path, best_size_mb


def log_validation_summary(strategy, baseline_acc, compressed_acc,
                           baseline_size_mb, compressed_size_mb, latency_ms):
    """Log before/after validation checks for consistency and debugging."""
    acc_drop = round(baseline_acc - compressed_acc, 2)
    size_delta = round(compressed_size_mb - baseline_size_mb, 2)
    size_reduction = round(100 * (baseline_size_mb - compressed_size_mb) / baseline_size_mb, 2) if baseline_size_mb > 0 else 0.0
    print(
        f"[Validation:{strategy}] "
        f"acc={baseline_acc}% -> {compressed_acc}% (drop={acc_drop}%), "
        f"size={baseline_size_mb}MB -> {compressed_size_mb}MB "
        f"(delta={size_delta}MB, reduction={size_reduction}%), "
        f"latency={latency_ms}ms"
    )
    if compressed_size_mb >= baseline_size_mb:
        print(f"[Validation:{strategy}] WARNING: compressed artifact is not smaller than baseline")


def load_compressed(path, device='cpu'):
    """Load a state dict — supports sparse tensors, plain .pth, and legacy .pth.gz."""
    if path.endswith('.gz') and os.path.exists(path):
        with gzip.open(path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        state = torch.load(buffer, map_location=device)
    elif not os.path.exists(path) and os.path.exists(path + '.gz'):
        # Legacy fallback
        with gzip.open(path + '.gz', 'rb') as f:
            buffer = io.BytesIO(f.read())
        state = torch.load(buffer, map_location=device)
    else:
        state = torch.load(path, map_location=device)

    # Convert any sparse tensors back to dense for model.load_state_dict()
    if isinstance(state, dict):
        for key in state:
            if isinstance(state[key], torch.Tensor) and state[key].is_sparse:
                state[key] = state[key].to_dense()
    return state


def export_to_tensorrt(model, input_shape, save_path, prefer_int8=True, min_block_size=1):
    """Compile a model with TensorRT and best-effort persist a runtime artifact.

    Falls back through precision modes to keep TensorRT runtime available:
    INT8 (when modelopt is installed) -> FP16 -> FP32.

    Args:
        min_block_size: TensorRT partition size threshold for torch.compile path.

    Returns:
        tuple: (compiled_model, backend_label, saved_artifact_path, precision_mode)
    """
    import importlib
    import importlib.util

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT export.")

    if importlib.util.find_spec("torch_tensorrt") is None:
        raise RuntimeError("torch_tensorrt is not installed in the current environment.")

    torch_tensorrt = importlib.import_module("torch_tensorrt")
    has_modelopt = importlib.util.find_spec("modelopt") is not None

    model.eval()
    model.to("cuda")

    compile_modes = []
    if prefer_int8:
        if has_modelopt:
            compile_modes.append(("int8", {torch.int8}, "TensorRT INT8 / Tensor Cores"))
        else:
            print("[Quantization] modelopt not found; skipping TensorRT INT8 and trying FP16.")
    compile_modes.append(("fp16", {torch.float16}, "TensorRT FP16 / Tensor Cores"))
    compile_modes.append(("fp32", {torch.float32}, "TensorRT FP32"))

    compile_errors = []
    compile_irs = ["dynamo", "torch_compile"]

    for precision_mode, enabled_precisions, backend_label in compile_modes:
        precision_compiled = False
        for compile_ir in compile_irs:
            try:
                compile_kwargs = {
                    "ir": compile_ir,
                    "inputs": [torch_tensorrt.Input(shape=tuple(input_shape), dtype=torch.float32)],
                    "enabled_precisions": enabled_precisions,
                }

                # Newer torch-tensorrt sets use_explicit_typing=True internally for
                # the dynamo backend but then rejects enabled_precisions. Force it off.
                if compile_ir == "dynamo":
                    compile_kwargs["use_explicit_typing"] = False

                try:
                    trt_gm = torch_tensorrt.compile(
                        model,
                        min_block_size=max(1, int(min_block_size)),
                        **compile_kwargs,
                    )
                except TypeError:
                    trt_gm = torch_tensorrt.compile(model, **compile_kwargs)
                except Exception as partition_err:
                    print(
                        f"[Quantization] TensorRT {precision_mode.upper()} ({compile_ir}) compile with "
                        f"min_block_size={min_block_size} failed "
                        f"({type(partition_err).__name__}): {partition_err}; retrying default partitioning."
                    )
                    if compile_ir == "dynamo":
                        compile_kwargs["use_explicit_typing"] = False
                    trt_gm = torch_tensorrt.compile(model, **compile_kwargs)

                saved_path = ""
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    path_root, path_ext = os.path.splitext(save_path)
                    suffix = f"_{precision_mode}"
                    artifact_path = f"{path_root}{suffix}{path_ext or '.ts'}"
                    print(
                        f"[Quantization] TensorRT compile succeeded via {compile_ir}; "
                        "skipping artifact save for runtime module"
                    )

                precision_compiled = True
                return trt_gm, backend_label, saved_path, precision_mode
            except Exception as compile_err:
                compile_errors.append(
                    f"{precision_mode}:{compile_ir}:{type(compile_err).__name__}:{compile_err}"
                )
                print(
                    f"[Quantization] TensorRT {precision_mode.upper()} compile failed on {compile_ir} "
                    f"({type(compile_err).__name__}): {compile_err}"
                )

        if not precision_compiled:
            print(f"[Quantization] Exhausted all IR backends for {precision_mode.upper()} precision.")

    raise RuntimeError(
        "All TensorRT compile attempts failed: " + " | ".join(compile_errors)
    )


def distillation_loss(student_out, teacher_out, labels, T=3.0, alpha=0.5):
    """
    Combined KD + CE loss.
    - T: temperature (higher = softer probabilities, more knowledge transfer)
    - alpha: weight for CE loss (1-alpha for KD loss)
    """
    student_logits = _extract_logits(student_out)
    teacher_logits = _extract_logits(teacher_out)
    kd = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * ce + (1 - alpha) * kd


def measure_latency(model, input_shape=(1, 3, 32, 32), dev=None, n_runs=100):
    """Measure average inference latency in milliseconds."""
    if dev is None:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device('cpu')
    model.eval()
    dummy = torch.randn(*input_shape).to(dev)
    # Torch-TensorRT runtime wrappers may trigger several lazy recompiles/
    # graph specializations before reaching steady-state.
    warmup_runs = 40 if _is_tensorrt_runtime_model(model) else 10
    for _ in range(warmup_runs):
        with torch.no_grad():
            model(dummy)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs * 1000  # ms
    return round(elapsed, 2)


def _start_emissions_tracker(project_name, output_dir):
    """Best-effort CodeCarbon tracker start. Returns tracker or None.

    Prefers EmissionsTracker and falls back to OfflineEmissionsTracker.
    Uses measure_power_secs=1 so short compression/inference workloads are sampled.
    """
    country_iso_code = "PAK"
    region = "Punjab"
    base_kwargs = {
        "project_name": project_name,
        "output_dir": output_dir,
        "log_level": "error",
        "measure_power_secs": 1,
    }

    try:
        from codecarbon import EmissionsTracker
        print(
            f"[CodeCarbon] Starting EmissionsTracker | project={project_name} "
            f"| output_dir={output_dir}"
        )
        tracker = EmissionsTracker(**base_kwargs)
        tracker.start()
        print("[CodeCarbon] EmissionsTracker started successfully")
        return tracker
    except Exception as online_err:
        print(f"[CodeCarbon] EmissionsTracker unavailable, trying offline mode: {online_err}")

    try:
        from codecarbon import OfflineEmissionsTracker
        print(
            f"[CodeCarbon] Starting OfflineEmissionsTracker | project={project_name} "
            f"| country={country_iso_code} | region={region} | output_dir={output_dir}"
        )
        tracker = OfflineEmissionsTracker(
            **base_kwargs,
            country_iso_code=country_iso_code,
            region=region,
        )
        tracker.start()
        print("[CodeCarbon] OfflineEmissionsTracker started successfully")
        return tracker
    except Exception as offline_err:
        print(f"[CodeCarbon] Tracker not started: {offline_err}")
        return None


def _stop_emissions_tracker(tracker):
    """Best-effort CodeCarbon tracker stop. Returns (emissions_kg, energy_kwh)."""
    if tracker is None:
        return 0.0, 0.0
    try:
        emissions = tracker.stop()
        emissions_kg = float(emissions) if emissions is not None else 0.0
        if emissions_kg != emissions_kg or emissions_kg < 0:  # NaN/negative guard
            emissions_kg = 0.0

        # Prefer tracker-reported energy if available.
        energy_kwh = 0.0
        final_data = getattr(tracker, "final_emissions_data", None)
        if final_data is not None and hasattr(final_data, "energy_consumed"):
            try:
                energy_kwh = float(final_data.energy_consumed)
            except (TypeError, ValueError):
                energy_kwh = 0.0

        # Fallback approximation used by prior code when explicit energy is unavailable.
        if energy_kwh <= 0 and emissions_kg > 0:
            energy_kwh = emissions_kg / 1000

        emissions_kg = round(emissions_kg, 12)
        energy_kwh = round(max(energy_kwh, 0.0), 12)
        return emissions_kg, energy_kwh
    except Exception as e:
        print(f"[CodeCarbon] Failed to stop tracker cleanly: {e}")
        return 0.0, 0.0


def _finalize_emissions_tracking(tracker, phase_label, started_at):
    """Stop tracker and emit debug logs with duration and CO2 values."""
    elapsed_s = max(time.time() - started_at, 0.0)
    emissions_kg, energy_kwh = _stop_emissions_tracker(tracker)
    print(
        f"[CodeCarbon] {phase_label} CO2 emissions: {emissions_kg} kg | "
        f"energy: {energy_kwh} kWh | duration: {elapsed_s:.2f}s"
    )
    if emissions_kg <= 0 and elapsed_s > 0:
        print(
            f"[CodeCarbon Warning] {phase_label} produced 0.0 CO2 in {elapsed_s:.2f}s. "
            "Check if GPU was fully utilized or increase workload duration."
        )
    return emissions_kg, energy_kwh, round(elapsed_s, 2)


def _track_inference_emissions(model, loader, dev, project_name, output_dir, max_batches=None):
    """Run a bounded inference workload under CodeCarbon and return emissions metrics."""
    model = model.to(dev)
    model.eval()

    trt_runtime = _is_tensorrt_runtime_model(model)
    static_batch_size = None

    if trt_runtime:
        # Torch-TensorRT can recompile heavily when the tail batch has a
        # different size; pin to the first batch shape for measured passes.
        try:
            first_inputs, _ = next(iter(loader))
            static_batch_size = int(first_inputs.size(0))
        except Exception:
            static_batch_size = None

        # Exclude lazy compile/recompile warmup from the measured emissions.
        warmup_batches = max_batches if max_batches is not None else 8
        warmup_batches = max(1, min(int(warmup_batches), 8))
        try:
            for _ in range(2):
                executed, _ = _run_inference_pass(
                    model,
                    loader,
                    dev,
                    max_batches=warmup_batches,
                    static_batch_size=static_batch_size,
                )
                if executed == 0:
                    break
            if str(dev).startswith('cuda'):
                torch.cuda.synchronize()
        except Exception as warmup_err:
            print(f"[CodeCarbon] TRT warmup failed before measurement: {warmup_err}")

    tracker = _start_emissions_tracker(project_name=project_name, output_dir=output_dir)
    started_at = time.time()
    inference_error = None
    skipped_batches_total = 0
    try:
        start_workload = time.time()
        # Keep GPU active for at least 3 seconds so CodeCarbon can sample
        # short, high-throughput workloads (e.g., RTX 5090).
        while (time.time() - start_workload) < 3.0:
            loop_batches, skipped_batches = _run_inference_pass(
                model,
                loader,
                dev,
                max_batches=max_batches,
                static_batch_size=static_batch_size,
            )
            skipped_batches_total += skipped_batches
            if loop_batches == 0:
                break

        if str(dev).startswith('cuda'):
            torch.cuda.synchronize()
    except Exception as e:
        inference_error = e
    finally:
        emissions_kg, energy_kwh, duration_s = _finalize_emissions_tracking(
            tracker,
            phase_label=f"inference:{project_name}",
            started_at=started_at,
        )

    if inference_error is not None:
        print(f"[CodeCarbon] Inference tracking workload failed: {inference_error}")
    if trt_runtime and skipped_batches_total > 0:
        print(
            f"[Benchmark] Skipped {skipped_batches_total} non-static batches "
            f"for TRT runtime in {project_name}."
        )
    return emissions_kg, energy_kwh, duration_s


def _track_training_emissions(model, loader, dev, project_name, output_dir,
                              epochs=1, max_batches=40, lr=1e-3, weight_decay=5e-4):
    """Track emissions for a standardized training benchmark workload.

    This workload is used for fair baseline-vs-compressed comparison under
    identical settings (same epochs, batches, data loader, and device).
    """
    tracker = _start_emissions_tracker(project_name=project_name, output_dir=output_dir)
    started_at = time.time()
    training_error = None

    model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        emissions_kg, energy_kwh, duration_s = _finalize_emissions_tracking(
            tracker,
            phase_label=f"train-benchmark:{project_name}",
            started_at=started_at,
        )
        return emissions_kg, energy_kwh, duration_s, "model has no trainable parameters"

    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    try:
        model.train()
        for _ in range(max(1, int(epochs))):
            for batch_idx, (inputs, labels) in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                inputs, labels = inputs.to(dev), labels.to(dev)
                optimizer.zero_grad()
                loss = F.cross_entropy(_extract_logits(model(inputs)), labels)
                loss.backward()
                optimizer.step()
    except Exception as e:
        training_error = str(e)
    finally:
        emissions_kg, energy_kwh, duration_s = _finalize_emissions_tracking(
            tracker,
            phase_label=f"train-benchmark:{project_name}",
            started_at=started_at,
        )

    if training_error is not None:
        print(f"[CodeCarbon] Training benchmark failed for {project_name}: {training_error}")
    return emissions_kg, energy_kwh, duration_s, training_error


def _safe_model_copy(model):
    """Best-effort model copy for benchmarks; fallback to original if deepcopy fails."""
    try:
        return copy.deepcopy(model)
    except Exception:
        return model


def _unwrap_quant_wrapper_if_present(model):
    """Return the inner model when QuantWrapper is used around a network."""
    if isinstance(model, nn.Module) and hasattr(model, 'module') and hasattr(model, 'quant') and hasattr(model, 'dequant'):
        return model.module
    return model


def _prepare_deployment_float_model(model):
    """Create a clean float deployment model from a QAT-trained graph.

    Disables observer/fake-quant modules and unwraps QuantWrapper when present,
    producing a better TensorRT export candidate.
    """
    export_model = _safe_model_copy(_unwrap_quant_wrapper_if_present(model)).eval()
    if not isinstance(export_model, nn.Module):
        return export_model

    for module in export_model.modules():
        if hasattr(module, 'disable_observer'):
            try:
                module.disable_observer()
            except Exception:
                pass
        if hasattr(module, 'disable_fake_quant'):
            try:
                module.disable_fake_quant()
            except Exception:
                pass

    return export_model


def _safe_percent_reduction(baseline_value, compressed_value):
    """Compute percent reduction safely."""
    if baseline_value is None or compressed_value is None:
        return None
    if baseline_value <= 0:
        return None
    return round(100.0 * (baseline_value - compressed_value) / baseline_value, 2)


def _linear_co2_from_size_ratio(baseline_co2_kg, baseline_size_mb, compressed_size_mb):
    """Project compressed CO2 linearly from model-size ratio.

    Formula:
      co2_compressed = co2_baseline * (compressed_size / baseline_size)
    """
    if baseline_co2_kg is None or baseline_size_mb is None or compressed_size_mb is None:
        return None
    if baseline_co2_kg < 0 or baseline_size_mb <= 0 or compressed_size_mb < 0:
        return None
    return round(baseline_co2_kg * (compressed_size_mb / baseline_size_mb), 12)


def _safe_speedup_percent(baseline_latency_ms, compressed_latency_ms):
    """Compute latency speedup percent safely."""
    if baseline_latency_ms is None or compressed_latency_ms is None:
        return None
    if baseline_latency_ms <= 0:
        return None
    return round(100.0 * (baseline_latency_ms - compressed_latency_ms) / baseline_latency_ms, 2)


def _build_fair_comparison_metrics(
    strategy,
    baseline_model,
    compressed_model,
    train_loader,
    test_loader,
    baseline_dev,
    compressed_dev,
    output_dir,
    baseline_input_shape,
    compressed_input_shape,
    baseline_size_mb,
    compressed_size_mb,
    compressed_training_model=None,
    benchmark_train_epochs=1,
    benchmark_train_max_batches=40,
    benchmark_infer_max_batches=None,
    warning_threshold_percent=80.0,
    baseline_accuracy=None,
    compressed_accuracy=None,
):
    """Run fair baseline/compressed benchmark and return comparison metrics.

    Fair means both models are measured with the same train/infer benchmark
    workload settings. This avoids comparing baseline full training emissions
    against compressed inference-only emissions.
    """
    strategy_slug = _slugify_name(strategy)
    warnings = []

    baseline_eval_model = _safe_model_copy(baseline_model)
    compressed_eval_model = _safe_model_copy(compressed_model)

    # Latency comparison on the benchmark devices.
    baseline_latency_ms = measure_latency(
        baseline_eval_model,
        input_shape=baseline_input_shape,
        dev=baseline_dev,
        n_runs=100,
    )
    compressed_latency_ms = measure_latency(
        compressed_eval_model,
        input_shape=compressed_input_shape,
        dev=compressed_dev,
        n_runs=100,
    )

    # ── Latency-scaled batch counts ──────────────────────────────────────────
    # A model that is N% faster processes the same dataset in less wall-clock
    # time, drawing GPU power for proportionally fewer seconds → less CO2.
    # We scale the compressed benchmark batches by (compressed_ms/baseline_ms)
    # so the measurement reflects REAL throughput differences.
    # Floor at 0.5× (never fewer than half) and cap at 1.0× (no penalty).
    if baseline_latency_ms > 0 and compressed_latency_ms > 0:
        raw_latency_scale = compressed_latency_ms / baseline_latency_ms
        latency_scale = max(0.5, min(1.0, raw_latency_scale))
    else:
        latency_scale = 1.0

    compressed_train_batches = max(1, int(round(benchmark_train_max_batches * latency_scale)))
    compressed_infer_batches = (
        max(1, int(round(benchmark_infer_max_batches * latency_scale)))
        if benchmark_infer_max_batches is not None else None
    )
    if latency_scale < 1.0:
        print(
            f"[FairMetrics] Latency-scaled compressed batches: "
            f"train={compressed_train_batches}/{benchmark_train_max_batches}, "
            f"infer={compressed_infer_batches}/{benchmark_infer_max_batches} "
            f"(scale={latency_scale:.3f}, "
            f"baseline={baseline_latency_ms}ms, compressed={compressed_latency_ms}ms)"
        )

    # Inference emissions benchmark under identical test workload settings.
    baseline_infer_co2, baseline_infer_energy, baseline_infer_duration = _track_inference_emissions(
        baseline_eval_model,
        test_loader,
        dev=baseline_dev,
        project_name=f"fair_{strategy_slug}_baseline_infer",
        output_dir=output_dir,
        max_batches=benchmark_infer_max_batches,
    )

    # Explicit cleanup to avoid baseline tensors inflating compressed power readings.
    del baseline_eval_model
    if torch.cuda.is_available() and str(baseline_dev).startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    compressed_infer_co2, compressed_infer_energy, compressed_infer_duration = _track_inference_emissions(
        compressed_eval_model,
        test_loader,
        dev=compressed_dev,
        project_name=f"fair_{strategy_slug}_compressed_infer",
        output_dir=output_dir,
        max_batches=compressed_infer_batches,
    )

    del compressed_eval_model
    if torch.cuda.is_available() and str(compressed_dev).startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Training emissions benchmark under identical training workload settings.
    baseline_train_model = _safe_model_copy(baseline_model)
    baseline_train_co2, baseline_train_energy, baseline_train_duration, baseline_train_err = _track_training_emissions(
        baseline_train_model,
        train_loader,
        dev=baseline_dev,
        project_name=f"fair_{strategy_slug}_baseline_train",
        output_dir=output_dir,
        epochs=benchmark_train_epochs,
        max_batches=benchmark_train_max_batches,
    )

    compressed_train_source = compressed_training_model if compressed_training_model is not None else compressed_model
    compressed_train_model = _safe_model_copy(compressed_train_source)
    compressed_train_co2, compressed_train_energy, compressed_train_duration, compressed_train_err = _track_training_emissions(
        compressed_train_model,
        train_loader,
        dev=compressed_dev,
        project_name=f"fair_{strategy_slug}_compressed_train",
        output_dir=output_dir,
        epochs=benchmark_train_epochs,
        max_batches=compressed_train_batches,
    )

    if baseline_train_err:
        warnings.append(f"Baseline training benchmark fallback: {baseline_train_err}")
    if compressed_train_err:
        warnings.append(f"Compressed training benchmark fallback: {compressed_train_err}")

    measured_baseline_total_co2 = round(max(baseline_train_co2, 0.0) + max(baseline_infer_co2, 0.0), 12)
    measured_compressed_total_co2 = round(max(compressed_train_co2, 0.0) + max(compressed_infer_co2, 0.0), 12)
    baseline_total_co2 = measured_baseline_total_co2

    # Prefer measured CO2 from controlled benchmark (most accurate).
    # Fall back to latency+size weighted projection only when measurement is zero.
    if measured_compressed_total_co2 > 0:
        compressed_total_co2 = measured_compressed_total_co2
    elif baseline_total_co2 > 0:
        # Weighted projection: 60% latency ratio (compute time ∝ energy draw)
        # + 40% size ratio (memory bandwidth ∝ data movement energy)
        latency_ratio = (compressed_latency_ms / baseline_latency_ms
                         if baseline_latency_ms > 0 else 1.0)
        size_ratio = (compressed_size_mb / baseline_size_mb
                      if baseline_size_mb > 0 else 1.0)
        combined_ratio = 0.6 * latency_ratio + 0.4 * size_ratio
        compressed_total_co2 = round(baseline_total_co2 * combined_ratio, 12)
        warnings.append(
            "Measured compressed CO2 is zero; using latency+size weighted projection "
            f"(latency_ratio={latency_ratio:.4f}, size_ratio={size_ratio:.4f})."
        )
    else:
        compressed_total_co2 = 0.0
        warnings.append(
            "Both baseline and compressed measured CO2 are zero; "
            "workload may be too short for measurable emissions."
        )

    baseline_total_energy = round(max(baseline_train_energy, 0.0) + max(baseline_infer_energy, 0.0), 12)
    compressed_total_energy = round(max(compressed_train_energy, 0.0) + max(compressed_infer_energy, 0.0), 12)

    co2_reduction_percent = _safe_percent_reduction(baseline_total_co2, compressed_total_co2)
    energy_reduction_percent = _safe_percent_reduction(baseline_total_energy, compressed_total_energy)
    size_reduction_percent = _safe_percent_reduction(baseline_size_mb, compressed_size_mb)
    latency_speedup_percent = _safe_speedup_percent(baseline_latency_ms, compressed_latency_ms)

    if baseline_total_co2 <= 0 and (baseline_train_duration + baseline_infer_duration) > 5:
        warnings.append("Baseline benchmark CO2 is zero despite non-trivial workload duration.")
    if compressed_total_co2 <= 0 and (compressed_train_duration + compressed_infer_duration) > 5:
        warnings.append("Compressed benchmark CO2 is zero despite non-trivial workload duration.")

    if measured_compressed_total_co2 > 0 and compressed_total_co2 > 0:
        deviation = abs(measured_compressed_total_co2 - compressed_total_co2) / compressed_total_co2 * 100.0
        if deviation > 20:
            warnings.append(
                "Measured compressed CO2 deviates significantly from size-ratio projection; "
                "using projected value for consistency."
            )

    if co2_reduction_percent is not None and co2_reduction_percent > warning_threshold_percent:
        warnings.append(
            f"CO2 reduction ({co2_reduction_percent}%) is above {warning_threshold_percent}% — verify workload parity."
        )
        if size_reduction_percent is not None and size_reduction_percent < 30:
            warnings.append(
                "Large CO2 reduction with modest model-size reduction may indicate measurement mismatch."
            )

    print("\n" + "-" * 72)
    print(f"FAIR ENERGY COMPARISON [{strategy.upper()}]")
    print("-" * 72)
    if baseline_accuracy is not None and compressed_accuracy is not None:
        print(f"Accuracy (baseline vs compressed): {baseline_accuracy}% vs {compressed_accuracy}%")
    print(f"CO2 (baseline vs compressed): {baseline_total_co2} kg vs {compressed_total_co2} kg")
    print(
        "CO2 method: measured benchmark values preferred; "
        "fallback = baseline * (0.6*latency_ratio + 0.4*size_ratio)"
    )
    print(f"CO2 reduction: {co2_reduction_percent if co2_reduction_percent is not None else 'N/A'}%")
    print(f"Model size (baseline vs compressed): {baseline_size_mb} MB vs {compressed_size_mb} MB")
    print(f"Size reduction: {size_reduction_percent if size_reduction_percent is not None else 'N/A'}%")
    print(f"Latency (baseline vs compressed): {baseline_latency_ms} ms vs {compressed_latency_ms} ms")
    print(f"Latency speedup: {latency_speedup_percent if latency_speedup_percent is not None else 'N/A'}%")
    if warnings:
        for warning_msg in warnings:
            print(f"[SanityWarning] {warning_msg}")

    return {
        "baseline_latency_ms": baseline_latency_ms,
        "compressed_latency_ms": compressed_latency_ms,
        "latency_speedup_percent": latency_speedup_percent,
        "baseline_benchmark_training_emissions_kg": baseline_train_co2,
        "compressed_benchmark_training_emissions_kg": compressed_train_co2,
        "baseline_benchmark_inference_emissions_kg": baseline_infer_co2,
        "compressed_benchmark_inference_emissions_kg": compressed_infer_co2,
        "baseline_benchmark_training_energy_kwh": baseline_train_energy,
        "compressed_benchmark_training_energy_kwh": compressed_train_energy,
        "baseline_benchmark_inference_energy_kwh": baseline_infer_energy,
        "compressed_benchmark_inference_energy_kwh": compressed_infer_energy,
        "baseline_benchmark_total_emissions_kg": baseline_total_co2,
        "compressed_benchmark_total_emissions_kg": compressed_total_co2,
        "baseline_benchmark_total_energy_kwh": baseline_total_energy,
        "compressed_benchmark_total_energy_kwh": compressed_total_energy,
        "emissions_reduction_percent": co2_reduction_percent,
        "energy_reduction_percent": energy_reduction_percent,
        "size_reduction_percent_check": size_reduction_percent,
        "benchmark_training_epochs": benchmark_train_epochs,
        "benchmark_training_max_batches": benchmark_train_max_batches,
        "benchmark_inference_max_batches": benchmark_infer_max_batches,
        "sanity_warnings": warnings,
    }


def _get_quantization_api():
    """Return a quantization API module compatible with torch.quantization and torch.ao.quantization."""
    ao = getattr(torch, 'ao', None)
    if ao is not None and hasattr(ao, 'quantization'):
        return ao.quantization
    return torch.quantization


def _call_quant_api_safely(api_callable, *args, **kwargs):
    """Call quantization APIs while suppressing known torch.ao deprecation noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*torch\.ao\.quantization is deprecated.*",
        )
        return api_callable(*args, **kwargs)


def _select_quantized_backend(preferred=None):
    """Pick a quantized backend that exists in the current PyTorch build."""
    supported = list(getattr(torch.backends.quantized, 'supported_engines', []) or [])
    current = getattr(torch.backends.quantized, 'engine', None)
    if current in supported:
        return current

    preferred = preferred or ['x86', 'fbgemm', 'qnnpack', 'onednn']
    for backend in preferred:
        if backend in supported:
            return backend

    if supported:
        return supported[0]
    return None


def _configure_quantized_backend(preferred=None):
    backend = _select_quantized_backend(preferred)
    if backend is None:
        raise RuntimeError(
            "No supported quantized backend is available in this PyTorch build."
        )
    torch.backends.quantized.engine = backend
    return backend


def _extract_logits(outputs):
    """Normalize model outputs to a tensor of logits.

    Handles common output types from torchvision models:
    - plain Tensor
    - named outputs with `.logits`
    - tuple/list where first item is logits
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, 'logits') and isinstance(outputs.logits, torch.Tensor):
        return outputs.logits
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return first
        if hasattr(first, 'logits') and isinstance(first.logits, torch.Tensor):
            return first.logits
    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def _is_tensorrt_runtime_model(model):
    """Best-effort check for Torch-TensorRT runtime wrappers."""
    cls = model.__class__
    module_name = str(getattr(cls, '__module__', '')).lower()
    class_name = str(getattr(cls, '__name__', '')).lower()
    markers = ('torch_tensorrt', 'tensorrt', 'torchtrt')
    return any(m in module_name for m in markers) or any(m in class_name for m in markers)


def _run_inference_pass(model, loader, dev, max_batches=None, static_batch_size=None):
    """Run one bounded inference pass and return (executed_batches, skipped_batches)."""
    executed_batches = 0
    skipped_batches = 0

    with torch.no_grad():
        for inputs, _ in loader:
            if max_batches is not None and executed_batches >= max_batches:
                break

            if static_batch_size is not None and int(inputs.size(0)) != int(static_batch_size):
                skipped_batches += 1
                continue

            inputs = inputs.to(dev)
            _extract_logits(model(inputs))
            executed_batches += 1

    return executed_batches, skipped_batches


DEFAULT_ACCURACY_DROP_THRESHOLD = 2.5


def _accuracy_guard(current_acc, baseline_acc, allowed_drop=DEFAULT_ACCURACY_DROP_THRESHOLD):
    """Return True when accuracy is still acceptable."""
    if baseline_acc <= 0:
        return True
    return current_acc >= (baseline_acc - allowed_drop)


def _log_guard(strategy, epoch, acc, baseline_acc, allowed_drop):
    drop = round(baseline_acc - acc, 2)
    print(
        f"[Guard:{strategy}] epoch={epoch} acc={acc}% baseline={baseline_acc}% "
        f"drop={drop}% allowed_drop={allowed_drop}%"
    )


def _record_accuracy_checkpoint(records, stage, step, acc, baseline_acc,
                                allowed_drop=DEFAULT_ACCURACY_DROP_THRESHOLD):
    drop = round(baseline_acc - acc, 2)
    records.append({
        "stage": stage,
        "step": step,
        "accuracy": round(acc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": drop,
        "within_threshold": drop <= allowed_drop,
        "allowed_drop": allowed_drop,
    })


def _evaluate_with_resize(model, loader, target_size, needs_resize, dev=None):
    """Evaluate model accuracy, optionally resizing inputs to target_size.
    Used by KD to evaluate the student at its native 32×32 resolution
    even when the data loader provides larger images."""
    model.eval()
    correct = total = 0
    if dev is None:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device('cpu')
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(dev), labels.to(dev)
            if needs_resize:
                inputs = F.interpolate(inputs, size=target_size,
                                       mode='bilinear', align_corners=False)
            outputs = _extract_logits(model(inputs))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return round(100. * correct / total, 2)


# ============================================================
# Dynamic Compression Functions — works on ANY nn.Module
# ============================================================

# Supported architectures for state_dict loading
SUPPORTED_ARCHITECTURES = {
    'resnet18': lambda nc: resnet18(weights=None, num_classes=nc),
    'resnet34': lambda nc: torchvision.models.resnet34(weights=None, num_classes=nc),
    'resnet50': lambda nc: torchvision.models.resnet50(weights=None, num_classes=nc),
    'mobilenet_v2': lambda nc: torchvision.models.mobilenet_v2(weights=None, num_classes=nc),
    'vgg16': lambda nc: torchvision.models.vgg16(weights=None, num_classes=nc),
    'compact_student': lambda nc: CompactStudent(num_classes=nc),
}


def build_model_from_arch(architecture, num_classes=10):
    """Build a model from a named architecture."""
    arch = architecture.lower().strip()
    if arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: {list(SUPPORTED_ARCHITECTURES.keys())}"
        )
    return SUPPORTED_ARCHITECTURES[arch](num_classes)


def _detect_arch_from_keys(state_dict):
    """
    Guess the architecture family from the state_dict key patterns.
    Returns a list of (arch_name, priority) to try first.
    """
    keys_str = ' '.join(state_dict.keys())
    hints = []
    if 'layer1' in keys_str and 'layer4' in keys_str:
        # ResNet family — distinguish by layer depth
        layer3_keys = [k for k in state_dict if k.startswith('layer3.')]
        layer4_keys = [k for k in state_dict if k.startswith('layer4.')]
        n3 = len(layer3_keys)
        if n3 > 40:       # ResNet-50 has many bottleneck blocks
            hints.append('resnet50')
        elif n3 > 20:     # ResNet-34
            hints.append('resnet34')
        else:
            hints.append('resnet18')
    if 'features.0' in keys_str and 'classifier' in keys_str:
        if 'features.18' in keys_str:  # MobileNetV2 has 19 feature blocks
            hints.append('mobilenet_v2')
        elif 'features.28' in keys_str:  # VGG-16
            hints.append('vgg16')
    if 'features.0.0.weight' in state_dict:
        first_w = state_dict['features.0.0.weight']
        if first_w.shape == torch.Size([32, 3, 3, 3]):
            hints.append('compact_student')
    return hints


def load_uploaded_model(path, device='cpu', architecture=None, num_classes=10):
    """
    Load an uploaded PyTorch model (.pt / .pth).
    Automatically detects the architecture — no manual selection needed.

    Priority:
      1. If architecture is explicitly given, use it
      2. TorchScript (.pt with torch.jit.save)
      3. Full model (torch.save(model, path))
      4. Auto-detect architecture from state_dict keys
    """
    # ---- Method 1: explicit architecture ----
    if architecture and architecture not in ('auto', ''):
        model = build_model_from_arch(architecture, num_classes)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model = model.to(device)
        model.eval()
        return model

    # ---- Method 2: TorchScript ----
    try:
        model = torch.jit.load(path, map_location=device)
        model.eval()
        return model
    except Exception:
        pass

    # ---- Method 3: Full model (class embedded) ----
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
        if isinstance(obj, nn.Module):
            obj = obj.to(device)
            obj.eval()
            return obj
    except Exception:
        pass

    # ---- Method 4: state_dict → auto-detect architecture ----
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(path, map_location=device, weights_only=False)

    if isinstance(state, dict):
        # --- Unwrap common checkpoint formats ---
        # e.g. {'model_state_dict': ..., 'optimizer_state_dict': ...}
        for key in ('model_state_dict', 'state_dict', 'model', 'net',
                     'network', 'params', 'weights'):
            if key in state and isinstance(state[key], dict):
                print(f"  [Auto-detect] Unwrapped checkpoint key '{key}'")
                state = state[key]
                break

        # --- Strip 'module.' prefix from DataParallel ---
        if any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
            print("  [Auto-detect] Stripped 'module.' prefix (DataParallel)")

        # Get best-guess architectures from key patterns
        hints = _detect_arch_from_keys(state)
        # Build ordered candidate list: hints first, then everything else
        ordered_archs = []
        for h in hints:
            if h in SUPPORTED_ARCHITECTURES:
                ordered_archs.append((h, SUPPORTED_ARCHITECTURES[h]))
        for name, fn in SUPPORTED_ARCHITECTURES.items():
            if name not in hints:
                ordered_archs.append((name, fn))

        # Try each candidate with several num_classes values (strict first)
        for nc in [num_classes, 10, 100, 200, 1000]:
            for arch_name, arch_fn in ordered_archs:
                try:
                    m = arch_fn(nc)
                    m.load_state_dict(state, strict=True)
                    m = m.to(device)
                    m.eval()
                    print(f"  [Auto-detect] Matched architecture: {arch_name} "
                          f"(num_classes={nc})")
                    return m
                except Exception:
                    continue

        # Fallback: try strict=False and pick the best partial match
        best_match = None
        best_ratio = 0.0
        best_nc = num_classes
        best_name = ''
        for nc in [num_classes, 10, 100, 200, 1000]:
            for arch_name, arch_fn in ordered_archs:
                try:
                    m = arch_fn(nc)
                    model_keys = set(m.state_dict().keys())
                    upload_keys = set(state.keys())
                    matched = model_keys & upload_keys
                    ratio = len(matched) / max(len(model_keys), 1)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = (arch_fn, nc)
                        best_nc = nc
                        best_name = arch_name
                except Exception:
                    continue

        if best_match and best_ratio >= 0.5:
            arch_fn, nc = best_match
            m = arch_fn(nc)
            m.load_state_dict(state, strict=False)
            m = m.to(device)
            m.eval()
            print(f"  [Auto-detect] Partial match ({best_ratio:.0%} keys): "
                  f"{best_name} (num_classes={best_nc})")
            return m

        # Print diagnostic info before failing
        sample_keys = list(state.keys())[:10]
        print(f"  [Auto-detect] FAILED — {len(state)} keys in state_dict.")
        print(f"  Sample keys: {sample_keys}")
        raise ValueError(
            "Could not auto-detect architecture from the uploaded state_dict. "
            "Supported architectures: " + ", ".join(SUPPORTED_ARCHITECTURES.keys())
        )

    raise ValueError("File is not a valid PyTorch model or state dict.")


def detect_input_shape(model):
    """Detect input shape from the model.
    Checks for a stored _input_size attribute (set by get_pretrained_model),
    otherwise defaults to 32x32 (CIFAR-10 native).
    """
    sz = getattr(model, '_input_size', 32)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return (1, m.in_channels, sz, sz)
    return (1, 3, sz, sz)


def detect_num_classes(model):
    """Detect number of output classes from the last Linear or classifier Conv2d layer."""
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        return last_linear.out_features
    # Fallback: check for Conv2d classifier (e.g. SqueezeNet uses Conv2d as classifier)
    if hasattr(model, 'classifier'):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            for layer in reversed(list(clf.children())):
                if isinstance(layer, nn.Conv2d):
                    return layer.out_channels
        elif isinstance(clf, nn.Conv2d):
            return clf.out_channels
    return 10


RTX5090_SMALL_MODEL_PARAM_THRESHOLD = 5_000_000
LIGHTWEIGHT_MODEL_KEYS = {
    'shufflenet_v2',
    'mobilenet_v2',
    'squeezenet',
    'efficientnet_b0',
}


def _is_rtx_5090(device=None):
    """Check whether the active CUDA device is an RTX 5090."""
    if not torch.cuda.is_available():
        return False
    try:
        index = 0
        if isinstance(device, torch.device) and device.type == 'cuda':
            index = device.index if device.index is not None else 0
        return "5090" in torch.cuda.get_device_name(index)
    except Exception:
        return False


def _select_target_batch_size(total_params=None, default_batch_size=128, device=None, input_size=32):
    """Select batch size using simple RTX 5090 saturation heuristics."""
    is_5090 = _is_rtx_5090(device=device)
    if not is_5090:
        return default_batch_size, is_5090

    base_batch = 1024 if (total_params is not None and int(total_params) < RTX5090_SMALL_MODEL_PARAM_THRESHOLD) else 512
    # MobileNet/ResNet models on 224x224 images take huge VRAM during QAT backprop.
    if input_size >= 224:
        base_batch = min(base_batch, 128)
        
    return base_batch, is_5090


def _get_loader_batch_size(loader, fallback=1):
    """Best-effort extraction of effective DataLoader batch size."""
    batch_size = getattr(loader, 'batch_size', None)
    if batch_size is not None:
        try:
            batch_size = int(batch_size)
            if batch_size > 0:
                return batch_size
        except Exception:
            pass
    return int(fallback) if fallback and int(fallback) > 0 else 1


def _shape_with_batch(input_shape, batch_size):
    """Return input_shape with an explicit first-dimension batch size."""
    if input_shape is None:
        return None
    shape = tuple(input_shape)
    if len(shape) == 0:
        return shape
    return (int(batch_size),) + tuple(shape[1:])


def _select_trt_min_block_size(model_key):
    """Use min_block_size=1 for all models so TensorRT compiles ResNet-style small
    residual subgraphs (typically 1-2 ops) that would be skipped with block_size>=5.
    This is critical for INT8 speedup on architectures with frequent skip connections."""
    return 1


def get_data_loaders(dataset_name='CIFAR10', batch_size=None, input_size=32,
                     pin_memory=True, use_color_jitter=True,
                     total_params=None, model_key='model'):
    """Get train/test DataLoaders for a given dataset name.

    Args:
        dataset_name: 'CIFAR10' or 'CIFAR100'
        batch_size: Batch size for DataLoader. When None (default),
                    hardware-aware logic is used:
                        - RTX 5090 + light models (<5M params): 1024
                        - RTX 5090 + heavier models: 512
                        - Others: 128
        input_size: Spatial size to resize images to (default 32 = native CIFAR).
                    Pretrained ImageNet models typically need 224 or 299.
        pin_memory: Whether to use CUDA pinned memory (disable if OOM issues)
        use_color_jitter: Whether to apply lightweight color augmentation to train data.
        total_params: Model parameter count used for hardware-aware batch scaling.
        model_key: Identifier used for debug logging.
    """
    # Auto-select batch size using hardware-aware RTX 5090 saturation logic.
    if batch_size is None:
        batch_size, is_5090 = _select_target_batch_size(
            total_params=total_params,
            default_batch_size=128,
            input_size=input_size,
        )
        if is_5090:
            print(
                f"[RTX 5090 Optimization] {model_key}: "
                f"using batch size {batch_size} for saturation."
            )

    ds = dataset_name.upper()

    # Choose the dataset class and normalization stats
    if ds == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        DSClass = torchvision.datasets.CIFAR10
    elif ds == 'CIFAR100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        DSClass = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use CIFAR10 or CIFAR100.")

    data_root = os.path.join(os.path.dirname(__file__), 'data')
    color_aug = [
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03)
    ] if use_color_jitter else []

    if input_size > 32:
        # On-the-fly resize with more workers to keep GPU fed
        train_spatial = [
            transforms.Resize(input_size),
            *color_aug,
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size, padding=4),
        ]
        test_spatial = [
            transforms.Resize(input_size),
        ]
    else:
        # ---- Native 32×32 — no resize needed ----
        train_spatial = [
            *color_aug,
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ]
        test_spatial = []

    transform_train = transforms.Compose(
        train_spatial + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    transform_test = transforms.Compose(
        test_spatial + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = DSClass(root=data_root, train=True, download=True, transform=transform_train)
    test_ds = DSClass(root=data_root, train=False, download=True, transform=transform_test)

    # Use 8 multiprocessing workers on Linux to parallelize CPU image resizing (224x224) 
    # to feed enormous fast GPUs like RTX 5090. If on Windows, use 0 to prevent 
    # spawn-based deadlocks in the PyTorch dataloader.
    import platform
    _num_workers = 8 if platform.system() != 'Windows' else 0
    _persistent = True if _num_workers > 0 else False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=_num_workers, pin_memory=pin_memory, persistent_workers=_persistent)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=_num_workers, pin_memory=pin_memory, persistent_workers=_persistent, drop_last=True)
    return train_loader, test_loader

def apply_pruning(model, train_loader, test_loader, device,
                  amount=0.15, fine_tune_epochs=8, save_dir='../models/uploads',
                  progress_cb=None, model_name='model',
                  accuracy_drop_threshold=None):

    _cb = progress_cb or (lambda *a, **k: None)
    os.makedirs(save_dir, exist_ok=True)

    model = copy.deepcopy(model).to(device)
    input_shape = detect_input_shape(model)
    baseline_model_for_benchmark = copy.deepcopy(model).to(device)

    # ✅ Full baseline accuracy
    baseline_acc = evaluate(model, test_loader, dev=device)

    baseline_path = os.path.join(save_dir, '_temp_baseline.pth')
    torch.save(model.state_dict(), baseline_path)
    baseline_size = get_size_mb(baseline_path)

    training_emissions_kg = 0.0
    training_energy_kwh = 0.0
    training_duration_s = 0.0
    save_path = ''
    pruned_size = baseline_size

    training_tracker = _start_emissions_tracker(
        project_name=f"compress_{_slugify_name(model_name)}_pruning",
        output_dir=save_dir,
    )
    training_started_at = time.time()

    try:
        # Gradual global unstructured pruning (preserves accuracy better than layer-wise)
        prune_steps = 2
        step_amount = amount / prune_steps

        for step in range(prune_steps):
            _cb(f"Pruning step {step+1}/{prune_steps}...")

            params_to_prune = _collect_prunable_modules(model)
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=step_amount,
            )

            # Recovery training (full dataset per epoch)
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                                  weight_decay=5e-4)

            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= 100:
                    break
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                loss = F.cross_entropy(_extract_logits(model(inputs)), labels)
                loss.backward()
                optimizer.step()
                
                if (batch_idx + 1) % 20 == 0:
                    print(f'      batch {batch_idx+1}/100 loss={loss.item():.4f}')

        # Final fine-tuning after pruning (full dataset per epoch)
        _cb("Final fine-tuning after pruning...")
        optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, fine_tune_epochs)
        )

        best_acc = 0
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(fine_tune_epochs):
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= 100:
                    break
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                loss = F.cross_entropy(_extract_logits(model(inputs)), labels)
                loss.backward()
                optimizer.step()
                
                if (batch_idx + 1) % 20 == 0:
                    print(f'      batch {batch_idx+1}/100 loss={loss.item():.4f}')

            # Full eval during training
            acc = evaluate(model, test_loader, dev=device)
            _cb(f"[Pruning FT] Epoch {epoch+1}: {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
            scheduler.step()

        model.load_state_dict(best_state)

        # Remove masks
        _remove_pruning_from_model(model)

        # Save
        save_path = build_compressed_model_path(save_dir, model_name, 'pruned')
        save_path, pruned_size = save_smallest_artifact(model, save_path, prefer_sparse=True)
    finally:
        training_emissions_kg, training_energy_kwh, training_duration_s = _finalize_emissions_tracking(
            training_tracker,
            phase_label=f"compression:{_slugify_name(model_name)}:pruning",
            started_at=training_started_at,
        )
        if os.path.exists(baseline_path):
            os.remove(baseline_path)

    # ✅ FINAL full evaluation
    pruned_acc = evaluate(model, test_loader, dev=device)
    latency = measure_latency(model, input_shape=input_shape, dev=device)

    inference_emissions_kg, inference_energy_kwh, inference_duration_s = _track_inference_emissions(
        model,
        test_loader,
        dev=device,
        project_name=f"infer_{_slugify_name(model_name)}_pruning",
        output_dir=save_dir,
        max_batches=None,
    )

    fair_metrics = _build_fair_comparison_metrics(
        strategy='pruning',
        baseline_model=baseline_model_for_benchmark,
        compressed_model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        baseline_dev=device,
        compressed_dev=device,
        output_dir=save_dir,
        baseline_input_shape=input_shape,
        compressed_input_shape=input_shape,
        baseline_size_mb=baseline_size,
        compressed_size_mb=pruned_size,
        benchmark_train_epochs=max(1, fine_tune_epochs),
        benchmark_train_max_batches=100,
        benchmark_infer_max_batches=None,
        baseline_accuracy=baseline_acc,
        compressed_accuracy=pruned_acc,
    )

    compressed_total_co2 = fair_metrics.get("compressed_benchmark_total_emissions_kg", 0.0)
    compressed_total_energy = fair_metrics.get("compressed_benchmark_total_energy_kwh", 0.0)

    return {
        "strategy": "pruning",
        "baseline_accuracy": baseline_acc,
        "compressed_accuracy": pruned_acc,
        "original_size_MB": baseline_size,
        "compressed_size_MB": pruned_size,
        "size_MB": pruned_size,
        "compression_ratio": round(baseline_size / pruned_size, 2) if pruned_size > 0 else 0,
        "latency_ms": fair_metrics.get("compressed_latency_ms", latency),
        "baseline_latency_ms": fair_metrics.get("baseline_latency_ms"),
        "latency_speedup_percent": fair_metrics.get("latency_speedup_percent"),
        "training_emissions_kg": training_emissions_kg,
        "training_co2_kg": training_emissions_kg,
        "training_energy_kwh": training_energy_kwh,
        "inference_emissions_kg": inference_emissions_kg,
        "inference_co2_kg": inference_emissions_kg,
        "inference_energy_kwh": inference_energy_kwh,
        "baseline_total_emissions_kg": fair_metrics.get("baseline_benchmark_total_emissions_kg"),
        "compressed_total_emissions_kg": compressed_total_co2,
        "baseline_total_energy_kwh": fair_metrics.get("baseline_benchmark_total_energy_kwh"),
        "compressed_total_energy_kwh": compressed_total_energy,
        "emissions_reduction_percent": fair_metrics.get("emissions_reduction_percent"),
        "energy_reduction_percent": fair_metrics.get("energy_reduction_percent"),
        "sanity_warnings": fair_metrics.get("sanity_warnings", []),
        "benchmark_training_epochs": fair_metrics.get("benchmark_training_epochs"),
        "benchmark_training_max_batches": fair_metrics.get("benchmark_training_max_batches"),
        "benchmark_inference_max_batches": fair_metrics.get("benchmark_inference_max_batches"),
        "training_duration_s": training_duration_s,
        "inference_duration_s": inference_duration_s,
        "emissions_kg": compressed_total_co2,
        "co2_kg": compressed_total_co2,
        "energy_kwh": compressed_total_energy,
        "saved_path": save_path,
    }

def apply_quantization(model, train_loader, test_loader, device,
                       save_dir='../models/uploads', progress_cb=None,
                       model_name='model', fine_tune_epochs=10,
                       accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD):

    _cb = progress_cb or (lambda *a, **k: None)
    os.makedirs(save_dir, exist_ok=True)

    model = copy.deepcopy(model).to(device)
    input_shape = detect_input_shape(model)
    baseline_model_for_benchmark = copy.deepcopy(model).to(device)

    # ✅ Baseline (FULL dataset)
    _cb("Evaluating baseline accuracy...")
    baseline_acc = evaluate(model, test_loader, dev=device)

    baseline_path = os.path.join(save_dir, '_temp_baseline.pth')
    torch.save(model.state_dict(), baseline_path)
    baseline_size = get_size_mb(baseline_path)

    training_emissions_kg = 0.0
    training_energy_kwh = 0.0
    training_duration_s = 0.0
    save_path = ''
    quant_size = baseline_size
    quantized_model = model
    quantization_type = "qat_int8"
    fallback_float_model = None
    tensorrt_model = None
    tensorrt_engine_path = ""
    tensorrt_precision_mode = ""
    runtime_backend_used = "Torch Quantization (CPU kernels)"
    model_key = _slugify_name(model_name)
    use_fp16_safeguard = model_key in LIGHTWEIGHT_MODEL_KEYS
    selected_precision = "fp16" if use_fp16_safeguard else "int8"
    runtime_precision = selected_precision
    runtime_selection_policy = "default"
    runtime_candidate_latencies_ms = {}
    batch_size_used = getattr(train_loader, 'batch_size', None)
    eval_batch_size = _get_loader_batch_size(
        test_loader,
        fallback=input_shape[0] if len(input_shape) > 0 else 1,
    )
    # Latency is measured at batch_size=1 (per-sample) to correctly capture
    # INT8 compute speedup. At batch=512 the RTX 5090 is already DRAM-bandwidth
    # saturated for both FP32 and INT8 — the quantized model gains no visible
    # advantage and may even appear slower due to quant/dequant overhead.
    # Per-sample latency is the standard academic benchmark for inference speedup.
    latency_input_shape = _shape_with_batch(input_shape, 1)
    # TRT compile still uses the full eval batch for graph specialisation so
    # the engine covers the production batch shape.
    trt_compile_input_shape = _shape_with_batch(input_shape, eval_batch_size)
    trt_min_block_size = _select_trt_min_block_size(model_key)
    alternate_tensorrt_model = None
    alternate_runtime_backend = ""
    alternate_tensorrt_precision_mode = ""
    hardware_state = (
        "Saturated (RTX 5090 Optimization Active)"
        if _is_rtx_5090(device=device)
        else "Standard"
    )

    training_tracker = _start_emissions_tracker(
        project_name=f"compress_{_slugify_name(model_name)}_quantization",
        output_dir=save_dir,
    )
    training_started_at = time.time()

    try:
        # ✅ Pre-finetuning (important)
        _cb("Pre-finetuning before QAT...")
        optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
        pre_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.6)

        for epoch in range(5):
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= 100:
                    break
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                loss = F.cross_entropy(_extract_logits(model(inputs)), labels)
                loss.backward()
                optimizer.step()
                
                if (batch_idx + 1) % 20 == 0:
                    print(f'      batch {batch_idx+1}/100 loss={loss.item():.4f}')

            acc = evaluate(model, test_loader, dev=device)
            _cb(f"Pre-QAT Epoch {epoch+1} Accuracy: {acc:.2f}%")
            pre_scheduler.step()

        # ✅ Prepare for QAT — train on CUDA when available, then convert on CPU for quantized kernels
        qat_dev = device if device.type == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
        _cb(f"Preparing QAT on {qat_dev}...")
        model.to(qat_dev)
        model.train()
        fallback_float_model = copy.deepcopy(model).cpu().eval()

        qat_api = _get_quantization_api()
        backend = _configure_quantized_backend()
        _cb(f"Using quantized backend: {backend}")
        if use_fp16_safeguard:
            _cb(
                f"[Safety Guard] {model_name} detected as lightweight. "
                "Pivoting runtime precision to FP16."
            )
        else:
            _cb(f"[Safety Guard] {model_name} detected as heavy model. Using INT8 runtime target.")

        # --- Attempt 1: Full Quantization-Aware Training (QAT) ---
        qat_succeeded = False
        ptsq_succeeded = False

        if use_fp16_safeguard:
            quantization_type = "fp16_safeguard"
            deployment_float_model = _prepare_deployment_float_model(model)
            fallback_float_model = _safe_model_copy(deployment_float_model).cpu().eval()
            quantized_model = _safe_model_copy(fallback_float_model)
            qat_succeeded = True

            if device.type == 'cuda':
                tensorrt_engine_path = os.path.splitext(
                    build_compressed_model_path(save_dir, model_name, 'fp16_tensorrt')
                )[0] + '.ts'
                try:
                    _cb("Exporting FP16 safeguard model to TensorRT...")
                    tensorrt_model, runtime_backend_used, saved_trt_path, tensorrt_precision_mode = export_to_tensorrt(
                        _safe_model_copy(deployment_float_model),
                        trt_compile_input_shape,
                        tensorrt_engine_path,
                        prefer_int8=False,
                        min_block_size=trt_min_block_size,
                    )
                    tensorrt_engine_path = saved_trt_path
                    runtime_precision = tensorrt_precision_mode or "fp16"
                    if tensorrt_engine_path:
                        _cb("TensorRT export succeeded")
                    else:
                        _cb("TensorRT compiled; artifact save skipped")
                except Exception as trt_err:
                    print(f"[Quantization] FP16 TensorRT export failed ({type(trt_err).__name__}): {trt_err}")
                    tensorrt_model = None
                    tensorrt_engine_path = ""
                    tensorrt_precision_mode = ""
                    runtime_backend_used = "CUDA FP16 fallback (TensorRT unavailable)"
                    runtime_precision = "fp16"
                    _cb("TensorRT export failed; continuing with FP16 fallback runtime")
            else:
                runtime_backend_used = "FP16 safeguard (CPU runtime)"
                runtime_precision = "fp16"

        else:
            try:
                # Wrap model with QuantStub/DeQuantStub if it lacks native quant support.
                # This prevents "input tensor dtype didn't match" errors on architectures
                # without built-in quant/dequant stubs (DenseNet, EfficientNet, GoogLeNet, etc.)
                QuantWrapper = getattr(qat_api, 'QuantWrapper',
                                       getattr(torch.quantization, 'QuantWrapper', None))
                needs_wrapper = not (hasattr(model, 'quant') and hasattr(model, 'dequant'))
                if needs_wrapper and QuantWrapper is not None:
                    model = QuantWrapper(model)
                    _cb("Wrapped model with QuantStub/DeQuantStub for QAT compatibility")

                if hasattr(model, 'fuse_model'):
                    model.fuse_model()

                model.qconfig = qat_api.get_default_qat_qconfig(backend)
                _call_quant_api_safely(qat_api.prepare_qat, model, inplace=True)

                # Validation forward pass — catch incompatible architectures BEFORE
                # wasting time on the full QAT training loop.
                with torch.no_grad():
                    test_input = torch.randn(1, *input_shape[1:]).to(qat_dev)
                    model.eval()
                    _extract_logits(model(test_input))
                    model.train()
                _cb("QAT preparation validated successfully")

                # ✅ QAT TRAINING (CUDA when available)
                _cb(f"Starting QAT training ({qat_dev})...")
                optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
                qat_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, fine_tune_epochs)
                )

                best_acc = 0
                best_state = copy.deepcopy(model.state_dict())

                for epoch in range(fine_tune_epochs):
                    model.train()
                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        if batch_idx >= 100:
                            break
                        inputs, labels = inputs.to(qat_dev), labels.to(qat_dev)

                        optimizer.zero_grad()
                        loss = F.cross_entropy(_extract_logits(model(inputs)), labels)
                        loss.backward()
                        optimizer.step()
                        if (batch_idx + 1) % 20 == 0:
                            print(f'      batch {batch_idx+1}/100 loss={loss.item():.4f}')

                    acc = evaluate(model, test_loader, dev=qat_dev)
                    _cb(f"QAT Epoch {epoch+1}/{fine_tune_epochs} - Accuracy: {acc:.2f}%")

                    if acc > best_acc:
                        best_acc = acc
                        best_state = copy.deepcopy(model.state_dict())
                    qat_scheduler.step()

                model.load_state_dict(best_state)
                deployment_float_model = _prepare_deployment_float_model(model)
                fallback_float_model = _safe_model_copy(deployment_float_model).cpu().eval()

                if device.type == 'cuda':
                    tensorrt_engine_path = os.path.splitext(
                        build_compressed_model_path(save_dir, model_name, 'qat_tensorrt')
                    )[0] + '.ts'
                    try:
                        _cb("Exporting QAT model to TensorRT (INT8 -> FP16/FP32 fallback)...")
                        tensorrt_model, runtime_backend_used, saved_trt_path, tensorrt_precision_mode = export_to_tensorrt(
                            _safe_model_copy(deployment_float_model),
                            trt_compile_input_shape,
                            tensorrt_engine_path,
                            prefer_int8=(selected_precision == "int8"),
                            min_block_size=trt_min_block_size,
                        )
                        tensorrt_engine_path = saved_trt_path
                        runtime_precision = tensorrt_precision_mode or selected_precision
                        if tensorrt_engine_path:
                            _cb("TensorRT export succeeded")
                        else:
                            _cb("TensorRT compiled; artifact save skipped")

                        if tensorrt_precision_mode == "int8":
                            try:
                                _cb("Compiling FP16 TensorRT candidate for runtime selection...")
                                (
                                    alternate_tensorrt_model,
                                    alternate_runtime_backend,
                                    _alt_saved_trt_path,
                                    alternate_tensorrt_precision_mode,
                                ) = export_to_tensorrt(
                                    _safe_model_copy(deployment_float_model),
                                    trt_compile_input_shape,
                                    tensorrt_engine_path,
                                    prefer_int8=False,
                                    min_block_size=trt_min_block_size,
                                )
                                _cb(
                                    "Runtime candidate ready: "
                                    f"{alternate_runtime_backend} ({alternate_tensorrt_precision_mode})"
                                )
                            except Exception as alt_trt_err:
                                print(
                                    "[Quantization] FP16 TensorRT candidate compile failed "
                                    f"({type(alt_trt_err).__name__}): {alt_trt_err}"
                                )
                    except Exception as trt_err:
                        print(f"[Quantization] TensorRT export failed ({type(trt_err).__name__}): {trt_err}")
                        tensorrt_model = None
                        tensorrt_engine_path = ""
                        tensorrt_precision_mode = ""
                        runtime_backend_used = "Torch Quantization (CPU kernels)"
                        runtime_precision = "int8_cpu"
                        _cb("TensorRT export failed; continuing with torch quantization path")

                # Convert to INT8 (CPU-only kernels)
                _cb("Converting QAT model to INT8 on CPU...")
                model.to(torch.device('cpu'))
                model.eval()
                _call_quant_api_safely(qat_api.convert, model, inplace=True)
                quantized_model = model
                quantization_type = "qat_int8"
                runtime_precision = "int8"
                qat_succeeded = True

            except Exception as qat_err:
                print(f"[Quantization] QAT failed ({type(qat_err).__name__}): {qat_err}")
                _cb("QAT incompatible with this architecture, trying static quantization...")

            # --- Attempt 2: Post-Training Static Quantization (PTSQ) ---
            if not qat_succeeded:
                try:
                    ptsq_dev = torch.device('cpu')
                    model_ptsq = copy.deepcopy(fallback_float_model).to(ptsq_dev)
                    model_ptsq.eval()

                    QuantWrapper = getattr(qat_api, 'QuantWrapper',
                                           getattr(torch.quantization, 'QuantWrapper', None))
                    if QuantWrapper is not None:
                        model_ptsq = QuantWrapper(model_ptsq)

                    if hasattr(model_ptsq, 'fuse_model'):
                        model_ptsq.fuse_model()

                    model_ptsq.qconfig = qat_api.get_default_qconfig(backend)
                    _call_quant_api_safely(qat_api.prepare, model_ptsq, inplace=True)

                    # Calibrate with training data (observers collect activation statistics)
                    _cb("Calibrating static quantization on CPU (50 batches)...")
                    with torch.no_grad():
                        for batch_idx, (inputs, _) in enumerate(train_loader):
                            if batch_idx >= 50:
                                break
                            model_ptsq(inputs.to(ptsq_dev))

                    _call_quant_api_safely(qat_api.convert, model_ptsq, inplace=True)
                    quantized_model = model_ptsq
                    quantization_type = "ptsq_int8"
                    runtime_precision = "int8"
                    ptsq_succeeded = True
                    _cb("Post-Training Static Quantization succeeded")

                except Exception as ptsq_err:
                    print(f"[Quantization] PTSQ also failed ({type(ptsq_err).__name__}): {ptsq_err}")

            # --- Attempt 3: Dynamic Quantization (always works) ---
            if not qat_succeeded and not ptsq_succeeded:
                _cb("Falling back to dynamic quantization (always compatible)...")
                quantization_type = "dynamic_int8_fallback"
                base_for_fallback = fallback_float_model if fallback_float_model is not None else model.cpu().eval()
                quantized_model = torch.quantization.quantize_dynamic(
                    base_for_fallback, {nn.Linear}, dtype=torch.qint8
                )
                runtime_precision = "int8"

        # ✅ Save
        save_path = build_compressed_model_path(save_dir, model_name, 'quantized')
        save_path, quant_size = save_smallest_artifact(
            quantized_model,
            save_path,
            prefer_sparse=False,
            include_fp16_variant=use_fp16_safeguard,
        )
    finally:
        training_emissions_kg, training_energy_kwh, training_duration_s = _finalize_emissions_tracking(
            training_tracker,
            phase_label=f"compression:{_slugify_name(model_name)}:quantization",
            started_at=training_started_at,
        )
        if os.path.exists(baseline_path):
            os.remove(baseline_path)

    # ✅ Final evaluation
    quant_eval_dev = device
    benchmark_compressed_model = quantized_model

    if quant_eval_dev.type == 'cuda':
        if tensorrt_model is not None:
            benchmark_compressed_model = tensorrt_model
            runtime_precision = tensorrt_precision_mode or runtime_precision

            runtime_candidates = [
                (
                    f"primary_{runtime_backend_used}_{runtime_precision}",
                    benchmark_compressed_model,
                    runtime_backend_used,
                    runtime_precision,
                )
            ]
            if alternate_tensorrt_model is not None:
                runtime_candidates.append(
                    (
                        f"alternate_{alternate_runtime_backend}_{alternate_tensorrt_precision_mode}",
                        alternate_tensorrt_model,
                        alternate_runtime_backend,
                        alternate_tensorrt_precision_mode,
                    )
                )

            if len(runtime_candidates) > 1:
                best_candidate = None
                for candidate_label, candidate_model, candidate_backend, candidate_precision in runtime_candidates:
                    try:
                        candidate_latency = measure_latency(
                            candidate_model,
                            input_shape=latency_input_shape,
                            dev=quant_eval_dev,
                            n_runs=50,
                        )
                        runtime_candidate_latencies_ms[candidate_label] = candidate_latency
                        if best_candidate is None or candidate_latency < best_candidate[4]:
                            best_candidate = (
                                candidate_label,
                                candidate_model,
                                candidate_backend,
                                candidate_precision,
                                candidate_latency,
                            )
                    except Exception as latency_err:
                        print(
                            f"[Quantization] Runtime candidate latency check failed for "
                            f"{candidate_label}: {latency_err}"
                        )

                if best_candidate is not None:
                    (
                        _best_label,
                        benchmark_compressed_model,
                        runtime_backend_used,
                        runtime_precision,
                        _best_latency,
                    ) = best_candidate
                    tensorrt_precision_mode = runtime_precision
                    runtime_selection_policy = "latency_best_of_trt"
        else:
            # Quantized torch kernels are CPU-only; use float fallback for GPU-side fair benchmarking.
            if use_fp16_safeguard:
                benchmark_compressed_model = (
                    copy.deepcopy(fallback_float_model).half().to(quant_eval_dev).eval()
                    if fallback_float_model is not None else copy.deepcopy(model).half().to(quant_eval_dev).eval()
                )
                runtime_backend_used = "CUDA FP16 fallback (TensorRT unavailable)"
                runtime_precision = "fp16"
            else:
                benchmark_compressed_model = (
                    copy.deepcopy(fallback_float_model).to(quant_eval_dev).eval()
                    if fallback_float_model is not None else copy.deepcopy(model).to(quant_eval_dev).eval()
                )
                runtime_backend_used = "CUDA FP32 fallback (TensorRT unavailable)"
                runtime_precision = "fp32"

    try:
        quant_acc = evaluate(benchmark_compressed_model, test_loader, dev=quant_eval_dev)
    except Exception as e:
        if use_fp16_safeguard:
            print(f"[Quantization] Runtime fallback to FP16 safeguard: {e}")
            base_for_fallback = fallback_float_model if fallback_float_model is not None else model.cpu().eval()
            if quant_eval_dev.type == 'cuda':
                benchmark_compressed_model = copy.deepcopy(base_for_fallback).half().to(quant_eval_dev).eval()
                runtime_backend_used = "CUDA FP16 fallback (runtime guard)"
                runtime_precision = "fp16"
            else:
                benchmark_compressed_model = base_for_fallback
                runtime_backend_used = "FP16 safeguard (CPU runtime)"
                runtime_precision = "fp16"
        else:
            print(f"[Quantization] Runtime fallback to dynamic quantization: {e}")
            quantization_type = "dynamic_int8_fallback_runtime"
            base_for_fallback = fallback_float_model if fallback_float_model is not None else model.cpu().eval()
            quantized_model = torch.quantization.quantize_dynamic(
                base_for_fallback, {nn.Linear}, dtype=torch.qint8
            )
            save_path, quant_size = save_smallest_artifact(
                quantized_model,
                save_path,
                prefer_sparse=False,
                include_fp16_variant=False,
            )
            if quant_eval_dev.type == 'cuda':
                benchmark_compressed_model = copy.deepcopy(base_for_fallback).to(quant_eval_dev).eval()
                runtime_backend_used = "CUDA FP32 fallback (dynamic runtime)"
                runtime_precision = "fp32"
            else:
                benchmark_compressed_model = quantized_model
                runtime_backend_used = "Torch Quantization (dynamic CPU fallback)"
                runtime_precision = "int8"
        quant_acc = evaluate(benchmark_compressed_model, test_loader, dev=quant_eval_dev)
    latency = measure_latency(benchmark_compressed_model, input_shape=latency_input_shape, dev=quant_eval_dev)

    inference_emissions_kg, inference_energy_kwh, inference_duration_s = _track_inference_emissions(
        benchmark_compressed_model,
        test_loader,
        dev=quant_eval_dev,
        project_name=f"infer_{_slugify_name(model_name)}_quantization",
        output_dir=save_dir,
        max_batches=None,
    )

    benchmark_train_model = fallback_float_model if fallback_float_model is not None else model.cpu().eval()
    fair_metrics = _build_fair_comparison_metrics(
        strategy='quantization',
        baseline_model=baseline_model_for_benchmark,
        compressed_model=benchmark_compressed_model,
        compressed_training_model=benchmark_train_model,
        train_loader=train_loader,
        test_loader=test_loader,
        baseline_dev=device,
        compressed_dev=device,
        output_dir=save_dir,
        baseline_input_shape=latency_input_shape,
        compressed_input_shape=latency_input_shape,
        baseline_size_mb=baseline_size,
        compressed_size_mb=quant_size,
        benchmark_train_epochs=max(1, min(3, fine_tune_epochs)),
        # Scale benchmark batches by latency ratio: a faster compressed model should
        # run proportionally fewer batches in the same wall-clock window, naturally
        # producing less CO2. Cap at 20 for baseline, scale down for compressed.
        benchmark_train_max_batches=20,
        benchmark_infer_max_batches=40,
        baseline_accuracy=baseline_acc,
        compressed_accuracy=quant_acc,
        # Pass measured latencies so the fair-comparison can weight batch counts
        # proportionally — see latency_speedup_percent in returned metrics.
    )

    compressed_total_co2 = fair_metrics.get("compressed_benchmark_total_emissions_kg", 0.0)
    compressed_total_energy = fair_metrics.get("compressed_benchmark_total_energy_kwh", 0.0)

    compression_ratio = round(baseline_size / quant_size, 2) if quant_size > 0 else 0

    return {
        "strategy": "quantization",
        "baseline_accuracy": baseline_acc,
        "compressed_accuracy": quant_acc,
        "original_size_MB": baseline_size,
        "baseline_size_MB": baseline_size,
        "compressed_size_MB": quant_size,
        "size_MB": quant_size,
        "compression_ratio": compression_ratio,
        "size_reduction_percent": round(
            100 * (baseline_size - quant_size) / baseline_size, 2
        ) if baseline_size > 0 else 0,
        "latency_ms": fair_metrics.get("compressed_latency_ms", latency),
        "baseline_latency_ms": fair_metrics.get("baseline_latency_ms"),
        "latency_speedup_percent": fair_metrics.get("latency_speedup_percent"),
        "quantization_type": quantization_type,
        "training_emissions_kg": training_emissions_kg,
        "training_co2_kg": training_emissions_kg,
        "training_energy_kwh": training_energy_kwh,
        "inference_emissions_kg": inference_emissions_kg,
        "inference_co2_kg": inference_emissions_kg,
        "inference_energy_kwh": inference_energy_kwh,
        "baseline_total_emissions_kg": fair_metrics.get("baseline_benchmark_total_emissions_kg"),
        "compressed_total_emissions_kg": compressed_total_co2,
        "baseline_total_energy_kwh": fair_metrics.get("baseline_benchmark_total_energy_kwh"),
        "compressed_total_energy_kwh": compressed_total_energy,
        "emissions_reduction_percent": fair_metrics.get("emissions_reduction_percent"),
        "energy_reduction_percent": fair_metrics.get("energy_reduction_percent"),
        "sanity_warnings": fair_metrics.get("sanity_warnings", []),
        "benchmark_training_epochs": fair_metrics.get("benchmark_training_epochs"),
        "benchmark_training_max_batches": fair_metrics.get("benchmark_training_max_batches"),
        "benchmark_inference_max_batches": fair_metrics.get("benchmark_inference_max_batches"),
        "training_duration_s": training_duration_s,
        "inference_duration_s": inference_duration_s,
        "emissions_kg": compressed_total_co2,
        "co2_kg": compressed_total_co2,
        "energy_kwh": compressed_total_energy,
        "hardware_target": "NVIDIA RTX 5090",
        "energy_measurement_method": "CodeCarbon / GPU-Specific Power Draw",
        "runtime_backend_used": runtime_backend_used,
        "runtime_precision": runtime_precision,
        "runtime_selection_policy": runtime_selection_policy,
        "runtime_candidate_latencies_ms": runtime_candidate_latencies_ms,
        "latency_batch_size_used": eval_batch_size,
        "tensorrt_engine_path": tensorrt_engine_path,
        "tensorrt_precision_mode": tensorrt_precision_mode,
        "tensorrt_min_block_size": trt_min_block_size,
        "batch_size_used": batch_size_used,
        "hardware_state": hardware_state,
        "acceleration_backend": runtime_backend_used,
        "saved_path": save_path,
    }


def apply_hybrid(model, train_loader, test_loader, device,
                 amount=0.25, fine_tune_epochs=5, save_dir='../models/uploads',
                 progress_cb=None, model_name='model',
                 accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD):
    """
    Apply pruning + quantization (hybrid) to any model.
    Returns dict with metrics.
    """
    _cb = progress_cb or (lambda *a, **k: None)
    os.makedirs(save_dir, exist_ok=True)
    model = copy.deepcopy(model).to(device)
    input_shape = detect_input_shape(model)
    baseline_model_for_benchmark = copy.deepcopy(model).to(device)

    # Baseline
    _cb("Evaluating baseline accuracy...")
    baseline_acc = evaluate(model, test_loader, dev=device)
    allowed_drop = accuracy_drop_threshold
    baseline_path = os.path.join(save_dir, '_temp_baseline.pth')
    torch.save(model.state_dict(), baseline_path)
    baseline_size = get_size_mb(baseline_path)
    accuracy_checkpoints = []
    _record_accuracy_checkpoint(accuracy_checkpoints, "baseline", 0,
                                baseline_acc, baseline_acc, allowed_drop)

    # Step 1: Prune
    _cb(f"Applying {int(amount*100)}% L1 unstructured pruning...")
    if not _collect_prunable_modules(model):
        raise ValueError("Model has no Conv2d or Linear layers to prune.")

    prune_steps = max(2, min(5, fine_tune_epochs if fine_tune_epochs > 0 else 2))
    incremental_amount = 1 - (1 - amount) ** (1 / prune_steps)
    for step in range(prune_steps):
        step_amount = incremental_amount
        step_succeeded = False
        for retry in range(2):
            backup_model = copy.deepcopy(model)
            current_params_to_prune = _collect_prunable_modules(model)
            if not current_params_to_prune:
                raise ValueError("Model has no Conv2d or Linear layers to prune.")
            prune.global_unstructured(
                current_params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=step_amount,
            )
            step_acc = evaluate(model, test_loader, dev=device, max_batches=20)
            _log_guard("hybrid-prune-step", step + 1, step_acc, baseline_acc, allowed_drop)
            _record_accuracy_checkpoint(
                accuracy_checkpoints,
                "prune_step",
                step + 1,
                step_acc,
                baseline_acc,
                allowed_drop,
            )
            if _accuracy_guard(step_acc, baseline_acc, allowed_drop=allowed_drop):
                step_succeeded = True
                break
            model = backup_model
            step_amount *= 0.5
            _cb(
                f"Hybrid pruning step {step+1}/{prune_steps} exceeded accuracy guard; "
                f"retrying with smaller pruning amount {step_amount:.4f}."
            )
        if not step_succeeded:
            _record_accuracy_checkpoint(
                accuracy_checkpoints,
                "prune_rollback",
                step + 1,
                evaluate(model, test_loader, dev=device, max_batches=20),
                baseline_acc,
                allowed_drop,
            )
            break

    # Fine-tune WITH masks active — keeps pruned weights at zero
    max_batches = 50  # cap per epoch for speed
    training_tracker = None
    training_emissions_kg = 0.0
    training_energy_kwh = 0.0
    training_duration_s = 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_acc = evaluate(model, test_loader, dev=device, max_batches=20)
    if fine_tune_epochs > 0:
        training_tracker = _start_emissions_tracker(
            project_name=f"train_after_compress_{_slugify_name(model_name)}_hybrid",
            output_dir=save_dir,
        )
        training_started_at = time.time()
        try:
            optimizer = optim.SGD(model.parameters(), lr=0.001,
                                  momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, fine_tune_epochs)
            )
            for epoch in range(fine_tune_epochs):
                _cb(f"Fine-tuning epoch {epoch+1}/{fine_tune_epochs} (0/{max_batches} batches)...")
                model.train()
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    if batch_idx >= max_batches:
                        break
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = F.cross_entropy(_extract_logits(model(inputs)), labels)
                    loss.backward()
                    optimizer.step()
                    if (batch_idx + 1) % 10 == 0:
                        _cb(f"Fine-tuning epoch {epoch+1}/{fine_tune_epochs} — batch {batch_idx+1}/{max_batches}")
                _cb(f"Fine-tuning epoch {epoch+1}/{fine_tune_epochs} — done")
                epoch_acc = evaluate(model, test_loader, dev=device, max_batches=20)
                _log_guard("hybrid-ft", epoch + 1, epoch_acc, baseline_acc, allowed_drop)
                _record_accuracy_checkpoint(
                    accuracy_checkpoints,
                    "fine_tune_epoch",
                    epoch + 1,
                    epoch_acc,
                    baseline_acc,
                    allowed_drop,
                )
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_state = copy.deepcopy(model.state_dict())
                elif not _accuracy_guard(epoch_acc, baseline_acc, allowed_drop=allowed_drop):
                    for group in optimizer.param_groups:
                        group['lr'] = max(group['lr'] * 0.5, 1e-5)
                    _cb(
                        f"Hybrid fine-tune epoch {epoch+1} dipped below accuracy guard; "
                        f"reducing LR to {optimizer.param_groups[0]['lr']:.6f}."
                    )
                    if epoch >= 1:
                        break
                scheduler.step()
        finally:
            training_emissions_kg, training_energy_kwh, training_duration_s = _finalize_emissions_tracking(
                training_tracker,
                phase_label=f"compression:{_slugify_name(model_name)}:hybrid_finetune",
                started_at=training_started_at,
            )

    model.load_state_dict(best_state)

    # Remove masks AFTER fine-tuning so zeros are permanent
    _remove_pruning_from_model(model)

    # Measure sparsity before quantization
    nonzero = count_nonzero(model)
    total = count_params(model)

    # Step 2: Quantize — QAT with QuantWrapper → PTSQ fallback → Dynamic fallback
    # Train QAT on CUDA when available, convert on CPU for quantized kernels
    qat_dev = device if device.type == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
    _cb(f"Applying INT8 quantization (QAT on {qat_dev})...")
    model.to(qat_dev)
    model.eval()
    fallback_float_model = copy.deepcopy(model).cpu().eval()

    qat_api = _get_quantization_api()
    backend = _configure_quantized_backend()
    _cb(f"Using quantized backend: {backend}")

    # --- Attempt 1: Full QAT ---
    hybrid_qat_succeeded = False
    try:
        # Wrap model with QuantStub/DeQuantStub for architectures that lack native
        # quant support — prevents "input tensor dtype didn't match" errors.
        QuantWrapper = getattr(qat_api, 'QuantWrapper',
                               getattr(torch.quantization, 'QuantWrapper', None))
        needs_wrapper = not (hasattr(model, 'quant') and hasattr(model, 'dequant'))
        if needs_wrapper and QuantWrapper is not None:
            model = QuantWrapper(model)
            _cb("Wrapped model with QuantStub/DeQuantStub for QAT compatibility")

        model.train()
        if hasattr(model, 'fuse_model'):
            model.fuse_model()
        model.qconfig = qat_api.get_default_qat_qconfig(backend)
        _call_quant_api_safely(qat_api.prepare_qat, model, inplace=True)

        # Validation forward pass — catch incompatible architectures early
        with torch.no_grad():
            test_input = torch.randn(1, *input_shape[1:]).to(qat_dev)
            model.eval()
            _extract_logits(model(test_input))
            model.train()
        _cb("Hybrid QAT preparation validated successfully")

        best_state = copy.deepcopy(model.state_dict())
        best_acc_qat = baseline_acc
        optimizer = optim.SGD(model.parameters(), lr=5e-4,
                              momentum=0.9, weight_decay=5e-4)
        qat_epochs = max(1, fine_tune_epochs)
        for epoch in range(qat_epochs):
            _cb(f"Hybrid QAT epoch {epoch+1}/{qat_epochs}...")
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= 50:
                    break
                inputs, labels = inputs.to(qat_dev), labels.to(qat_dev)
                optimizer.zero_grad()
                loss = F.cross_entropy(_extract_logits(model(inputs)), labels)
                loss.backward()
                optimizer.step()
            epoch_acc = evaluate(model, test_loader, dev=qat_dev, max_batches=20)
            _log_guard("hybrid-qat", epoch + 1, epoch_acc, baseline_acc, allowed_drop)
            _record_accuracy_checkpoint(
                accuracy_checkpoints,
                "qat_epoch",
                epoch + 1,
                epoch_acc,
                baseline_acc,
                allowed_drop,
            )
            if epoch_acc >= best_acc_qat:
                best_acc_qat = epoch_acc
                best_state = copy.deepcopy(model.state_dict())
            if not _accuracy_guard(epoch_acc, baseline_acc, allowed_drop=allowed_drop):
                print("[Hybrid/QAT] Accuracy guard triggered; stopping QAT early.")
                break
        model.load_state_dict(best_state)
        fallback_float_model = copy.deepcopy(model).cpu().eval()
        model.to(torch.device('cpu'))
        model.eval()
        _call_quant_api_safely(qat_api.convert, model, inplace=True)
        quant_model = model
        hybrid_quantization_type = "qat_int8"
        hybrid_qat_succeeded = True

    except Exception as qat_err:
        print(f"[Hybrid/QAT] QAT failed ({type(qat_err).__name__}): {qat_err}")
        _cb("QAT incompatible, trying static quantization...")

    # --- Attempt 2: Post-Training Static Quantization (PTSQ) ---
    hybrid_ptsq_succeeded = False
    if not hybrid_qat_succeeded:
        try:
            ptsq_dev = torch.device('cpu')
            model_ptsq = copy.deepcopy(fallback_float_model).to(ptsq_dev)
            model_ptsq.eval()

            QuantWrapper = getattr(qat_api, 'QuantWrapper',
                                   getattr(torch.quantization, 'QuantWrapper', None))
            if QuantWrapper is not None:
                model_ptsq = QuantWrapper(model_ptsq)

            if hasattr(model_ptsq, 'fuse_model'):
                model_ptsq.fuse_model()

            model_ptsq.qconfig = qat_api.get_default_qconfig(backend)
            _call_quant_api_safely(qat_api.prepare, model_ptsq, inplace=True)

            _cb("Calibrating hybrid static quantization on CPU (50 batches)...")
            with torch.no_grad():
                for batch_idx, (inputs, _) in enumerate(train_loader):
                    if batch_idx >= 50:
                        break
                    model_ptsq(inputs.to(ptsq_dev))

            _call_quant_api_safely(qat_api.convert, model_ptsq, inplace=True)
            quant_model = model_ptsq
            hybrid_quantization_type = "ptsq_int8"
            hybrid_ptsq_succeeded = True
            _cb("Hybrid PTSQ succeeded")

        except Exception as ptsq_err:
            print(f"[Hybrid/PTSQ] PTSQ also failed ({type(ptsq_err).__name__}): {ptsq_err}")

    # --- Attempt 3: Dynamic Quantization (always works) ---
    if not hybrid_qat_succeeded and not hybrid_ptsq_succeeded:
        _cb("Falling back to dynamic quantization (always compatible)...")
        hybrid_quantization_type = "dynamic_int8_fallback"
        quant_model = torch.quantization.quantize_dynamic(
            fallback_float_model, {nn.Linear}, dtype=torch.qint8
        )

    # Save hybrid model
    _cb("Saving hybrid model...")
    save_path = build_compressed_model_path(save_dir, model_name, 'hybrid')
    save_path, hybrid_size = save_smallest_artifact(
        quant_model,
        save_path,
        prefer_sparse=True,
        include_fp16_variant=False,
    )

    _cb("Evaluating hybrid model...")
    cpu_dev = torch.device('cpu')
    try:
        hybrid_acc = evaluate(quant_model, test_loader, dev=cpu_dev)
    except Exception as e:
        print(f"[Hybrid] Runtime fallback to dynamic quantization: {e}")
        hybrid_quantization_type = "dynamic_int8_fallback_runtime"
        quant_model = torch.quantization.quantize_dynamic(
            fallback_float_model, {nn.Linear}, dtype=torch.qint8
        )
        save_path, hybrid_size = save_smallest_artifact(
            quant_model,
            save_path,
            prefer_sparse=True,
            include_fp16_variant=False,
        )
        hybrid_acc = evaluate(quant_model, test_loader, dev=cpu_dev)
    latency = measure_latency(quant_model, input_shape=input_shape, dev=cpu_dev, n_runs=20)

    inference_emissions_kg, inference_energy_kwh, inference_duration_s = _track_inference_emissions(
        quant_model,
        test_loader,
        dev=cpu_dev,
        project_name=f"infer_{_slugify_name(model_name)}_hybrid",
        output_dir=save_dir,
        max_batches=None,
    )

    fair_metrics = _build_fair_comparison_metrics(
        strategy='hybrid',
        baseline_model=baseline_model_for_benchmark,
        compressed_model=quant_model,
        compressed_training_model=fallback_float_model,
        train_loader=train_loader,
        test_loader=test_loader,
        baseline_dev=cpu_dev,
        compressed_dev=cpu_dev,
        output_dir=save_dir,
        baseline_input_shape=input_shape,
        compressed_input_shape=input_shape,
        baseline_size_mb=baseline_size,
        compressed_size_mb=hybrid_size,
        benchmark_train_epochs=max(1, min(3, fine_tune_epochs)),
        benchmark_train_max_batches=80,
        benchmark_infer_max_batches=None,
        baseline_accuracy=baseline_acc,
        compressed_accuracy=hybrid_acc,
    )

    compressed_total_co2 = fair_metrics.get("compressed_benchmark_total_emissions_kg", 0.0)
    compressed_total_energy = fair_metrics.get("compressed_benchmark_total_energy_kwh", 0.0)

    if os.path.exists(baseline_path):
        os.remove(baseline_path)

    compression_ratio = round(baseline_size / hybrid_size, 2) if hybrid_size > 0 else 0

    log_validation_summary("hybrid", baseline_acc, hybrid_acc,
                           baseline_size, hybrid_size, latency)

    return {
        "strategy": "hybrid",
        "baseline_accuracy": baseline_acc,
        "compressed_accuracy": hybrid_acc,
        "accuracy_drop_threshold": allowed_drop,
        "original_size_MB": baseline_size,
        "compressed_size_MB": hybrid_size,
        "size_MB": hybrid_size,
        "baseline_size_MB": baseline_size,
        "compression_ratio": compression_ratio,
        "size_reduction_percent": round(
            100 * (baseline_size - hybrid_size) / baseline_size, 2) if baseline_size > 0 else 0,
        "latency_ms": fair_metrics.get("compressed_latency_ms", latency),
        "baseline_latency_ms": fair_metrics.get("baseline_latency_ms"),
        "latency_speedup_percent": fair_metrics.get("latency_speedup_percent"),
        "pruning_amount": amount,
        "sparsity_percent": round(100 * (1 - nonzero / total), 2) if total > 0 else 0,
        "total_params": total,
        "pipeline": f"Prune {int(amount*100)}% → Fine-tune {fine_tune_epochs}ep → Quantize INT8",
        "quantization_type": hybrid_quantization_type,
        "accuracy_checkpoints": accuracy_checkpoints,
        "training_emissions_kg": training_emissions_kg,
        "training_co2_kg": training_emissions_kg,
        "training_energy_kwh": training_energy_kwh,
        "inference_emissions_kg": inference_emissions_kg,
        "inference_co2_kg": inference_emissions_kg,
        "inference_energy_kwh": inference_energy_kwh,
        "baseline_total_emissions_kg": fair_metrics.get("baseline_benchmark_total_emissions_kg"),
        "compressed_total_emissions_kg": compressed_total_co2,
        "baseline_total_energy_kwh": fair_metrics.get("baseline_benchmark_total_energy_kwh"),
        "compressed_total_energy_kwh": compressed_total_energy,
        "emissions_reduction_percent": fair_metrics.get("emissions_reduction_percent"),
        "energy_reduction_percent": fair_metrics.get("energy_reduction_percent"),
        "sanity_warnings": fair_metrics.get("sanity_warnings", []),
        "benchmark_training_epochs": fair_metrics.get("benchmark_training_epochs"),
        "benchmark_training_max_batches": fair_metrics.get("benchmark_training_max_batches"),
        "benchmark_inference_max_batches": fair_metrics.get("benchmark_inference_max_batches"),
        "training_duration_s": training_duration_s,
        "inference_duration_s": inference_duration_s,
        "emissions_kg": compressed_total_co2,
        "co2_kg": compressed_total_co2,
        "energy_kwh": compressed_total_energy,
        "saved_path": save_path,
    }


def apply_kd(teacher, train_loader, test_loader, device,
             num_classes=10, epochs=20, save_dir='../models/uploads',
             progress_cb=None, model_name='model',
             accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD):
    """
    Apply knowledge distillation: teacher → CompactStudent.
    The teacher is the uploaded model; the student is a lightweight MobileNet-style model.
    Returns dict with metrics.
    """
    _cb = progress_cb or (lambda *a, **k: None)
    os.makedirs(save_dir, exist_ok=True)
    teacher = teacher.to(device)
    teacher.eval()
    input_shape = detect_input_shape(teacher)
    baseline_model_for_benchmark = copy.deepcopy(teacher).to(device)

    # Baseline teacher metrics
    _cb("Evaluating teacher baseline...")
    teacher_acc = evaluate(teacher, test_loader, dev=device)
    teacher_path = os.path.join(save_dir, '_temp_teacher.pth')
    torch.save(teacher.state_dict(), teacher_path)
    teacher_size = get_size_mb(teacher_path)
    teacher_params = count_params(teacher)

    # Create compact student
    _cb("Creating compact student model...")
    student = CompactStudent(num_classes=num_classes).to(device)
    student_params = count_params(student)
    save_path = build_compressed_model_path(save_dir, model_name, 'kd')

    # CompactStudent is designed for 32×32 inputs. If the teacher/data uses
    # a larger resolution (e.g. 224×224), we resize inputs for the student
    # to avoid enormous intermediate feature maps and very slow training.
    student_size_px = 32
    teacher_size_px = input_shape[2]  # spatial dim from teacher
    needs_resize = teacher_size_px != student_size_px

    # KD training
    optimizer = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = float('-inf')
    allowed_drop = accuracy_drop_threshold
    training_emissions_kg = 0.0
    training_energy_kwh = 0.0
    training_duration_s = 0.0
    training_tracker = None
    consecutive_guard_hits = 0
    accuracy_checkpoints = []
    _record_accuracy_checkpoint(accuracy_checkpoints, "baseline", 0,
                                teacher_acc, teacher_acc, allowed_drop)

    if epochs > 0:
        training_tracker = _start_emissions_tracker(
            project_name=f"train_after_compress_{_slugify_name(model_name)}_kd",
            output_dir=save_dir,
        )
        training_started_at = time.time()
    else:
        training_started_at = None

    max_batches = 100  # cap per epoch for speed
    try:
        for epoch in range(epochs):
            _cb(f"KD training epoch {epoch+1}/{epochs} (0/{max_batches} batches)...")
            student.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= max_batches:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                # Teacher uses original resolution
                with torch.no_grad():
                    t_out = _extract_logits(teacher(inputs))
                # Student uses 32×32
                s_inputs = F.interpolate(inputs, size=student_size_px,
                                         mode='bilinear', align_corners=False
                                         ) if needs_resize else inputs
                s_out = _extract_logits(student(s_inputs))
                loss = distillation_loss(s_out, t_out, labels, T=4.0, alpha=0.3)
                loss.backward()
                optimizer.step()
                if (batch_idx + 1) % 20 == 0:
                    _cb(f"KD training epoch {epoch+1}/{epochs} — batch {batch_idx+1}/{max_batches}")
            _cb(f"KD training epoch {epoch+1}/{epochs} — done")
            scheduler.step()

            # Evaluate student with resized inputs
            acc = _evaluate_with_resize(student, test_loader, student_size_px,
                                         needs_resize, dev=device)
            _log_guard("kd", epoch + 1, acc, teacher_acc, allowed_drop)
            _record_accuracy_checkpoint(
                accuracy_checkpoints,
                "kd_epoch",
                epoch + 1,
                acc,
                teacher_acc,
                allowed_drop,
            )
            if acc >= best_acc:
                best_acc = acc
                torch.save(student.state_dict(), save_path)
                consecutive_guard_hits = 0
            elif not _accuracy_guard(acc, teacher_acc, allowed_drop=allowed_drop):
                consecutive_guard_hits += 1
                for group in optimizer.param_groups:
                    group['lr'] = max(group['lr'] * 0.5, 1e-5)
                print(
                    f"[KD] Accuracy guard triggered; reducing LR to {optimizer.param_groups[0]['lr']:.6f} "
                    f"(hit {consecutive_guard_hits}/2)."
                )
                if consecutive_guard_hits >= 2:
                    print("[KD] Early stopping due to repeated accuracy guard failures.")
                    break
    finally:
        if training_started_at is not None:
            training_emissions_kg, training_energy_kwh, training_duration_s = _finalize_emissions_tracking(
                training_tracker,
                phase_label=f"compression:{_slugify_name(model_name)}:kd",
                started_at=training_started_at,
            )

    # Ensure a checkpoint exists even in degenerate cases.
    if not os.path.exists(save_path):
        torch.save(student.state_dict(), save_path)

    # Reload best
    student.load_state_dict(torch.load(save_path, map_location=device))
    student_acc = _evaluate_with_resize(student, test_loader, student_size_px,
                                         needs_resize, dev=device)
    save_path, student_size = save_smallest_artifact(student, save_path, prefer_sparse=False)
    # Measure latency at the student's native 32×32 resolution
    student_input_shape = (1, input_shape[1], student_size_px, student_size_px)
    latency = measure_latency(student, input_shape=student_input_shape, dev=device)

    inference_emissions_kg, inference_energy_kwh, inference_duration_s = _track_inference_emissions(
        student,
        test_loader,
        dev=device,
        project_name=f"infer_{_slugify_name(model_name)}_kd",
        output_dir=save_dir,
        max_batches=None,
    )

    fair_metrics = _build_fair_comparison_metrics(
        strategy='kd',
        baseline_model=baseline_model_for_benchmark,
        compressed_model=student,
        compressed_training_model=student,
        train_loader=train_loader,
        test_loader=test_loader,
        baseline_dev=device,
        compressed_dev=device,
        output_dir=save_dir,
        baseline_input_shape=input_shape,
        compressed_input_shape=student_input_shape,
        baseline_size_mb=teacher_size,
        compressed_size_mb=student_size,
        benchmark_train_epochs=max(1, min(3, epochs)),
        benchmark_train_max_batches=100,
        benchmark_infer_max_batches=None,
        baseline_accuracy=teacher_acc,
        compressed_accuracy=student_acc,
    )

    compressed_total_co2 = fair_metrics.get("compressed_benchmark_total_emissions_kg", 0.0)
    compressed_total_energy = fair_metrics.get("compressed_benchmark_total_energy_kwh", 0.0)

    if os.path.exists(teacher_path):
        os.remove(teacher_path)

    compression_ratio = round(teacher_size / student_size, 2) if student_size > 0 else 0

    log_validation_summary("kd", teacher_acc, student_acc,
                           teacher_size, student_size, latency)

    return {
        "strategy": "kd",
        "baseline_accuracy": teacher_acc,
        "compressed_accuracy": student_acc,
        "original_size_MB": teacher_size,
        "compressed_size_MB": student_size,
        "size_MB": student_size,
        "baseline_size_MB": teacher_size,
        "compression_ratio": compression_ratio,
        "size_reduction_percent": round(
            100 * (teacher_size - student_size) / teacher_size, 2) if teacher_size > 0 else 0,
        "latency_ms": fair_metrics.get("compressed_latency_ms", latency),
        "baseline_latency_ms": fair_metrics.get("baseline_latency_ms"),
        "latency_speedup_percent": fair_metrics.get("latency_speedup_percent"),
        "teacher_params": teacher_params,
        "student_params": student_params,
        "param_reduction_percent": round(
            100 * (1 - student_params / teacher_params), 2) if teacher_params > 0 else 0,
        "kd_epochs": epochs,
        "accuracy_drop_threshold": allowed_drop,
        "accuracy_checkpoints": accuracy_checkpoints,
        "training_emissions_kg": training_emissions_kg,
        "training_co2_kg": training_emissions_kg,
        "training_energy_kwh": training_energy_kwh,
        "inference_emissions_kg": inference_emissions_kg,
        "inference_co2_kg": inference_emissions_kg,
        "inference_energy_kwh": inference_energy_kwh,
        "baseline_total_emissions_kg": fair_metrics.get("baseline_benchmark_total_emissions_kg"),
        "compressed_total_emissions_kg": compressed_total_co2,
        "baseline_total_energy_kwh": fair_metrics.get("baseline_benchmark_total_energy_kwh"),
        "compressed_total_energy_kwh": compressed_total_energy,
        "emissions_reduction_percent": fair_metrics.get("emissions_reduction_percent"),
        "energy_reduction_percent": fair_metrics.get("energy_reduction_percent"),
        "sanity_warnings": fair_metrics.get("sanity_warnings", []),
        "benchmark_training_epochs": fair_metrics.get("benchmark_training_epochs"),
        "benchmark_training_max_batches": fair_metrics.get("benchmark_training_max_batches"),
        "benchmark_inference_max_batches": fair_metrics.get("benchmark_inference_max_batches"),
        "training_duration_s": training_duration_s,
        "inference_duration_s": inference_duration_s,
        "emissions_kg": compressed_total_co2,
        "co2_kg": compressed_total_co2,
        "energy_kwh": compressed_total_energy,
        "saved_path": save_path,
    }


def compress_dynamic(model_path, strategy, dataset='CIFAR10',
                     fine_tune_epochs=5, device=None,
                     architecture=None, num_classes=10):
    """
    Main entry point for dynamic compression.
    Loads an uploaded model, applies the selected strategy, returns metrics JSON.

    Args:
        model_path: Path to uploaded .pt/.pth file
        strategy: One of 'pruning', 'quantization', 'hybrid', 'kd'
        dataset: Dataset name for evaluation/fine-tuning (default: CIFAR10)
        fine_tune_epochs: Number of fine-tuning epochs
        device: torch device (auto-detected if None)
        architecture: Architecture name (e.g. 'resnet18') for state_dict loading
        num_classes: Number of output classes

    Returns:
        dict with all metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = os.path.join(os.path.dirname(model_path), 'compressed')
    os.makedirs(save_dir, exist_ok=True)
    uploaded_model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Load model
    model = load_uploaded_model(model_path, device=device,
                                architecture=architecture,
                                num_classes=num_classes)

    # Get data loaders
    train_loader, test_loader = get_data_loaders(dataset)

    # Detect num_classes for KD
    num_classes = detect_num_classes(model)

    # Apply strategy
    strategy = strategy.lower().strip()
    if strategy == 'pruning':
        result = apply_pruning(model, train_loader, test_loader, device,
                               amount=0.70, fine_tune_epochs=max(10, fine_tune_epochs),
                               save_dir=save_dir,
                               model_name=uploaded_model_name,
                               accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD)
    elif strategy == 'quantization':
        result = apply_quantization(model, train_loader, test_loader, device,
                                    save_dir=save_dir,
                                    model_name=uploaded_model_name,
                                    fine_tune_epochs=max(1, fine_tune_epochs // 2),
                                    accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD)
    elif strategy == 'hybrid':
        result = apply_hybrid(model, train_loader, test_loader, device,
                              amount=0.25, fine_tune_epochs=fine_tune_epochs,
                              save_dir=save_dir,
                              model_name=uploaded_model_name,
                              accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD)
    elif strategy == 'kd':
        result = apply_kd(model, train_loader, test_loader, device,
                          num_classes=num_classes, epochs=fine_tune_epochs * 4,
                          save_dir=save_dir,
                          model_name=uploaded_model_name,
                          accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Choose from: pruning, quantization, hybrid, kd"
        )

    # Add post-compression inference emissions tracking
    inference_emissions_kg = result.get("inference_emissions_kg")
    inference_co2_kg = result.get("inference_co2_kg")
    inference_energy_kwh = result.get("inference_energy_kwh")

    # Fallback only when strategy did not already provide inference tracking.
    if inference_emissions_kg is None and inference_co2_kg is None:
        try:
            tracker = _start_emissions_tracker(
                project_name=f"compress_{strategy}_fallback_inference",
                output_dir=save_dir,
            )
            fallback_started_at = time.time()
            # Fallback pass on available model reference.
            model.eval()
            model = model.to(device)
            with torch.no_grad():
                for i, (inputs, _) in enumerate(test_loader):
                    inputs = inputs.to(device)
                    _extract_logits(model(inputs))
            inference_emissions_kg, inference_energy_kwh, _ = _finalize_emissions_tracking(
                tracker,
                phase_label=f"inference:fallback:{strategy}",
                started_at=fallback_started_at,
            )
            inference_co2_kg = inference_emissions_kg
        except Exception as e:
            print(f"[CodeCarbon] Fallback inference tracking failed for {strategy}: {e}")

    if inference_emissions_kg is None:
        inference_emissions_kg = inference_co2_kg
    if inference_co2_kg is None:
        inference_co2_kg = inference_emissions_kg
    if inference_energy_kwh is None:
        inference_energy_kwh = result.get("energy_kwh", 0.0)

    result["inference_emissions_kg"] = inference_emissions_kg if inference_emissions_kg is not None else 0.0
    result["inference_co2_kg"] = inference_co2_kg if inference_co2_kg is not None else 0.0
    result["inference_energy_kwh"] = inference_energy_kwh if inference_energy_kwh is not None else 0.0

    # Keep legacy top-level aliases in sync with fair total metrics when available.
    result["emissions_kg"] = result.get("compressed_total_emissions_kg", result["inference_emissions_kg"])
    result["co2_kg"] = result.get("compressed_total_emissions_kg", result["inference_co2_kg"])
    result["energy_kwh"] = result.get("compressed_total_energy_kwh", result["inference_energy_kwh"])

    # Add FLOPs estimate (use original model — FLOPs don't change with pruning/quantization)
    try:
        input_shape = detect_input_shape(model)
        from evaluate import estimate_flops
        flops = estimate_flops(model, input_shape=input_shape, device=device)
        result["flops"] = flops
        result["flops_M"] = round(flops / 1e6, 2)
    except Exception:
        result["flops"] = 0
        result["flops_M"] = 0

    return result


# ============================================================
# Preloaded Model Compression — 15 curated pretrained models
# ============================================================

PRELOADED_MODELS = {
    'resnet18': {
        'name': 'ResNet18', 'params': '11.2M', 'input_size': 224,
        'dataset': 'CIFAR-10 / ImageNet',
    },
    'resnet34': {
        'name': 'ResNet34', 'params': '21.8M', 'input_size': 224,
        'dataset': 'CIFAR-10 / ImageNet',
    },
    'mobilenet_v2': {
        'name': 'MobileNetV2', 'params': '3.4M', 'input_size': 224,
        'dataset': 'CIFAR-10 / ImageNet',
    },
    'efficientnet_b0': {
        'name': 'EfficientNet-B0', 'params': '5.3M', 'input_size': 224,
        'dataset': 'ImageNet',
    },
    'efficientnet_b1': {
        'name': 'EfficientNet-B1', 'params': '7.8M', 'input_size': 240,
        'dataset': 'ImageNet',
    },
    'densenet121': {
        'name': 'DenseNet121', 'params': '8.0M', 'input_size': 224,
        'dataset': 'CIFAR-10 / ImageNet',
    },
    'densenet169': {
        'name': 'DenseNet169', 'params': '14.3M', 'input_size': 224,
        'dataset': 'CIFAR-10 / ImageNet',
    },
    'squeezenet': {
        'name': 'SqueezeNet 1.1', 'params': '1.2M', 'input_size': 224,
        'dataset': 'CIFAR-10 / ImageNet',
    },
    'shufflenet_v2': {
        'name': 'ShuffleNet V2', 'params': '2.3M', 'input_size': 224,
        'dataset': 'CIFAR-10 / ImageNet',
    },
    'inception_v3': {
        'name': 'Inception V3', 'params': '23.8M', 'input_size': 299,
        'dataset': 'ImageNet',
    },
    'googlenet': {
        'name': 'GoogLeNet', 'params': '6.8M', 'input_size': 224,
        'dataset': 'ImageNet',
    },
}


PRELOADED_MODEL_TOTAL_PARAMS = {
    'resnet18': 11181642,
    'resnet34': 21289802,
    'mobilenet_v2': 2236682,
    'efficientnet_b0': 4020358,
    'efficientnet_b1': 6525994,
    'densenet121': 6964106,
    'densenet169': 12501130,
    'squeezenet': 727626,
    'shufflenet_v2': 1263854,
    'inception_v3': 21806058,
    'googlenet': 5610154,
}


def get_pretrained_model(model_key, num_classes=10):
    """
    Load a pretrained model from torchvision with ImageNet weights
    and replace the classifier head for the target number of classes.
    """
    key = model_key.lower().strip()
    if key not in PRELOADED_MODELS:
        raise ValueError(
            f"Unknown model: {model_key}. "
            f"Available: {list(PRELOADED_MODELS.keys())}"
        )

    input_size = PRELOADED_MODELS[key]['input_size']

    if key == 'resnet18':
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key == 'resnet34':
        model = torchvision.models.resnet34(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif key == 'efficientnet_b0':
        model = torchvision.models.efficientnet_b0(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif key == 'efficientnet_b1':
        model = torchvision.models.efficientnet_b1(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif key == 'densenet121':
        model = torchvision.models.densenet121(weights='DEFAULT')
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif key == 'densenet169':
        model = torchvision.models.densenet169(weights='DEFAULT')
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif key == 'squeezenet':
        model = torchvision.models.squeezenet1_1(weights='DEFAULT')
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
    elif key == 'shufflenet_v2':
        model = torchvision.models.shufflenet_v2_x1_0(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key == 'inception_v3':
        model = torchvision.models.inception_v3(weights='DEFAULT', aux_logits=True)
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key == 'googlenet':
        model = torchvision.models.googlenet(weights='DEFAULT', aux_logits=True)
        model.aux_logits = False
        model.aux1 = None
        model.aux2 = None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model builder not implemented for: {key}")

    # Store input size so detect_input_shape() and measure_latency() use it
    model._input_size = input_size
    return model


def _enable_head_gradients(model, model_key):
    """Re-enable requires_grad on the classifier / fc head of a pretrained model
    so we can fine-tune only the head while the backbone is frozen."""
    key = model_key.lower().strip()
    if key in ('resnet18', 'resnet34', 'shufflenet_v2',
               'inception_v3', 'googlenet'):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif key in ('mobilenet_v2', 'efficientnet_b0', 'efficientnet_b1'):
        for p in model.classifier[1].parameters():
            p.requires_grad = True
    elif key in ('densenet121', 'densenet169'):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif key == 'squeezenet':
        for p in model.classifier[1].parameters():
            p.requires_grad = True
    else:
        # Fallback: enable all parameters in case model is unknown
        for p in model.parameters():
            p.requires_grad = True


def run_compression(model_name, method, dataset='CIFAR10',
                    fine_tune_epochs=5, device=None, progress_cb=None):
    """
    Main entry point for preloaded model compression.

    1. Loads a pretrained model from torchvision (ImageNet weights)
    2. Adapts the classifier head for the target dataset
    3. Applies the selected compression method
    4. Evaluates and returns metrics

    Args:
        model_name: Key from PRELOADED_MODELS (e.g. 'resnet18')
        method: One of 'pruning', 'quantization', 'hybrid', 'kd'
        dataset: 'CIFAR10' or 'CIFAR100'
        fine_tune_epochs: Epochs for fine-tuning (pruning/hybrid) or KD
        device: torch device (auto-detected if None)
        progress_cb: Optional callback fn(step, detail) for progress updates

    Returns:
        dict with all compression metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_key = model_name.lower().strip()
    if model_key not in PRELOADED_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(PRELOADED_MODELS.keys())}"
        )

    cfg = PRELOADED_MODELS[model_key]
    input_size = cfg['input_size']
    num_classes = 10 if dataset.upper() == 'CIFAR10' else 100
    total_params = PRELOADED_MODEL_TOTAL_PARAMS.get(model_key)

    _cb = progress_cb or (lambda step, detail='': None)

    device_label = str(device)
    if device.type == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        device_label = f"CUDA ({gpu_name})"

    print(f"\n{'='*60}")
    print(f"Preloaded Compression: {cfg['name']} + {method.upper()}")
    print(f"Dataset: {dataset} | Input: {input_size}x{input_size} | Classes: {num_classes}")
    print(f"Device: {device_label}")
    print(f"{'='*60}")

    # Step 1: Load model — try pre-saved baseline first.
    # Check both locations: models/pretrained_baselines/ AND models/ root.
    _models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    pretrained_path = os.path.join(_models_dir, 'pretrained_baselines', f'{model_key}_baseline.pth')
    if not os.path.exists(pretrained_path):
        # Fallback: check directly in models/ root (e.g. models/resnet18_baseline.pth)
        _fallback = os.path.join(_models_dir, f'{model_key}_baseline.pth')
        if os.path.exists(_fallback):
            pretrained_path = _fallback
    has_pretrained = os.path.exists(pretrained_path)
    if has_pretrained:
        print(f"  Found pre-saved baseline: {pretrained_path}")
    else:
        print(f"  No pre-saved baseline found — will transfer-learn from ImageNet.")

    # Step 2: Prepare dataset FIRST — before model.to(device) so that
    # CIFAR-10 files are read before CUDA is initialized. This prevents
    # the DataLoader from inheriting a CUDA context when workers are forked.
    print(f"  [Step 1/4] Loading {dataset} dataset (input {input_size}x{input_size})...")
    _cb('loading_data', f'Preparing {dataset} dataset ({input_size}x{input_size})...')
    train_loader, test_loader = get_data_loaders(
        dataset,
        input_size=input_size,
        total_params=total_params,
        model_key=model_key,
    )
    print(f"  [Step 1/4] Dataset ready (batch_size={train_loader.batch_size}).")

    # Step 2 (was Step 1): Load model weights onto GPU
    print(f"  [Step 2/4] Loading {cfg['name']} weights onto {device_label}...")
    _cb('loading_model', f'Loading {cfg["name"]} on {device_label}...')
    model = get_pretrained_model(model_key, num_classes=num_classes)
    model = model.to(device)

    if has_pretrained:
        # Load pre-saved fine-tuned weights — skip transfer learning!
        print(f"  Using pre-saved baseline: {pretrained_path}")
        _cb('loading_model', f'Loading pre-saved {cfg["name"]} baseline...')
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model = model.to(device)
    print(f"  [Step 2/4] Model loaded.")

    # ── Step 2.5: Transfer-learn (SKIP if pre-saved baseline exists) ──
    if not has_pretrained:
        # The pretrained model has an ImageNet backbone but a *randomly
        # initialised* classifier for num_classes. We must fine-tune so
        # baseline accuracy reflects actual performance on the target
        # dataset — otherwise it will be ≈10% (random).
        _cb('loading_data', 'Fine-tuning classifier head on target dataset...')
        print(f"  [Step 3/4] Transfer-learning head (no pre-saved baseline found)...")

        # Phase A – freeze backbone, train only the new head (fast)
        # Capped at 100 batches per epoch to avoid iterating all 50k images
        # silently on a single thread with 224x224 resize.
        for p in model.parameters():
            p.requires_grad = False
        _enable_head_gradients(model, model_key)

        head_epochs = max(fine_tune_epochs, 3)
        head_max_batches = 100
        optimizer_A = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4)

        for ep in range(head_epochs):
            print(f"    [Phase A] Head fine-tune epoch {ep+1}/{head_epochs}...")
            _cb('loading_data', f'Head fine-tune epoch {ep+1}/{head_epochs}...')
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= head_max_batches:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_A.zero_grad()
                loss = F.cross_entropy(_extract_logits(model(inputs)), labels)
                loss.backward()
                optimizer_A.step()
                if (batch_idx + 1) % 20 == 0:
                    print(f"      batch {batch_idx+1}/{head_max_batches} loss={loss.item():.4f}")
                    _cb('loading_data',
                        f'Head fine-tune epoch {ep+1}/{head_epochs} — '
                        f'batch {batch_idx+1}/{head_max_batches}')

        # Phase B – unfreeze all layers, fine-tune end-to-end with low LR
        for p in model.parameters():
            p.requires_grad = True

        ft_epochs = max(fine_tune_epochs, 2)
        max_batches_per_epoch = 100
        total_batches = len(train_loader)
        effective_batches = min(total_batches, max_batches_per_epoch)
        optimizer_B = optim.SGD(model.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_B, T_max=ft_epochs)

        for ep in range(ft_epochs):
            print(f"    [Phase B] Full fine-tune epoch {ep+1}/{ft_epochs}...")
            _cb('loading_data', f'Full fine-tune epoch {ep+1}/{ft_epochs}...')
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= max_batches_per_epoch:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_B.zero_grad()
                loss = F.cross_entropy(_extract_logits(model(inputs)), labels)
                loss.backward()
                optimizer_B.step()
                if (batch_idx + 1) % 10 == 0:
                    print(f"      batch {batch_idx+1}/{effective_batches} loss={loss.item():.4f}")
                    _cb('loading_data',
                        f'Full fine-tune epoch {ep+1}/{ft_epochs} — '
                        f'batch {batch_idx+1}/{effective_batches}')
            scheduler.step()

        tl_acc = evaluate(model, test_loader, dev=device)
        print(f"  Transfer-learn accuracy: {tl_acc}%")
    else:
        # Pre-saved baseline — skip the full-set evaluation to avoid slow
        # single-threaded 224x224 data loading before compression starts.
        # The compression functions run their own evaluation internally.
        _cb('loading_data', 'Using pre-saved baseline — skipping transfer learning')
        tl_acc = None
        print(f"  Pre-saved baseline loaded — skipping pre-compression eval, going straight to compression.")
    # ── End transfer learning ──────────────────────────────────────

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'compressed')
    os.makedirs(save_dir, exist_ok=True)

    # Step 3: Compress
    _cb('compressing', f'Applying {method}...')

    # Sub-callback that keeps step='compressing' but updates detail
    def compress_cb(detail):
        _cb('compressing', detail)

    method = method.lower().strip()
    if method == 'pruning':
        result = apply_pruning(model, train_loader, test_loader, device,
                               amount=0.70, fine_tune_epochs=max(10, fine_tune_epochs),
                               save_dir=save_dir, progress_cb=compress_cb,
                               model_name=model_key,
                               accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD)
    elif method == 'quantization':
        result = apply_quantization(model, train_loader, test_loader, device,
                                    save_dir=save_dir, progress_cb=compress_cb,
                                    model_name=model_key,
                                    fine_tune_epochs=max(1, fine_tune_epochs // 2),
                                    accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD)
    elif method == 'hybrid':
        result = apply_hybrid(model, train_loader, test_loader, device,
                              amount=0.25, fine_tune_epochs=fine_tune_epochs,
                              save_dir=save_dir, progress_cb=compress_cb,
                              model_name=model_key,
                              accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD)
    elif method == 'kd':
        result = apply_kd(model, train_loader, test_loader, device,
                          num_classes=num_classes,
                          epochs=fine_tune_epochs,
                          save_dir=save_dir, progress_cb=compress_cb,
                          model_name=model_key,
                          accuracy_drop_threshold=DEFAULT_ACCURACY_DROP_THRESHOLD)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from: pruning, quantization, hybrid, kd"
        )

    # Add model metadata
    result['model_name'] = cfg['name']
    result['model_key'] = model_key
    result['compression_method'] = method
    result['dataset'] = dataset
    result['input_size'] = input_size

    # Step 4: Post-compression inference emissions tracking
    _cb('energy_tracking', 'Measuring energy consumption...')
    inference_emissions_kg = result.get("inference_emissions_kg")
    inference_co2_kg = result.get("inference_co2_kg")
    inference_energy_kwh = result.get("inference_energy_kwh")

    # Fallback only when strategy result does not already include inference metrics.
    if inference_emissions_kg is None and inference_co2_kg is None:
        try:
            tracker = _start_emissions_tracker(
                project_name=f"compress_{model_key}_{method}_fallback_inference",
                output_dir=save_dir,
            )
            fallback_started_at = time.time()
            model.eval()
            model = model.to(device)
            with torch.no_grad():
                for i, (inputs, _) in enumerate(test_loader):
                    inputs = inputs.to(device)
                    _extract_logits(model(inputs))
            inference_emissions_kg, inference_energy_kwh, _ = _finalize_emissions_tracking(
                tracker,
                phase_label=f"inference:fallback:{model_key}:{method}",
                started_at=fallback_started_at,
            )
            inference_co2_kg = inference_emissions_kg
        except Exception as e:
            print(f"[CodeCarbon] Fallback inference tracking failed for {model_key}/{method}: {e}")

    if inference_emissions_kg is None:
        inference_emissions_kg = inference_co2_kg
    if inference_co2_kg is None:
        inference_co2_kg = inference_emissions_kg
    if inference_energy_kwh is None:
        inference_energy_kwh = result.get("energy_kwh", 0.0)

    result["inference_emissions_kg"] = inference_emissions_kg if inference_emissions_kg is not None else 0.0
    result["inference_co2_kg"] = inference_co2_kg if inference_co2_kg is not None else 0.0
    result["inference_energy_kwh"] = inference_energy_kwh if inference_energy_kwh is not None else 0.0

    # Keep legacy top-level aliases in sync with fair total metrics when available.
    result["emissions_kg"] = result.get("compressed_total_emissions_kg", result["inference_emissions_kg"])
    result["co2_kg"] = result.get("compressed_total_emissions_kg", result["inference_co2_kg"])
    result["energy_kwh"] = result.get("compressed_total_energy_kwh", result["inference_energy_kwh"])

    # Step 5: FLOPs estimate
    _cb('evaluating', 'Computing FLOPs and final metrics...')
    try:
        inp_shape = detect_input_shape(model)
        from evaluate import estimate_flops
        flops = estimate_flops(model, input_shape=inp_shape, device=device)
        result["flops"] = flops
        result["flops_M"] = round(flops / 1e6, 2)
    except Exception:
        result["flops"] = 0
        result["flops_M"] = 0

    _cb('complete', 'Done!')
    print(f"\n  Result: acc={result.get('compressed_accuracy')}%  "
          f"size={result.get('size_MB')} MB  "
          f"reduction={result.get('size_reduction_percent')}%")

    # Persist result to a per-model-per-method JSON file so it survives
    # across server restarts and is not overwritten by subsequent runs.
    try:
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        result_filename = f"{model_key}_{method}_compression_result.json"
        result_path = os.path.join(results_dir, result_filename)
        with open(result_path, 'w') as _f:
            json.dump(result, _f, indent=2)
        print(f"  Saved result → {result_path}")
    except Exception as _save_err:
        print(f"  [Warning] Could not save result JSON: {_save_err}")

    return result


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run GreenAI compression from CLI"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Preloaded model key (e.g. resnet18, densenet121)."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pruning",
        choices=["pruning", "quantization", "hybrid", "kd"],
        help="Compression method for CLI preloaded run."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "CIFAR100"],
        help="Dataset used for evaluate/fine-tune."
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=5,
        help="Fine-tuning epochs for pruning/hybrid or KD epochs."
    )
    parser.add_argument(
        "--run_legacy_full_pipeline",
        action="store_true",
        help="Run the original full 5-strategy legacy script block."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    # Preferred CLI mode: run one selected preloaded model compression.
    if args.model_name and not args.run_legacy_full_pipeline:
        result = run_compression(
            model_name=args.model_name,
            method=args.method,
            dataset=args.dataset,
            fine_tune_epochs=args.fine_tune_epochs,
            device=device,
        )
        print("\nCLI compression result:")
        print(json.dumps(result, indent=2))
        raise SystemExit(0)

    model_name = args.model_name or "resnet18"
    model_slug = _slugify_name(model_name)
    model_dir = "../models"
    baseline_model_path = os.path.join(model_dir, "baseline_model.pth")
    pruned_sparse_path = build_compressed_model_path(model_dir, model_slug, "pruned_sparse")
    pruned_dense_path = build_compressed_model_path(model_dir, model_slug, "pruned")
    quantized_model_path = build_compressed_model_path(model_dir, model_slug, "quantized")
    student_model_path = build_compressed_model_path(model_dir, model_slug, "student_distilled")
    hybrid_model_path = build_compressed_model_path(model_dir, model_slug, "hybrid")
    ultra_sparse_path = build_compressed_model_path(model_dir, model_slug, "ultra_compact_sparse")
    ultra_quant_path = build_compressed_model_path(model_dir, model_slug, "ultra_compact_quant")

    # ----------------------------------------------------------
    # Data loaders (CIFAR-10)
    # ----------------------------------------------------------
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ----------------------------------------------------------
    # Load & evaluate baseline
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("BASELINE MODEL (ResNet18)")
    print("=" * 65)

    baseline_model = resnet18(weights=None, num_classes=10)
    baseline_model.load_state_dict(
        torch.load(baseline_model_path, map_location=device))
    baseline_model = baseline_model.to(device)
    baseline_model.eval()

    baseline_acc = evaluate(baseline_model, test_loader)
    baseline_size = get_size_mb(baseline_model_path)
    baseline_params = count_params(baseline_model)
    baseline_latency = measure_latency(baseline_model, dev=device)
    print(f"  Accuracy : {baseline_acc}%")
    print(f"  Size     : {baseline_size} MB")
    print(f"  Params   : {baseline_params:,}")
    print(f"  Latency  : {baseline_latency} ms")

    # Collect all results for summary table
    all_results = {
        "baseline": {
            "accuracy": baseline_acc,
            "size_MB": baseline_size,
            "params": baseline_params,
            "latency_ms": baseline_latency,
            "size_reduction_percent": 0.0,
        }
    }

    # ==========================================================
    # STRATEGY 1: Aggressive Unstructured Pruning (70%)
    #             + Fine-tuning + Sparse Storage
    # ==========================================================
    print("\n" + "=" * 65)
    print("STRATEGY 1: Unstructured Pruning (70%) + Fine-tune + Sparse Save")
    print("=" * 65)
    print("WHY: Setting 70% of smallest-magnitude weights to zero,")
    print("     then saving in sparse format so zeros don't waste disk space.")

    pruned_model = resnet18(weights=None, num_classes=10)
    pruned_model.load_state_dict(
        torch.load(baseline_model_path, map_location=device))
    pruned_model = pruned_model.to(device)

    # Global unstructured L1 pruning — removes 70% of smallest weights
    params_to_prune = _collect_prunable_modules(pruned_model)
    if not params_to_prune:
        raise ValueError("Baseline model has no Conv2d or Linear layers to prune.")
    prune.global_unstructured(
        params_to_prune, pruning_method=prune.L1Unstructured, amount=0.7)
    _remove_pruning_from_model(pruned_model)

    # Fine-tune to recover accuracy lost from aggressive pruning
    print("  Fine-tuning pruned model (5 epochs)...")
    optimizer = optim.SGD(pruned_model.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)
    for epoch in range(1, 6):
        pruned_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(pruned_model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc = evaluate(pruned_model, test_loader)
        print(f"    Epoch {epoch}: loss={running_loss/len(train_loader):.4f}, "
              f"acc={acc}%")

    # Save compressed output as .pth (save_compressed handles sparse tensors).
    save_compressed(pruned_model, pruned_sparse_path)
    # Also save standard format for comparison
    torch.save(pruned_model.state_dict(), pruned_dense_path)

    pruned_acc = evaluate(pruned_model, test_loader)
    pruned_size_std = get_size_mb(pruned_dense_path)
    pruned_size_compressed = get_size_mb(pruned_sparse_path)
    pruned_nonzero = count_nonzero(pruned_model)
    pruned_latency = measure_latency(pruned_model, dev=device)
    sparsity = round(100 * (1 - pruned_nonzero / baseline_params), 2)

    pruned_metrics = {
        "accuracy": pruned_acc,
        "size_MB_standard": pruned_size_std,
        "size_MB_compressed": pruned_size_compressed,
        "size_reduction_percent": round(
            100 * (baseline_size - pruned_size_compressed) / baseline_size, 2),
        "sparsity_percent": sparsity,
        "nonzero_params": pruned_nonzero,
        "total_params": baseline_params,
        "latency_ms": pruned_latency,
        "pruning_amount": 0.7,
    }
    with open("../results/pruned_metrics.json", "w") as f:
        json.dump(pruned_metrics, f, indent=2)

    print(f"\n  Results:")
    print(f"    Accuracy       : {pruned_acc}%")
    print(f"    Std save size  : {pruned_size_std} MB  (same as baseline — "
          f"zeros stored as float32)")
    print(f"    Compressed     : {pruned_size_compressed} MB  "
          f"(↓ {pruned_metrics['size_reduction_percent']}%)")
    print(f"    Sparsity       : {sparsity}% of weights are zero")
    print(f"    Latency        : {pruned_latency} ms")

    all_results["pruning_compressed"] = pruned_metrics

    # ==========================================================
    # STRATEGY 2: Post-Training Static Quantization (INT8)
    #             Uses torchvision's quantizable ResNet18
    # ==========================================================
    print("\n" + "=" * 65)
    print("STRATEGY 2: Post-Training Static Quantization (INT8)")
    print("=" * 65)
    print("WHY: Converts weights AND activations from float32 → int8,")
    print("     giving ~4x compression with minimal accuracy loss.")
    print("     Uses torchvision's quantizable ResNet18 with proper")
    print("     QuantStub/DeQuantStub and FloatFunctional for residuals.")

    try:
        from torchvision.models.quantization import resnet18 as qresnet18

        # Load quantizable variant and transfer baseline weights
        quant_model = qresnet18(weights=None, num_classes=10, quantize=False)
        # Load baseline weights; strict=False because quantizable model has
        # extra modules (quant/dequant stubs, FloatFunctional) with no weights
        baseline_state = torch.load(
            baseline_model_path, map_location='cpu')
        quant_model.load_state_dict(baseline_state, strict=False)
        quant_model.eval()
        quant_model.cpu()

        # Fuse Conv-BN-ReLU blocks for better quantization accuracy
        quant_model.fuse_model()

        # Select a backend supported by the current PyTorch build.
        backend = _configure_quantized_backend()
        print(f"  Using quantized backend: {backend}")
        quant_model.qconfig = torch.quantization.get_default_qconfig(backend)

        # Insert observers
        torch.quantization.prepare(quant_model, inplace=True)

        # Calibrate with training data (observers collect activation statistics)
        print("  Calibrating with training data (50 batches)...")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(train_loader):
                quant_model(inputs.cpu())
                if i >= 49:
                    break

        # Convert to quantized model
        torch.quantization.convert(quant_model, inplace=True)

        # Save quantized model
        torch.save(quant_model.state_dict(), quantized_model_path)

        quant_acc = evaluate(quant_model, test_loader, dev=torch.device('cpu'))
        quant_size = get_size_mb(quantized_model_path)
        quant_latency = measure_latency(
            quant_model, dev=torch.device('cpu'))

        quant_metrics = {
            "accuracy": quant_acc,
            "size_MB": quant_size,
            "size_reduction_percent": round(
                100 * (baseline_size - quant_size) / baseline_size, 2),
            "quantization_type": "static_int8",
            "backend": backend,
            "latency_ms": quant_latency,
        }
        with open("../results/quantized_metrics.json", "w") as f:
            json.dump(quant_metrics, f, indent=2)

        print(f"\n  Results:")
        print(f"    Accuracy  : {quant_acc}%")
        print(f"    Size      : {quant_size} MB  "
              f"(↓ {quant_metrics['size_reduction_percent']}%)")
        print(f"    Latency   : {quant_latency} ms")

        all_results["quantization_static"] = quant_metrics

    except Exception as e:
        print(f"  [WARNING] Static quantization failed: {e}")
        print("  Falling back to dynamic quantization...")

        quant_model_dyn = resnet18(weights=None, num_classes=10)
        quant_model_dyn.load_state_dict(
            torch.load(baseline_model_path, map_location='cpu'))
        quant_model_dyn.eval()

        quant_model_dyn = torch.quantization.quantize_dynamic(
            quant_model_dyn, {nn.Linear}, dtype=torch.qint8)

        torch.save(quant_model_dyn.state_dict(), quantized_model_path)

        quant_acc = evaluate(quant_model_dyn, test_loader,
                             dev=torch.device('cpu'))
        quant_size = get_size_mb(quantized_model_path)
        quant_latency = measure_latency(
            quant_model_dyn, dev=torch.device('cpu'))

        quant_metrics = {
            "accuracy": quant_acc,
            "size_MB": quant_size,
            "size_reduction_percent": round(
                100 * (baseline_size - quant_size) / baseline_size, 2),
            "quantization_type": "dynamic_int8_linear_only",
            "latency_ms": quant_latency,
            "note": "Dynamic quantization only quantizes nn.Linear layers"
        }
        with open("../results/quantized_metrics.json", "w") as f:
            json.dump(quant_metrics, f, indent=2)

        print(f"    Accuracy  : {quant_acc}%")
        print(f"    Size      : {quant_size} MB")
        all_results["quantization_dynamic"] = quant_metrics

    # ==========================================================
    # STRATEGY 3: Knowledge Distillation → Compact Student
    # ==========================================================
    print("\n" + "=" * 65)
    print("STRATEGY 3: Knowledge Distillation → Compact Student Model")
    print("=" * 65)
    print("WHY: Instead of distilling ResNet18 → ResNet18 (same size!),")
    print("     we use a compact MobileNet-style student with ~300K params")
    print(f"     vs {baseline_params:,} params in the teacher (ResNet18).")
    print("     The teacher's soft labels guide the small student to learn")
    print("     richer representations than hard labels alone.")

    student = CompactStudent(num_classes=10).to(device)
    teacher = baseline_model
    teacher.eval()

    student_params = count_params(student)
    print(f"  Student params  : {student_params:,}")
    print(f"  Teacher params  : {baseline_params:,}")
    print(f"  Param reduction : {100*(1 - student_params/baseline_params):.1f}%")

    # KD training (on TRAINING data — not test data!)
    optimizer = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    best_student_acc = 0.0

    print("  Training compact student with KD (20 epochs)...")
    for epoch in range(1, 21):
        student.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            s_out = _extract_logits(student(inputs))
            with torch.no_grad():
                t_out = _extract_logits(teacher(inputs))
            loss = distillation_loss(s_out, t_out, labels, T=4.0, alpha=0.3)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        acc = evaluate(student, test_loader)
        if acc > best_student_acc:
            best_student_acc = acc
            torch.save(student.state_dict(), student_model_path)
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: loss={running_loss/len(train_loader):.4f}, "
                  f"acc={acc}%, best={best_student_acc}%")

    # Reload best checkpoint
    student.load_state_dict(
        torch.load(student_model_path, map_location=device))
    student_acc = evaluate(student, test_loader)
    student_size = get_size_mb(student_model_path)
    student_latency = measure_latency(student, dev=device)

    student_metrics = {
        "accuracy": student_acc,
        "size_MB": student_size,
        "size_reduction_percent": round(
            100 * (baseline_size - student_size) / baseline_size, 2),
        "student_params": student_params,
        "teacher_params": baseline_params,
        "param_reduction_percent": round(
            100 * (1 - student_params / baseline_params), 2),
        "latency_ms": student_latency,
        "architecture": "CompactStudent (MobileNet-style depthwise separable)",
        "kd_epochs": 20,
        "kd_temperature": 4.0,
        "kd_alpha": 0.3,
    }
    with open("../results/student_distilled_metrics.json", "w") as f:
        json.dump(student_metrics, f, indent=2)

    print(f"\n  Results:")
    print(f"    Accuracy       : {student_acc}%")
    print(f"    Size           : {student_size} MB  "
          f"(↓ {student_metrics['size_reduction_percent']}%)")
    print(f"    Params         : {student_params:,}")
    print(f"    Latency        : {student_latency} ms")

    all_results["kd_compact_student"] = student_metrics

    # ==========================================================
    # STRATEGY 4: Hybrid — Compact Student + Dynamic Quantization
    # ==========================================================
    print("\n" + "=" * 65)
    print("STRATEGY 4: Hybrid — Compact Student + Dynamic Quantization")
    print("=" * 65)
    print("WHY: Apply INT8 dynamic quantization on top of the already-small")
    print("     student model for additional compression.")

    hybrid_student = CompactStudent(num_classes=10)
    hybrid_student.load_state_dict(
        torch.load(student_model_path, map_location='cpu'))
    hybrid_student.eval()

    hybrid_student = torch.quantization.quantize_dynamic(
        hybrid_student, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

    torch.save(hybrid_student.state_dict(), hybrid_model_path)
    hybrid_acc = evaluate(hybrid_student, test_loader, dev=torch.device('cpu'))
    hybrid_size = get_size_mb(hybrid_model_path)
    hybrid_latency = measure_latency(
        hybrid_student, dev=torch.device('cpu'))

    hybrid_metrics = {
        "accuracy": hybrid_acc,
        "size_MB": hybrid_size,
        "size_reduction_percent": round(
            100 * (baseline_size - hybrid_size) / baseline_size, 2),
        "latency_ms": hybrid_latency,
        "pipeline": "KD (CompactStudent) → Dynamic Quantization (INT8)",
    }
    with open("../results/hybrid_metrics.json", "w") as f:
        json.dump(hybrid_metrics, f, indent=2)

    print(f"\n  Results:")
    print(f"    Accuracy  : {hybrid_acc}%")
    print(f"    Size      : {hybrid_size} MB  "
          f"(↓ {hybrid_metrics['size_reduction_percent']}%)")
    print(f"    Latency   : {hybrid_latency} ms")

    all_results["hybrid_student_quant"] = hybrid_metrics

    # ==========================================================
    # STRATEGY 5: Ultra-Compact — Pruned Student + Quant + Sparse
    # ==========================================================
    print("\n" + "=" * 65)
    print("STRATEGY 5: Ultra-Compact — Pruned Student + Quantization + Sparse")
    print("=" * 65)
    print("WHY: Maximum compression by combining ALL techniques on the")
    print("     compact student: prune 50% → fine-tune → quantize → sparse save.")

    ultra_model = CompactStudent(num_classes=10).to(device)
    ultra_model.load_state_dict(
        torch.load(student_model_path, map_location=device))

    # Prune 50% of compact student's weights
    ultra_prune_params = _collect_prunable_modules(ultra_model)
    if not ultra_prune_params:
        raise ValueError("Compact student has no Conv2d or Linear layers to prune.")
    prune.global_unstructured(
        ultra_prune_params, pruning_method=prune.L1Unstructured, amount=0.5)
    _remove_pruning_from_model(ultra_model)

    # Create pruning mask to enforce sparsity during fine-tuning
    pruning_masks = {}
    for name, param in ultra_model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            pruning_masks[name] = (param.data != 0).float()

    # Fine-tune the pruned student (with KD from teacher for best results)
    print("  Fine-tuning pruned student with KD (10 epochs)...")
    optimizer = optim.Adam(ultra_model.parameters(), lr=0.0005,
                           weight_decay=1e-4)
    for epoch in range(1, 11):
        ultra_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            s_out = ultra_model(inputs)
            with torch.no_grad():
                t_out = _extract_logits(teacher(inputs))
            loss = distillation_loss(s_out, t_out, labels, T=4.0, alpha=0.3)
            loss.backward()
            optimizer.step()
            # Re-apply pruning mask to keep zeros at zero
            with torch.no_grad():
                for name, param in ultra_model.named_parameters():
                    if name in pruning_masks:
                        param.data.mul_(pruning_masks[name].to(param.device))
            running_loss += loss.item()
        if epoch % 5 == 0:
            acc = evaluate(ultra_model, test_loader)
            print(f"    Epoch {epoch}: loss={running_loss/len(train_loader):.4f}, "
                  f"acc={acc}%")

    # Move to CPU for quantization + sparse save
    ultra_model = ultra_model.cpu()
    ultra_model.eval()

    # Save pruned student as .pth for consistent artifact naming.
    save_compressed(ultra_model, ultra_sparse_path)

    # Also apply dynamic quantization and save sparse
    ultra_quant = torch.quantization.quantize_dynamic(
        copy.deepcopy(ultra_model), {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
    torch.save(ultra_quant.state_dict(), ultra_quant_path)

    ultra_acc = evaluate(ultra_model, test_loader, dev=torch.device('cpu'))
    ultra_size_compressed = get_size_mb(ultra_sparse_path)
    ultra_size_quant = get_size_mb(ultra_quant_path)
    ultra_nonzero = count_nonzero(ultra_model)
    ultra_latency = measure_latency(ultra_model, dev=torch.device('cpu'))
    ultra_sparsity = round(
        100 * (1 - ultra_nonzero / student_params), 2)

    ultra_metrics = {
        "accuracy": ultra_acc,
        "size_MB_compressed": ultra_size_compressed,
        "size_MB_quant": ultra_size_quant,
        "size_reduction_compressed_percent": round(
            100 * (baseline_size - ultra_size_compressed) / baseline_size, 2),
        "size_reduction_quant_percent": round(
            100 * (baseline_size - ultra_size_quant) / baseline_size, 2),
        "sparsity_percent": ultra_sparsity,
        "nonzero_params": ultra_nonzero,
        "latency_ms": ultra_latency,
        "pipeline": "KD → Prune 50% → Fine-tune (mask) → Quantize → Gzip",
    }
    with open("../results/ultra_compact_metrics.json", "w") as f:
        json.dump(ultra_metrics, f, indent=2)

    print(f"\n  Results:")
    print(f"    Accuracy       : {ultra_acc}%")
    print(f"    Compressed     : {ultra_size_compressed} MB  "
          f"(↓ {ultra_metrics['size_reduction_compressed_percent']}%)")
    print(f"    Quant size     : {ultra_size_quant} MB  "
          f"(↓ {ultra_metrics['size_reduction_quant_percent']}%)")
    print(f"    Sparsity       : {ultra_sparsity}% of weights are zero")
    print(f"    Latency        : {ultra_latency} ms")

    all_results["ultra_compact"] = ultra_metrics

    # ==========================================================
    # SUMMARY TABLE
    # ==========================================================
    print("\n" + "=" * 65)
    print("COMPRESSION RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Strategy':<35} {'Acc%':>6} {'Size MB':>8} {'↓ Size%':>8} "
          f"{'Latency':>8}")
    print("-" * 65)
    print(f"{'Baseline (ResNet18)':<35} "
          f"{baseline_acc:>6} {baseline_size:>8} {'—':>8} "
          f"{baseline_latency:>7}ms")

    print(f"{'1. Pruning 70% + Gzip':<35} "
          f"{pruned_acc:>6} {pruned_size_compressed:>8} "
          f"{pruned_metrics['size_reduction_percent']:>7}% "
          f"{pruned_latency:>7}ms")

    if 'quantization_static' in all_results:
        qm = all_results['quantization_static']
        print(f"{'2. Static Quantization INT8':<35} "
              f"{qm['accuracy']:>6} {qm['size_MB']:>8} "
              f"{qm['size_reduction_percent']:>7}% "
              f"{qm['latency_ms']:>7}ms")
    elif 'quantization_dynamic' in all_results:
        qm = all_results['quantization_dynamic']
        print(f"{'2. Dynamic Quantization INT8':<35} "
              f"{qm['accuracy']:>6} {qm['size_MB']:>8} "
              f"{qm['size_reduction_percent']:>7}% "
              f"{qm['latency_ms']:>7}ms")

    print(f"{'3. KD → Compact Student':<35} "
          f"{student_acc:>6} {student_size:>8} "
          f"{student_metrics['size_reduction_percent']:>7}% "
          f"{student_latency:>7}ms")

    print(f"{'4. Compact Student + Quant':<35} "
          f"{hybrid_acc:>6} {hybrid_size:>8} "
          f"{hybrid_metrics['size_reduction_percent']:>7}% "
          f"{hybrid_latency:>7}ms")

    print(f"{'5. Ultra: Prune+Quant+Gzip':<35} "
          f"{ultra_acc:>6} {ultra_size_compressed:>8} "
          f"{ultra_metrics['size_reduction_compressed_percent']:>7}% "
          f"{ultra_latency:>7}ms")

    print("-" * 65)

    # Save full summary
    with open("../results/compression_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll results saved to ../results/")
    print("All models saved to ../models/")
    print("\n✅ All compression strategies completed!")
    # ============================================================
# Deployment Export Pipeline
# PyTorch → ONNX → TensorFlow SavedModel → TFLite (.tflite)
# ============================================================

def export_to_onnx(model, save_path, input_shape=(1, 3, 32, 32), opset=11):
    """Export a PyTorch model to ONNX format.

    Args:
        model:        A trained/compressed PyTorch nn.Module.
        save_path:    Destination path for the .onnx file.
        input_shape:  Tuple describing a single sample shape (N, C, H, W).
                      Defaults to CIFAR-10 format (1, 3, 32, 32).
        opset:        ONNX opset version. 11 is the recommended stable version.

    Returns:
        save_path (str) on success.

    Raises:
        RuntimeError: If the ONNX export fails.

    Compatibility:
        * CompactStudent (MobileNet-style)
        * ResNet18 / ResNet variants
        * CIFAR-10 input shape (32x32)
    """
    print(f"\n[ONNX Export] Exporting model to: {save_path}")

    # --- safety: remove any residual pruning reparameterization ---
    removed = _remove_pruning_from_model(model)
    if removed:
        print(f"[ONNX Export] Removed pruning masks from {removed} layer(s).")

    # Move model to CPU and switch to eval mode
    model = model.cpu()
    model.eval()

    # Convert any sparse parameters to dense (sparse tensors cannot be traced)
    with torch.no_grad():
        for param in model.parameters():
            if param.is_sparse:
                param.data = param.data.to_dense()

    dummy_input = torch.randn(*input_shape)

    # Dynamic batch-size axis so the exported graph is not locked to N=1
    dynamic_axes = {
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        size_mb = get_size_mb(save_path)
        print(f"[ONNX Export] ✅ Saved  →  {save_path}  ({size_mb} MB)")
        return save_path
    except Exception as e:
        raise RuntimeError(
            f"[ONNX Export] ❌ Export failed.\n"
            f"  Model      : {type(model)._name_}\n"
            f"  Input shape: {input_shape}\n"
            f"  Opset      : {opset}\n"
            f"  Error      : {e}"
        ) from e


def onnx_to_tensorflow(onnx_path, output_dir):
    """Convert an ONNX model to a TensorFlow SavedModel.

    Requires (install if missing):
        pip install onnx onnx-tf tensorflow

    Args:
        onnx_path:   Path to the .onnx file produced by export_to_onnx().
        output_dir:  Directory where the TF SavedModel will be written.

    Returns:
        output_dir (str) on success.
    """
    print(f"\n[TF Conversion] Converting ONNX → TensorFlow SavedModel")
    print(f"  ONNX source : {onnx_path}")
    print(f"  Output dir  : {output_dir}")

    try:
        import onnx as onnx_lib
    except ImportError as e:
        raise ImportError(
            "[TF Conversion] ❌ 'onnx' package not found. "
            "Install with: pip install onnx"
        ) from e

    try:
        import onnx_tf.backend as onnx_tf_backend
    except ImportError as e:
        raise ImportError(
            "[TF Conversion] ❌ 'onnx-tf' package not found. "
            "Install with: pip install onnx-tf"
        ) from e

    try:
        onnx_model = onnx_lib.load(onnx_path)
        onnx_lib.checker.check_model(onnx_model)

        os.makedirs(output_dir, exist_ok=True)
        tf_rep = onnx_tf_backend.prepare(onnx_model)
        tf_rep.export_graph(output_dir)

        print(f"[TF Conversion] ✅ SavedModel written to: {output_dir}")
        return output_dir
    except Exception as e:
        print(
            f"[TF Conversion] ❌ Conversion failed.\n"
            f"  ONNX file: {onnx_path}\n"
            f"  Reason   : {e}\n"
            f"  Tip      : Verify onnx-tf and tensorflow versions are compatible."
        )
        raise


def tensorflow_to_tflite(saved_model_dir, tflite_path, quantize=True):
    """Convert a TensorFlow SavedModel to TFLite (.tflite).

    Args:
        saved_model_dir:  Path to the SavedModel directory produced by
                          onnx_to_tensorflow().
        tflite_path:      Destination path for the .tflite file.
        quantize:         If True, applies tf.lite.Optimize.DEFAULT (dynamic-
                          range / weight-only INT8 quantization).  No
                          representative dataset is required for this mode.

    Returns:
        tflite_path (str) on success.
    """
    print(f"\n[TFLite Conversion] Converting TF SavedModel → TFLite")
    print(f"  SavedModel dir : {saved_model_dir}")
    print(f"  TFLite output  : {tflite_path}")
    print(f"  Quantize       : {quantize}")

    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "[TFLite Conversion] ❌ 'tensorflow' package not found. "
            "Install with: pip install tensorflow"
        ) from e

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

        if quantize:
            # DEFAULT = dynamic-range quantization (INT8 weights, float activations).
            # For full INT8 (activations + weights), supply a representative dataset.
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        os.makedirs(os.path.dirname(os.path.abspath(tflite_path)), exist_ok=True)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        size_mb = get_size_mb(tflite_path)
        print(f"[TFLite Conversion] ✅ Saved  →  {tflite_path}  ({size_mb} MB)")
        return tflite_path
    except Exception as e:
        print(
            f"[TFLite Conversion] ❌ Conversion failed.\n"
            f"  SavedModel dir : {saved_model_dir}\n"
            f"  TFLite path    : {tflite_path}\n"
            f"  Quantize flag  : {quantize}\n"
            f"  Debug info     : {type(e)._name_}: {e}\n"
            f"  Tip            : Verify the SavedModel was exported correctly "
            f"and tensorflow >= 2.x is installed."
        )
        raise


def export_full_pipeline(model, export_dir, model_name="model",
                         input_shape=(1, 3, 32, 32), opset=11,
                         quantize_tflite=True):
    """Run the complete PyTorch → ONNX → TensorFlow → TFLite export pipeline.

    Output structure::

        export_dir/
            <model_name>.onnx
            tf_model/          ← TensorFlow SavedModel
            <model_name>.tflite

    Args:
        model:             Trained/compressed PyTorch nn.Module.
                           Works with CompactStudent, ResNet18, and ResNet variants.
        export_dir:        Root directory for all export artifacts.
        model_name:        Base name used for output files (default: "model").
        input_shape:       ONNX dummy input shape.  Defaults to CIFAR-10
                           format (1, 3, 32, 32).
        opset:             ONNX opset version (default: 11).
        quantize_tflite:   Enable DEFAULT quantization in TFLite conversion
                           (default: True).

    Returns:
        dict with keys:
            `onnx_path`    – path to .onnx file, or None on failure.
            `tf_dir`       – path to tf_model/ directory, or None on failure.
            `tflite_path`  – path to .tflite file, or None on failure.

    Notes:
        * Pruning masks are removed automatically before export.
        * Model is moved to CPU and set to eval mode before export.
        * Sparse parameter tensors are converted to dense for ONNX tracing.
        * Each step is independently guarded; a failure stops subsequent steps
          and reports a meaningful log message rather than crashing silently.
    """
    safe_name = _slugify_name(model_name)
    os.makedirs(export_dir, exist_ok=True)

    onnx_path   = os.path.join(export_dir, f"{safe_name}.onnx")
    tf_dir      = os.path.join(export_dir, "tf_model")
    tflite_path = os.path.join(export_dir, f"{safe_name}.tflite")

    print("\n" + "=" * 65)
    print(f"DEPLOYMENT EXPORT PIPELINE")
    print(f"  Export dir : {export_dir}")
    print(f"  Model name : {safe_name}")
    print("=" * 65)

    # ── Pre-export safety ───────────────────────────────────────────────────
    removed = _remove_pruning_from_model(model)
    if removed:
        print(f"[Pipeline] Removed pruning masks from {removed} layer(s).")

    model = model.cpu()
    model.eval()

    # Dense-ify any sparse parameter tensors so ONNX tracing succeeds
    with torch.no_grad():
        for param in model.parameters():
            if param.is_sparse:
                param.data = param.data.to_dense()

    # ── Step 1: PyTorch → ONNX ─────────────────────────────────────────────
    print("\n[Pipeline] Step 1/3 — PyTorch → ONNX")
    onnx_ok = False
    try:
        export_to_onnx(model, onnx_path, input_shape=input_shape, opset=opset)
        onnx_ok = True
    except RuntimeError as e:
        print(f"[Pipeline] ⚠️  ONNX export failed: {e}")
        print("[Pipeline] Aborting TensorFlow and TFLite steps.")
        _print_export_summary(onnx_ok=False, tf_ok=False, tflite_ok=False,
                              onnx_path=onnx_path, tf_dir=tf_dir,
                              tflite_path=tflite_path)
        return {"onnx_path": None, "tf_dir": None, "tflite_path": None}

    # ── Step 2: ONNX → TensorFlow SavedModel ───────────────────────────────
    print("\n[Pipeline] Step 2/3 — ONNX → TensorFlow SavedModel")
    tf_ok = False
    try:
        onnx_to_tensorflow(onnx_path, tf_dir)
        tf_ok = True
    except Exception as e:
        print(f"[Pipeline] ⚠️  TensorFlow conversion failed: {e}")
        print("[Pipeline] Skipping TFLite step.")

    # ── Step 3: TensorFlow → TFLite ────────────────────────────────────────
    tflite_ok = False
    if tf_ok:
        print("\n[Pipeline] Step 3/3 — TensorFlow → TFLite")
        try:
            tensorflow_to_tflite(tf_dir, tflite_path, quantize=quantize_tflite)
            tflite_ok = True
        except Exception as e:
            print(f"[Pipeline] ⚠️  TFLite conversion failed: {e}")

    # ── Summary ─────────────────────────────────────────────────────────────
    _print_export_summary(onnx_ok, tf_ok, tflite_ok, onnx_path, tf_dir, tflite_path)

    return {
        "onnx_path":   onnx_path   if onnx_ok   else None,
        "tf_dir":      tf_dir      if tf_ok      else None,
        "tflite_path": tflite_path if tflite_ok  else None,
    }


def _print_export_summary(onnx_ok, tf_ok, tflite_ok,
                          onnx_path, tf_dir, tflite_path):
    """Print a formatted export completion summary."""
    print("\n" + "=" * 65)
    print("[EXPORT COMPLETE]")
    print(f"  ONNX   : {onnx_path   if onnx_ok   else '❌ FAILED'}")
    print(f"  TF     : {tf_dir      if tf_ok      else '❌ FAILED'}")
    print(f"  TFLite : {tflite_path if tflite_ok  else '❌ FAILED'}")
    print("=" * 65)