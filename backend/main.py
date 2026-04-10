# main.py
"""
Green AI FYP — Phase 8: FastAPI Backend
========================================
REST API endpoints for the GreenAI compression dashboard:

  GET  /api/health            — Health check
  GET  /api/results           — All compression results summary
  GET  /api/results/{strategy} — Metrics for a specific strategy
  GET  /api/energy            — Energy & emissions report
  GET  /api/evaluation        — Full evaluation report
  GET  /api/models            — List available models with sizes
  POST /api/compress          — Trigger compression on the baseline model
  POST /api/compress/dynamic  — Upload any model + apply dynamic compression
  GET  /api/compare           — Side-by-side comparison of all strategies

Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import base64
import glob
import io
import os
import json
import math
import re
import subprocess
import sys
import time
import shutil
import traceback
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

app = FastAPI(
    title="GreenAI — Model Compression API",
    description="API for compressing deep learning models and tracking energy savings",
    version="1.0.0",
)

# CORS — allow frontend (Next.js dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "uploads")
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "pretrained_baselines")
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure directories
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRETRAINED_DIR, exist_ok=True)

# Track background task status
task_status = {
    "compress": {"running": False, "last_run": None, "error": None},
    "evaluate": {"running": False, "last_run": None, "error": None},
    "energy": {"running": False, "last_run": None, "error": None},
}


# ============================================================
# Helpers
# ============================================================
def load_json(filename):
    """Load a JSON file from the results directory."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _normalize_model_key(value: str) -> str:
    """Normalize model key names to a stable lowercase underscore format."""
    key = (value or "").strip().lower().replace("-", "_")
    key = re.sub(r"\s+", "_", key)
    key = re.sub(r"[^a-z0-9_]", "", key)
    return key


def _format_params_label(total_params: Optional[int]) -> str:
    """Format parameter counts as compact human-readable labels (e.g., 11.2M)."""
    if total_params is None:
        return "N/A"
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.1f}B"
    if total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.1f}M"
    if total_params >= 1_000:
        return f"{total_params / 1_000:.1f}K"
    return str(total_params)


def _to_float(value) -> Optional[float]:
    """Best-effort conversion to a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _to_int(value) -> Optional[int]:
    """Best-effort conversion to an integer via finite float."""
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _first_numeric_value(payload: dict, keys: tuple[str, ...]) -> Optional[float]:
    """Return the first finite numeric value present in payload for given keys."""
    for key in keys:
        if key not in payload:
            continue
        numeric = _to_float(payload.get(key))
        if numeric is not None:
            return numeric
    return None


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_COMPARISON_TEMPERATURE = 1.0
CIFAR10_INPUT_SIZE = 32
CIFAR10_DISPLAY_SIZE = 192

SAMPLE_CLASS_ORDER = [3, 5, 2, 0, 1, 4, 6, 7, 8, 9]

_COMPARISON_SAMPLE_CACHE: Optional[list[dict]] = None


def _strategy_label(strategy: str) -> str:
    labels = {
        "pruning": "Pruned",
        "quantization": "Quantized",
        "hybrid": "Hybrid",
        "kd": "Distilled",
    }
    return labels.get(strategy.lower().strip(), strategy.title())


def _load_compression_history_data() -> dict:
    payload = load_json("compression_history.json")
    if isinstance(payload, dict):
        return payload
    return {}


def _find_compression_history_entry(model_key: str, strategy: str) -> Optional[dict]:
    key = _normalize_model_key(model_key)
    strategy_key = _normalize_model_key(strategy)
    history = _load_compression_history_data()
    for hist_key, entries in history.items():
        if _normalize_model_key(hist_key) != key:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_strategy = _normalize_model_key(
                entry.get("strategy") or entry.get("compression_method") or ""
            )
            if entry_strategy == strategy_key:
                return entry
    return None


def _comparison_data_roots() -> list[str]:
    return [
        os.path.join(BACKEND_DIR, "data"),
        os.path.join(BACKEND_DIR, "..", "data"),
    ]


def _load_cifar10_test_dataset_for_comparison():
    for root in _comparison_data_roots():
        if os.path.exists(os.path.join(root, "cifar-10-batches-py")):
            return torchvision.datasets.CIFAR10(root=root, train=False, download=False)

    # Last-resort fallback if local dataset is missing.
    fallback_root = os.path.join(BACKEND_DIR, "data")
    return torchvision.datasets.CIFAR10(root=fallback_root, train=False, download=True)


def _pil_image_to_data_url(image, display_size: Optional[int] = None) -> str:
    render_image = image
    if display_size is not None:
        target_size = max(int(display_size), CIFAR10_INPUT_SIZE)
        if render_image.size != (target_size, target_size):
            bicubic = getattr(Image, "Resampling", Image).BICUBIC
            render_image = render_image.resize((target_size, target_size), resample=bicubic)

    buffer = io.BytesIO()
    render_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_cifar10_inference_transform(size: int = CIFAR10_INPUT_SIZE):
    target_size = max(int(size), CIFAR10_INPUT_SIZE)
    return transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _build_comparison_sample_images(limit: int = 10) -> list[dict]:
    global _COMPARISON_SAMPLE_CACHE

    if _COMPARISON_SAMPLE_CACHE is not None and len(_COMPARISON_SAMPLE_CACHE) >= min(limit, 10):
        return _COMPARISON_SAMPLE_CACHE[:limit]

    dataset = _load_cifar10_test_dataset_for_comparison()
    targets = list(getattr(dataset, "targets", []))
    if not targets:
        raise RuntimeError("CIFAR-10 targets are unavailable for sample image selection.")

    samples: list[dict] = []
    for class_id in SAMPLE_CLASS_ORDER:
        try:
            idx = next(i for i, label in enumerate(targets) if int(label) == class_id)
        except StopIteration:
            continue

        image, label_idx = dataset[idx]
        label_name = (
            CIFAR10_CLASSES[int(label_idx)]
            if 0 <= int(label_idx) < len(CIFAR10_CLASSES)
            else f"class_{int(label_idx)}"
        )
        samples.append({
            "id": int(idx),
            "label": label_name,
            "class_index": int(label_idx),
            "image_data_url": _pil_image_to_data_url(image, display_size=CIFAR10_DISPLAY_SIZE),
        })
        if len(samples) >= limit:
            break

    _COMPARISON_SAMPLE_CACHE = samples
    return samples


def _get_sample_image_by_id(sample_id: int):
    dataset = _load_cifar10_test_dataset_for_comparison()
    if sample_id < 0 or sample_id >= len(dataset):
        raise ValueError(f"Sample image id {sample_id} is out of range.")
    image, label_idx = dataset[sample_id]
    return image, int(label_idx)


def _build_preloaded_model_architecture(model_key: str, num_classes: int = 10):
    from compress import PRELOADED_MODELS

    key = _normalize_model_key(model_key)
    if key not in PRELOADED_MODELS:
        raise ValueError(
            f"Unknown baseline model '{model_key}'. Available: {list(PRELOADED_MODELS.keys())}"
        )

    if key == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key == "resnet34":
        model = torchvision.models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif key == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif key == "efficientnet_b1":
        model = torchvision.models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif key == "densenet121":
        model = torchvision.models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif key == "densenet169":
        model = torchvision.models.densenet169(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif key == "squeezenet":
        model = torchvision.models.squeezenet1_1(weights=None)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
    elif key == "shufflenet_v2":
        model = torchvision.models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key == "inception_v3":
        model = torchvision.models.inception_v3(weights=None, aux_logits=True)
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key == "googlenet":
        model = torchvision.models.googlenet(weights=None, aux_logits=True)
        model.aux_logits = False
        model.aux1 = None
        model.aux2 = None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported baseline model architecture '{model_key}'.")

    model._input_size = PRELOADED_MODELS[key]["input_size"]
    return model


def _find_compressed_artifact(model_key: str, strategy: str) -> Optional[str]:
    normalized_model = _normalize_model_key(model_key)
    normalized_strategy = _normalize_model_key(strategy)

    strategy_prefixes = {
        "pruning": ["pruned"],
        "quantization": ["quantized"],
        "hybrid": ["hybrid"],
        "kd": ["kd"],
    }
    prefixes = strategy_prefixes.get(normalized_strategy, [normalized_strategy])

    compressed_dir = os.path.join(MODELS_DIR, "compressed")
    candidates: list[str] = []
    for prefix in prefixes:
        candidates.extend(glob.glob(os.path.join(compressed_dir, f"{normalized_model}_{prefix}*.pth")))
        candidates.extend(glob.glob(os.path.join(compressed_dir, f"{normalized_model}_{prefix}*.pth.gz")))

    if not candidates:
        return None

    candidates = [path for path in candidates if os.path.isfile(path)]
    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def _normalize_loaded_state_dict(state: dict) -> dict:
    normalized = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            tensor = value.to_dense() if value.is_sparse else value
            if tensor.is_quantized:
                tensor = tensor.dequantize()
            normalized[key] = tensor
        else:
            normalized[key] = value
    return normalized


def _load_state_dict_best_effort(model: nn.Module, state: dict):
    try:
        model.load_state_dict(state, strict=True)
        return
    except Exception:
        pass

    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    matched = len(model_keys & state_keys)
    if matched == 0:
        raise ValueError("No matching keys found between model architecture and saved artifact.")

    model.load_state_dict(state, strict=False)


def _load_baseline_model_for_comparison(model_key: str, device):
    key = _normalize_model_key(model_key)
    model = _build_preloaded_model_architecture(key, num_classes=10)

    baseline_weights_path = os.path.join(PRETRAINED_DIR, f"{key}_baseline.pth")
    if os.path.exists(baseline_weights_path):
        state = torch.load(baseline_weights_path, map_location=device)
        if isinstance(state, dict):
            _load_state_dict_best_effort(model, _normalize_loaded_state_dict(state))

    model = model.to(device)
    model.eval()
    return model


def _load_compressed_model_for_comparison(model_key: str, strategy: str, device):
    from compress import CompactStudent, load_compressed

    key = _normalize_model_key(model_key)
    strategy_key = _normalize_model_key(strategy)

    artifact_path = _find_compressed_artifact(key, strategy_key)
    if artifact_path is None:
        raise FileNotFoundError(
            f"No compressed artifact found for model='{key}' and strategy='{strategy_key}'."
        )

    if strategy_key == "kd":
        model = CompactStudent(num_classes=10)
        model._input_size = 32
    else:
        model = _build_preloaded_model_architecture(key, num_classes=10)

    loaded = load_compressed(artifact_path, device=device)
    if isinstance(loaded, nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        normalized_state = _normalize_loaded_state_dict(loaded)
        _load_state_dict_best_effort(model, normalized_state)
    else:
        raise ValueError("Compressed artifact is not a state dict or nn.Module.")

    model = model.to(device)
    model.eval()
    return model, artifact_path


def _predict_model_on_image(model: nn.Module, image, device) -> dict:
    from compress import detect_input_shape, _extract_logits

    model.eval()

    input_shape = detect_input_shape(model)
    model_input_size = int(input_shape[-1]) if len(input_shape) >= 4 else CIFAR10_INPUT_SIZE
    inference_input_size = CIFAR10_INPUT_SIZE

    # Temperature scaling improves confidence reliability without retraining.
    try:
        temperature = float(os.getenv("MODEL_COMPARISON_TEMPERATURE", DEFAULT_COMPARISON_TEMPERATURE))
    except (TypeError, ValueError):
        temperature = DEFAULT_COMPARISON_TEMPERATURE
    if not math.isfinite(temperature) or temperature <= 0:
        temperature = DEFAULT_COMPARISON_TEMPERATURE

    print(f"[ModelCompare] Original image size: {image.size}")
    transform = _build_cifar10_inference_transform(inference_input_size)
    tensor_chw = transform(image)
    print(f"[ModelCompare] Tensor shape after transform (C,H,W): {tuple(tensor_chw.shape)}")
    if tuple(tensor_chw.shape) != (3, inference_input_size, inference_input_size):
        print(
            "[ModelCompare] WARNING: unexpected transformed shape "
            f"{tuple(tensor_chw.shape)} (expected (3, {inference_input_size}, {inference_input_size}))"
        )

    tensor = tensor_chw.unsqueeze(0).to(device)
    used_input_size = inference_input_size
    with torch.no_grad():
        try:
            logits = _extract_logits(model(tensor))
        except Exception as inference_error:
            if model_input_size != inference_input_size:
                print(
                    "[ModelCompare] 32x32 inference failed; retrying with "
                    f"{model_input_size}x{model_input_size}: {inference_error}"
                )
                fallback_transform = _build_cifar10_inference_transform(model_input_size)
                fallback_tensor_chw = fallback_transform(image)
                print(
                    "[ModelCompare] Fallback tensor shape (C,H,W): "
                    f"{tuple(fallback_tensor_chw.shape)}"
                )
                tensor = fallback_tensor_chw.unsqueeze(0).to(device)
                logits = _extract_logits(model(tensor))
                used_input_size = model_input_size
            else:
                raise

        logits = logits / temperature
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs[0], dim=0)

    top_k = min(3, int(probs.shape[1]))
    top_values, top_indices = torch.topk(probs[0], k=top_k)
    top_predictions = []
    for prob, cls_idx in zip(top_values.tolist(), top_indices.tolist()):
        cls_idx = int(cls_idx)
        cls_name = (
            CIFAR10_CLASSES[cls_idx]
            if 0 <= cls_idx < len(CIFAR10_CLASSES)
            else f"class_{cls_idx}"
        )
        top_predictions.append({
            "class_name": cls_name,
            "class_index": cls_idx,
            "probability": round(float(prob * 100), 2),
        })

    class_index = int(pred_idx.item())
    class_name = (
        CIFAR10_CLASSES[class_index]
        if 0 <= class_index < len(CIFAR10_CLASSES)
        else f"class_{class_index}"
    )

    return {
        "predicted_class": class_name,
        "predicted_index": class_index,
        "confidence": round(float(confidence.item() * 100), 2),
        "input_size": used_input_size,
        "temperature": round(float(temperature), 3),
        "top_k": top_predictions,
    }


def _build_model_comparison_options() -> dict:
    from compress import PRELOADED_MODELS

    baseline_metrics = load_baseline_results()
    baseline_models = []
    for key, cfg in PRELOADED_MODELS.items():
        metrics = baseline_metrics.get(key, {})
        baseline_models.append({
            "key": key,
            "name": cfg["name"],
            "input_size": cfg["input_size"],
            "dataset": cfg["dataset"],
            "size_MB": metrics.get("size_MB"),
            "co2_kg": metrics.get("training_co2_kg"),
            "status": metrics.get("status", "ready" if key in baseline_metrics else "not_ready"),
        })

    compressed_map = {}
    history = _load_compression_history_data()
    for hist_key, entries in history.items():
        model_key = _normalize_model_key(hist_key)
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            strategy = _normalize_model_key(entry.get("strategy") or entry.get("compression_method") or "")
            if not strategy or strategy == "baseline":
                continue

            artifact = _find_compressed_artifact(model_key, strategy)
            if artifact is None:
                continue

            option_key = f"{model_key}::{strategy}"
            model_name = (
                entry.get("model_name")
                or PRELOADED_MODELS.get(model_key, {}).get("name")
                or model_key
            )
            size_mb = _first_numeric_value(entry, ("size_MB", "compressed_size_MB"))
            co2_kg = _first_numeric_value(
                entry,
                (
                    "training_co2_kg",
                    "training_emissions_kg",
                    "inference_co2_kg",
                    "inference_emissions_kg",
                    "co2_kg",
                    "emissions_kg",
                ),
            )

            compressed_map[option_key] = {
                "key": option_key,
                "model_key": model_key,
                "model_name": model_name,
                "strategy": strategy,
                "strategy_label": _strategy_label(strategy),
                "label": f"{model_name} - {_strategy_label(strategy)}",
                "size_MB": round(size_mb, 2) if size_mb is not None else None,
                "co2_kg": co2_kg,
                "artifact_name": os.path.basename(artifact),
            }

    compressed_models = sorted(
        compressed_map.values(),
        key=lambda item: (str(item.get("model_name", "")).lower(), str(item.get("strategy", "")).lower()),
    )

    return {
        "baseline_models": baseline_models,
        "compressed_models": compressed_models,
    }


def list_models():
    """List all model files with sizes."""
    models = []
    if os.path.exists(MODELS_DIR):
        for f in sorted(os.listdir(MODELS_DIR)):
            fpath = os.path.join(MODELS_DIR, f)
            if os.path.isfile(fpath):
                models.append({
                    "filename": f,
                    "size_MB": round(os.path.getsize(fpath) / 1e6, 2),
                })
    return models


def _save_to_compression_history(model_key: str, result: dict):
    """Append a compression result to the history file, keyed by model."""
    history_path = os.path.join(RESULTS_DIR, "compression_history.json")
    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = {}

    if model_key not in history:
        history[model_key] = []

    # Deduplicate by strategy (keep latest)
    strategy = result.get("strategy", result.get("compression_method", "unknown"))
    history[model_key] = [
        r for r in history[model_key]
        if (r.get("strategy") or r.get("compression_method")) != strategy
    ]
    history[model_key].append(result)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# ============================================================
# Baseline Models — pre-trained baselines for all 15 models
# ============================================================

def load_baseline_results():
    """Load latest valid baseline metrics by aggregating per-model training result files."""
    suffixes = ("_training_result.json", "_training_results.json")
    baselines = {}

    if not os.path.exists(RESULTS_DIR):
        return baselines

    candidates = []
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(suffixes):
            continue

        path = os.path.join(RESULTS_DIR, filename)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue

        candidates.append((mtime, filename, path))

    # Newest files first so we keep only the latest valid entry per model key.
    for mtime, filename, path in sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True):
        if not os.path.isfile(path):
            continue

        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        filename_model_key = _normalize_model_key(
            filename.replace("_training_result.json", "").replace("_training_results.json", "")
        )
        model_name_raw = payload.get("model_name") or filename_model_key or filename
        model_key = filename_model_key or _normalize_model_key(model_name_raw)
        if not model_key:
            model_key = _normalize_model_key(
                filename.replace("_training_result.json", "").replace("_training_results.json", "")
            )
        if not model_key or model_key in baselines:
            continue

        total_params = _to_int(payload.get("total_params", payload.get("parameters")))
        size_mb = _first_numeric_value(payload, ("size_MB", "original_size_MB", "baseline_size_MB"))
        accuracy = _first_numeric_value(payload, ("accuracy", "baseline_accuracy", "accuracy_top1"))
        training_co2_kg = _first_numeric_value(
            payload,
            ("train_co2_kg", "training_co2_kg", "training_emissions_kg", "co2_kg", "emissions_kg"),
        )
        training_energy_kwh = _first_numeric_value(
            payload,
            ("train_energy_kwh", "training_energy_kwh", "energy_kwh"),
        )

        # Skip malformed payloads and continue searching older files for this model.
        if accuracy is None and size_mb is None and training_co2_kg is None:
            continue

        updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

        baselines[model_key] = {
            "model_key": model_key,
            "model_name": model_name_raw,
            "params_label": _format_params_label(total_params),
            "total_params": total_params,
            "input_size": payload.get("input_size"),
            "dataset": payload.get("dataset", "CIFAR10"),
            "accuracy": round(accuracy, 2) if accuracy is not None else None,
            "size_MB": round(size_mb, 2) if size_mb is not None else None,
            "latency_ms": payload.get("latency_ms"),
            "training_co2_kg": training_co2_kg,
            "training_energy_kwh": training_energy_kwh,
            "result_updated_at": updated_at,
            "result_file": filename,
            "status": "ready",
        }

    return baselines


@app.get("/api/baselines")
async def get_baselines():
    """
    Get baseline metrics for all 15 pretrained models.
    Returns pre-computed accuracy, size, latency, params for each model.
    If not yet prepared, returns available model list with 'not_ready' status.
    """
    sys.path.insert(0, BACKEND_DIR)
    from compress import PRELOADED_MODELS

    baselines = load_baseline_results()

    # Merge with PRELOADED_MODELS metadata and mark missing entries as not_ready
    for key, cfg in PRELOADED_MODELS.items():
        if key not in baselines:
            baselines[key] = {
                "model_key": key,
                "model_name": cfg["name"],
                "params_label": cfg["params"],
                "input_size": cfg["input_size"],
                "dataset": cfg["dataset"],
                "training_co2_kg": None,
                "training_energy_kwh": None,
                "status": "not_ready",
            }
            continue

        # Keep canonical display metadata from configured preloaded model catalog.
        baselines[key]["model_name"] = cfg["name"]
        baselines[key]["params_label"] = baselines[key].get("params_label") or cfg["params"]
        baselines[key]["input_size"] = baselines[key].get("input_size") or cfg["input_size"]
        baselines[key]["dataset"] = baselines[key].get("dataset") or cfg["dataset"]

    canonical_baselines = {key: baselines[key] for key in PRELOADED_MODELS.keys()}
    ready = sum(1 for v in canonical_baselines.values() if v.get("status") == "ready")

    return {
        "models": canonical_baselines,
        "ready_count": ready,
        "total_count": len(canonical_baselines),
    }


@app.get("/api/baselines/{model_key}")
async def get_baseline_detail(model_key: str):
    """Get baseline metrics for a specific model."""
    baselines = load_baseline_results()
    key = _normalize_model_key(model_key)
    if key not in baselines:
        raise HTTPException(status_code=404, detail=f"Baseline not found for '{model_key}'")
    return baselines[key]


class ModelComparisonRequest(BaseModel):
    sample_id: int
    baseline_model_key: str
    compressed_model_key: str


@app.get("/api/model-comparison/options")
async def get_model_comparison_options():
    """Return baseline and compressed model options for the image-comparison UI."""
    return _build_model_comparison_options()


@app.get("/api/model-comparison/sample-images")
async def get_model_comparison_sample_images(
    limit: int = Query(default=10, ge=4, le=20),
):
    """Return a curated grid of CIFAR-10 sample images with labels."""
    try:
        samples = _build_comparison_sample_images(limit=limit)
        return {"samples": samples}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load sample images: {exc}")


@app.post("/api/model-comparison/compare")
async def compare_models_on_image(req: ModelComparisonRequest):
    """Run baseline and compressed model inference on the exact same selected sample image."""
    baseline_key = _normalize_model_key(req.baseline_model_key)
    compressed_key_raw = (req.compressed_model_key or "").strip()

    if "::" not in compressed_key_raw:
        raise HTTPException(
            status_code=400,
            detail="Invalid compressed model selection format. Expected '<model_key>::<strategy>'.",
        )

    compressed_model_key_raw, compressed_strategy_raw = compressed_key_raw.split("::", 1)
    compressed_model_key = _normalize_model_key(compressed_model_key_raw)
    compressed_strategy = _normalize_model_key(compressed_strategy_raw)

    if not baseline_key:
        raise HTTPException(status_code=400, detail="Baseline model key is required.")
    if not compressed_model_key or not compressed_strategy:
        raise HTTPException(status_code=400, detail="Compressed model selection is incomplete.")

    history_entry = _find_compression_history_entry(compressed_model_key, compressed_strategy)
    if history_entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No compression history found for model '{compressed_model_key}' "
                f"with strategy '{compressed_strategy}'."
            ),
        )

    try:
        image, label_idx = _get_sample_image_by_id(int(req.sample_id))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sample image selection: {exc}")

    device = torch.device("cpu")

    try:
        baseline_model = _load_baseline_model_for_comparison(baseline_key, device=device)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load baseline model: {exc}")

    try:
        compressed_model, compressed_artifact = _load_compressed_model_for_comparison(
            compressed_model_key,
            compressed_strategy,
            device=device,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load compressed model: {exc}")

    baseline_pred = _predict_model_on_image(baseline_model, image, device=device)
    compressed_pred = _predict_model_on_image(compressed_model, image, device=device)

    baseline_metrics = load_baseline_results().get(baseline_key, {})
    baseline_size = _to_float(baseline_metrics.get("size_MB"))
    baseline_co2 = _to_float(baseline_metrics.get("training_co2_kg"))

    if baseline_size is None:
        baseline_size = _first_numeric_value(history_entry, ("baseline_size_MB", "original_size_MB"))

    compressed_size = _first_numeric_value(history_entry, ("size_MB", "compressed_size_MB"))
    compressed_co2 = _first_numeric_value(
        history_entry,
        (
            "training_co2_kg",
            "training_emissions_kg",
            "inference_co2_kg",
            "inference_emissions_kg",
            "co2_kg",
            "emissions_kg",
        ),
    )

    confidence_delta = round(compressed_pred["confidence"] - baseline_pred["confidence"], 2)

    size_reduction = None
    if baseline_size is not None and baseline_size > 0 and compressed_size is not None:
        size_reduction = round(((baseline_size - compressed_size) / baseline_size) * 100, 2)

    co2_reduction = None
    if baseline_co2 is not None and baseline_co2 > 0 and compressed_co2 is not None:
        co2_reduction = round(((baseline_co2 - compressed_co2) / baseline_co2) * 100, 2)

    size_reduction_text = f"{size_reduction:.2f}%" if size_reduction is not None else "N/A"
    summary = (
        f"Compressed model reduced size by {size_reduction_text} "
        f"with {confidence_delta:+.2f}% change in confidence"
    )

    if co2_reduction is not None:
        summary = f"{summary} and {co2_reduction:.2f}% CO2 reduction"

    true_label = (
        CIFAR10_CLASSES[label_idx]
        if 0 <= int(label_idx) < len(CIFAR10_CLASSES)
        else f"class_{int(label_idx)}"
    )

    return {
        "sample": {
            "id": int(req.sample_id),
            "true_label": true_label,
            "image_data_url": _pil_image_to_data_url(image, display_size=CIFAR10_DISPLAY_SIZE),
        },
        "baseline": {
            "model_key": baseline_key,
            "model_name": baseline_metrics.get("model_name") or baseline_key,
            "prediction": baseline_pred,
            "size_MB": baseline_size,
            "co2_kg": baseline_co2,
        },
        "compressed": {
            "model_key": compressed_model_key,
            "strategy": compressed_strategy,
            "model_name": history_entry.get("model_name") or compressed_model_key,
            "strategy_label": _strategy_label(compressed_strategy),
            "prediction": compressed_pred,
            "size_MB": compressed_size,
            "co2_kg": compressed_co2,
            "artifact": os.path.basename(compressed_artifact),
        },
        "comparison": {
            "confidence_delta_percent": confidence_delta,
            "size_reduction_percent": size_reduction,
            "co2_reduction_percent": co2_reduction,
            "summary": summary,
        },
    }


# Track prepare task
prepare_task = {
    "running": False,
    "progress": "",
    "completed": 0,
    "total": 0,
    "current_model": "",
    "error": None,
}


@app.post("/api/prepare")
async def trigger_prepare(models: list[str] = None):
    """
    Start pre-downloading and evaluating one or more models.
    If models is None, prepares all 15 models.
    Runs in background — poll /api/prepare/status.
    """
    if prepare_task["running"]:
        raise HTTPException(status_code=409, detail="Preparation already running")

    sys.path.insert(0, BACKEND_DIR)
    from compress import PRELOADED_MODELS

    if models is None:
        model_keys = list(PRELOADED_MODELS.keys())
    else:
        model_keys = [m.lower().strip() for m in models]
        invalid = [m for m in model_keys if m not in PRELOADED_MODELS]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model(s): {invalid}. Available: {list(PRELOADED_MODELS.keys())}"
            )

    prepare_task["running"] = True
    prepare_task["progress"] = "Starting..."
    prepare_task["completed"] = 0
    prepare_task["total"] = len(model_keys)
    prepare_task["current_model"] = ""
    prepare_task["error"] = None

    def _run():
        try:
            from prepare_models import prepare_single_model
            import torch

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load existing
            results = {}
            results_path = os.path.join(RESULTS_DIR, "baseline_all_models.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    results = json.load(f)

            for i, key in enumerate(model_keys):
                if key in results and results[key].get("status") == "ready":
                    prepare_task["completed"] = i + 1
                    prepare_task["progress"] = f"{key} already ready, skipping"
                    prepare_task["current_model"] = key
                    continue

                prepare_task["current_model"] = key
                prepare_task["progress"] = f"Preparing {key} ({i+1}/{len(model_keys)})"

                try:
                    result = prepare_single_model(key, device=device)
                    results[key] = result
                except Exception as e:
                    results[key] = {
                        "model_key": key,
                        "model_name": PRELOADED_MODELS[key]["name"],
                        "status": "error",
                        "error": str(e),
                    }

                prepare_task["completed"] = i + 1

                # Save incrementally
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

            prepare_task["progress"] = "Complete"
        except Exception as e:
            prepare_task["error"] = str(e)
            traceback.print_exc()
        finally:
            prepare_task["running"] = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return JSONResponse(
        status_code=202,
        content={
            "message": f"Preparing {len(model_keys)} model(s)",
            "models": model_keys,
            "status_endpoint": "/api/prepare/status",
        }
    )


@app.get("/api/prepare/status")
async def prepare_status():
    """Check preparation progress."""
    return prepare_task


# ============================================================
# Compression History — saved results per model
# ============================================================

@app.get("/api/compression-history")
async def get_compression_history():
    """Get all saved compression results, organized by model."""
    path = os.path.join(RESULTS_DIR, "compression_history.json")
    if not os.path.exists(path):
        return {"history": {}}
    with open(path, "r") as f:
        return {"history": json.load(f)}


@app.get("/api/results")
async def get_results():
    """Get the full compression results summary."""
    data = load_json("compression_summary.json")
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No compression results found. Run compress.py first."
        )
    return data


@app.get("/api/results/{strategy}")
async def get_strategy_results(strategy: str):
    """
    Get results for a specific compression strategy.
    Valid keys: baseline, pruning_compressed, quantization_static,
    kd_compact_student, hybrid_student_quant, ultra_compact
    """
    data = load_json("compression_summary.json")
    if data is None:
        raise HTTPException(status_code=404, detail="No results found")

    if strategy not in data:
        available = list(data.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy}' not found. Available: {available}"
        )
    return {strategy: data[strategy]}


@app.get("/api/energy")
async def get_energy():
    """Get energy and emissions tracking report."""
    data = load_json("energy_report.json")
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No energy report found. Run energy.py first."
        )
    return data


@app.get("/api/evaluation")
async def get_evaluation():
    """Get full evaluation report (accuracy, FLOPs, latency, per-class)."""
    data = load_json("evaluation_report.json")
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation report found. Run evaluate.py first."
        )
    return data


@app.get("/api/models")
async def get_models():
    """List all saved model files and their sizes."""
    return {"models": list_models()}


@app.get("/api/compare")
async def compare_strategies():
    """
    Side-by-side comparison table of all compression strategies.
    Returns a normalized format for easy frontend rendering.
    """
    summary = load_json("compression_summary.json")
    energy = load_json("energy_report.json")
    evaluation = load_json("evaluation_report.json")

    if summary is None:
        raise HTTPException(status_code=404, detail="No results found")

    comparison = []
    for key, metrics in summary.items():
        entry = {
            "strategy": key,
            "accuracy": metrics.get("accuracy",
                        metrics.get("accuracy_top1", None)),
            "size_MB": metrics.get("size_MB",
                       metrics.get("size_MB_compressed",
                       metrics.get("size_MB_sparse", None))),
            "size_reduction_percent": metrics.get("size_reduction_percent",
                                     metrics.get("size_reduction_compressed_percent", 0)),
            "latency_ms": metrics.get("latency_ms", None),
            "params": metrics.get("params",
                      metrics.get("student_params",
                      metrics.get("total_params", None))),
        }

        # Add energy data if available
        if energy and "inference" in energy:
            inf_data = energy["inference"].get(key, {})
            entry["inference_energy_kWh"] = inf_data.get("energy_kWh", None)
            entry["co2_kg"] = inf_data.get("co2_kg", None)

        # Add evaluation data if available
        if evaluation and key in evaluation:
            eval_data = evaluation[key]
            entry["accuracy_top5"] = eval_data.get("accuracy_top5", None)
            entry["flops_M"] = eval_data.get("flops_M", None)
            entry["sparsity_percent"] = eval_data.get("sparsity_percent", None)

        comparison.append(entry)

    return {"comparison": comparison}


# ============================================================
# Background task endpoints
# ============================================================

class CompressRequest(BaseModel):
    """Request body for compression endpoint."""
    strategies: Optional[list] = None  # If None, run all
    description: Optional[str] = "Run compression pipeline"


def run_script(script_name, task_key):
    """Run a Python script as subprocess and track status."""
    task_status[task_key]["running"] = True
    task_status[task_key]["error"] = None
    try:
        python_exe = sys.executable
        result = subprocess.run(
            [python_exe, script_name],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        if result.returncode != 0:
            task_status[task_key]["error"] = result.stderr[-500:]
        task_status[task_key]["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except subprocess.TimeoutExpired:
        task_status[task_key]["error"] = "Script timed out (1 hour limit)"
    except Exception as e:
        task_status[task_key]["error"] = str(e)
    finally:
        task_status[task_key]["running"] = False


@app.post("/api/compress")
async def trigger_compression(background_tasks: BackgroundTasks):
    """
    Trigger the full compression pipeline in the background.
    Returns immediately; check /api/task-status for progress.
    """
    if task_status["compress"]["running"]:
        raise HTTPException(
            status_code=409,
            detail="Compression is already running"
        )
    background_tasks.add_task(run_script, "compress.py", "compress")
    return {
        "message": "Compression pipeline started in background",
        "status_endpoint": "/api/task-status"
    }


@app.post("/api/evaluate")
async def trigger_evaluation(background_tasks: BackgroundTasks):
    """Trigger model evaluation in the background."""
    if task_status["evaluate"]["running"]:
        raise HTTPException(status_code=409, detail="Evaluation already running")
    background_tasks.add_task(run_script, "evaluate.py", "evaluate")
    return {"message": "Evaluation started", "status_endpoint": "/api/task-status"}


@app.post("/api/energy/track")
async def trigger_energy_tracking(background_tasks: BackgroundTasks):
    """Trigger energy tracking in the background."""
    if task_status["energy"]["running"]:
        raise HTTPException(status_code=409, detail="Energy tracking already running")
    background_tasks.add_task(run_script, "energy.py", "energy")
    return {"message": "Energy tracking started", "status_endpoint": "/api/task-status"}


@app.get("/api/task-status")
async def get_task_status():
    """Check the status of background tasks."""
    return task_status


# ============================================================
# Preloaded Model Compression — 15 curated models
# ============================================================

class PreloadedCompressRequest(BaseModel):
    """Request body for preloaded model compression."""
    model_name: str
    method: str = "pruning"
    dataset: str = "CIFAR10"
    fine_tune_epochs: int = 5


# Track preloaded compression status
preloaded_task = {
    "running": False,
    "step": "",          # current step key: loading_model, loading_data, compressing, energy_tracking, evaluating, complete
    "detail": "",        # human-readable detail text
    "progress": "",      # same as detail (kept for backwards compatibility)
    "result": None,
    "error": None,
}

# Steps in order — frontend uses this list to render the progress bar
COMPRESSION_STEPS = [
    {"key": "loading_model", "label": "Loading Model"},
    {"key": "loading_data", "label": "Preparing Dataset"},
    {"key": "compressing", "label": "Compressing"},
    {"key": "energy_tracking", "label": "Energy Tracking"},
    {"key": "evaluating", "label": "Final Evaluation"},
    {"key": "complete", "label": "Complete"},
]


@app.get("/api/preloaded-models")
async def get_preloaded_models():
    """List all 15 available pretrained models."""
    sys.path.insert(0, BACKEND_DIR)
    from compress import PRELOADED_MODELS
    models = []
    for key, cfg in PRELOADED_MODELS.items():
        models.append({
            "key": key,
            "name": cfg["name"],
            "params": cfg["params"],
            "input_size": cfg["input_size"],
            "dataset": cfg["dataset"],
        })
    return {"models": models}


@app.post("/api/compress/preloaded")
async def compress_preloaded(req: PreloadedCompressRequest):
    """
    Start compressing a preloaded pretrained model in the background.
    Returns 202 immediately. Poll /api/compress/preloaded/status for progress.

    Body JSON:
        {
            "model_name": "resnet18",
            "method": "pruning",
            "dataset": "CIFAR10",
            "fine_tune_epochs": 5
        }
    """
    if preloaded_task["running"]:
        raise HTTPException(
            status_code=409,
            detail="A compression task is already running."
        )

    # Validate model name
    sys.path.insert(0, BACKEND_DIR)
    from compress import PRELOADED_MODELS
    model_key = req.model_name.lower().strip()
    if model_key not in PRELOADED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model_name}'. "
                   f"Available: {list(PRELOADED_MODELS.keys())}"
        )

    # Validate method
    valid_methods = ['pruning', 'quantization', 'hybrid', 'kd']
    method = req.method.lower().strip()
    if method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method '{req.method}'. Choose from: {valid_methods}"
        )

    # Validate dataset
    valid_datasets = ['CIFAR10', 'CIFAR100']
    dataset = req.dataset.upper()
    if dataset not in valid_datasets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset '{req.dataset}'. Choose from: {valid_datasets}"
        )

    # Reset task state
    preloaded_task["running"] = True
    preloaded_task["step"] = "loading_model"
    preloaded_task["detail"] = f"Loading {req.model_name}..."
    preloaded_task["progress"] = f"Loading {req.model_name}..."
    preloaded_task["error"] = None
    preloaded_task["result"] = None

    def _run_in_thread():
        try:
            from compress import run_compression

            def progress_cb(step, detail=''):
                preloaded_task["step"] = step
                preloaded_task["detail"] = detail
                preloaded_task["progress"] = detail

            result = run_compression(
                model_name=model_key,
                method=method,
                dataset=dataset,
                fine_tune_epochs=req.fine_tune_epochs,
                progress_cb=progress_cb,
            )

            # Remove internal path info
            result.pop("saved_path", None)

            preloaded_task["result"] = result
            preloaded_task["step"] = "complete"
            preloaded_task["detail"] = "Done!"
            preloaded_task["progress"] = "Complete"

            # Save result to last-result file
            result_path = os.path.join(RESULTS_DIR, "preloaded_compression_result.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            # Append to compression history (keyed by model)
            _save_to_compression_history(model_key, result)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            preloaded_task["error"] = error_msg
            preloaded_task["step"] = "error"
            preloaded_task["detail"] = error_msg
            traceback.print_exc()
        finally:
            preloaded_task["running"] = False

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()

    return JSONResponse(
        status_code=202,
        content={
            "message": "Compression started",
            "status_endpoint": "/api/compress/preloaded/status",
        }
    )


@app.get("/api/compress/preloaded/status")
async def preloaded_compress_status():
    """Check the status of the preloaded compression task, including step-by-step progress."""
    return {
        **preloaded_task,
        "steps": COMPRESSION_STEPS,
    }


@app.get("/api/compress/preloaded/result")
async def preloaded_compress_result():
    """Get the last preloaded compression result."""
    result = load_json("preloaded_compression_result.json")
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No preloaded compression result found yet."
        )
    return result


# ============================================================
# Dynamic Model Compression — upload any model
# ============================================================

# Track dynamic compression status
dynamic_task = {
    "running": False,
    "progress": "",
    "result": None,
    "error": None,
}


@app.post("/api/compress/dynamic")
async def dynamic_compress(
    model_file: UploadFile = File(...),
    strategy: str = Form("pruning"),
    dataset: str = Form("CIFAR10"),
    fine_tune_epochs: int = Form(5),
    architecture: str = Form("auto"),
    num_classes: int = Form(10),
):
    """
    Upload any PyTorch model (.pt/.pth) and apply dynamic compression.

    The architecture is auto-detected from the uploaded file.
    You can optionally specify it if auto-detection fails.

    - model_file: The PyTorch model file (state_dict or full model)
    - strategy: pruning | quantization | hybrid | kd
    - dataset: CIFAR10 | CIFAR100
    - fine_tune_epochs: Number of fine-tuning epochs (default: 5)
    - architecture: auto (default) | resnet18 | resnet34 | resnet50 | mobilenet_v2 | vgg16 | compact_student
    - num_classes: Number of output classes (default: 10)

    Returns JSON with compression metrics.
    """
    if dynamic_task["running"]:
        raise HTTPException(
            status_code=409,
            detail="A dynamic compression task is already running."
        )

    # Validate file extension
    filename = model_file.filename or "uploaded_model.pth"
    if not filename.endswith(('.pt', '.pth')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a .pt or .pth file."
        )

    # Validate strategy
    valid_strategies = ['pruning', 'quantization', 'hybrid', 'kd']
    if strategy.lower() not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy '{strategy}'. Choose from: {valid_strategies}"
        )

    # Validate dataset
    valid_datasets = ['CIFAR10', 'CIFAR100']
    if dataset.upper() not in valid_datasets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset '{dataset}'. Choose from: {valid_datasets}"
        )

    # Save uploaded file
    upload_path = os.path.join(UPLOADS_DIR, filename)
    try:
        with open(upload_path, "wb") as f:
            content = await model_file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Run compression synchronously (can take minutes for KD)
    dynamic_task["running"] = True
    dynamic_task["progress"] = f"Compressing with {strategy}..."
    dynamic_task["error"] = None
    dynamic_task["result"] = None

    try:
        # Import the dynamic compress function
        sys.path.insert(0, BACKEND_DIR)
        from compress import compress_dynamic

        result = compress_dynamic(
            model_path=upload_path,
            strategy=strategy.lower(),
            dataset=dataset.upper(),
            fine_tune_epochs=fine_tune_epochs,
            architecture=architecture.lower().strip(),
            num_classes=num_classes,
        )

        # Remove internal path info from result before returning
        result.pop("saved_path", None)

        dynamic_task["result"] = result
        dynamic_task["progress"] = "Complete"

        # Save result to results dir
        result_path = os.path.join(RESULTS_DIR, "dynamic_compression_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        dynamic_task["error"] = error_msg
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        dynamic_task["running"] = False


@app.get("/api/compress/dynamic/status")
async def dynamic_compress_status():
    """Check the status of the dynamic compression task."""
    return dynamic_task


@app.get("/api/compress/dynamic/result")
async def dynamic_compress_result():
    """Get the last dynamic compression result."""
    result = load_json("dynamic_compression_result.json")
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No dynamic compression result found. Upload a model first."
        )
    return result


@app.get("/api/uploads")
async def list_uploads():
    """List all uploaded model files."""
    uploads = []
    if os.path.exists(UPLOADS_DIR):
        for f in sorted(os.listdir(UPLOADS_DIR)):
            fpath = os.path.join(UPLOADS_DIR, f)
            if os.path.isfile(fpath) and f.endswith(('.pt', '.pth', '.pth.gz')):
                uploads.append({
                    "filename": f,
                    "size_MB": round(os.path.getsize(fpath) / 1e6, 2),
                })
    return {"uploads": uploads}


# ============================================================
# Metrics summary endpoint (for frontend dashboard)
# ============================================================

@app.get("/api/dashboard")
async def dashboard():
    """
    Aggregated dashboard data — combines results, energy, evaluation
    into one response for the frontend to render.
    """
    summary = load_json("compression_summary.json") or {}
    energy = load_json("energy_report.json") or {}
    evaluation = load_json("evaluation_report.json") or {}
    models = list_models()

    # Build strategy cards
    strategies = []
    baseline_data = summary.get("baseline", {})
    baseline_size = baseline_data.get("size_MB", 44.81)
    baseline_acc = baseline_data.get("accuracy", 0)

    strategy_names = {
        "baseline": "Baseline (ResNet18)",
        "pruning_compressed": "Pruning 70% + Gzip",
        "quantization_static": "Static Quantization INT8",
        "quantization_dynamic": "Dynamic Quantization",
        "kd_compact_student": "KD → Compact Student",
        "hybrid_student_quant": "Student + Quantization",
        "ultra_compact": "Ultra-Compact",
    }

    for key, display_name in strategy_names.items():
        if key not in summary:
            continue
        m = summary[key]
        size = m.get("size_MB", m.get("size_MB_compressed",
               m.get("size_MB_sparse", baseline_size)))
        strategies.append({
            "key": key,
            "name": display_name,
            "accuracy": m.get("accuracy", 0),
            "size_MB": size,
            "size_reduction": round(100 * (baseline_size - size) / baseline_size, 2)
                             if baseline_size > 0 else 0,
            "latency_ms": m.get("latency_ms", 0),
            "params": m.get("params", m.get("student_params",
                     m.get("total_params", 0))),
        })

    return {
        "strategies": strategies,
        "energy": energy.get("savings", {}),
        "models": models,
        "gpu_available": torch.cuda.is_available(),
        "task_status": task_status,
    }


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("Starting GreenAI API server...")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True,
                reload_excludes=["*.pth", "*.json", "*.csv"])
