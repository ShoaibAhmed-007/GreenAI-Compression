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
    POST /api/compress/preloaded — Compress a prepared baseline model
  POST /api/compress/dynamic  — Upload any model + apply dynamic compression
    POST /api/compress          — Deprecated (use /api/compress/preloaded or /api/compress/dynamic)
  GET  /api/compare           — Side-by-side comparison of all strategies

Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any
import base64
import glob
import io
import os
import json
import copy
import math
import logging
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
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

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
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "Assets")

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

logger = logging.getLogger("greenai.model_compare")


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
CIFAR10_DISPLAY_SIZE = 224
MODEL_COMPARE_TOP_K = 3
MODEL_COMPARE_FOCUS_CLASSES = {"cat", "dog"}
MODEL_COMPARE_CONFIDENCE_DROP_ALERT_PCT = 15.0
MODEL_COMPARE_BLUR_EDGE_THRESHOLD = 8.0
MODEL_COMPARE_TEST_IMAGES_SUBDIR = "test-images"
COMPARE_IMAGE_MAX_UPLOAD_MB = 10
COMPARE_IMAGE_MAX_UPLOAD_BYTES = COMPARE_IMAGE_MAX_UPLOAD_MB * 1024 * 1024

SAMPLE_CLASS_ORDER = [3, 5, 2, 0, 1, 4, 6, 7, 8, 9]

_COMPARISON_SAMPLE_CACHE: Optional[list[dict]] = None

CIFAR10_CLASS_ALIASES = {
    "airplane": "airplane",
    "aeroplane": "airplane",
    "plane": "airplane",
    "automobile": "automobile",
    "auto": "automobile",
    "car": "automobile",
    "bird": "bird",
    "cat": "cat",
    "deer": "deer",
    "dog": "dog",
    "frog": "frog",
    "horse": "horse",
    "ship": "ship",
    "truck": "truck",
}

ASSET_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

VALID_STRATEGIES = [
    "smart",
    "maximize_speed",
    "minimize_size",
    "preserve_accuracy",
    "pruning",
    "quantization",
    "hybrid",
    "kd",
]


def _strategy_label(strategy: str) -> str:
    labels = {
        "smart": "Auto-Green (Smart)",
        "maximize_speed": "Preset: Maximize Speed",
        "minimize_size": "Preset: Minimize Size",
        "preserve_accuracy": "Preset: Preserve Accuracy",
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
                return _normalize_compression_result(key, entry)
    return None


def _normalize_cifar10_label(raw_label: str) -> Optional[str]:
    token = re.sub(r"[^a-z0-9]", "", str(raw_label).strip().lower())
    return CIFAR10_CLASS_ALIASES.get(token)


def _folder_has_supported_images(folder_path: str) -> bool:
    if not os.path.isdir(folder_path):
        return False
    for root, _, files in os.walk(folder_path):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in ASSET_IMAGE_EXTENSIONS:
                return True
    return False


def _get_assets_root() -> str:
    override = str(os.getenv("MODEL_COMPARISON_ASSETS_DIR", "")).strip()
    if override:
        candidate = override
        if not os.path.isabs(candidate):
            candidate = os.path.abspath(os.path.join(BACKEND_DIR, "..", candidate))
        if _folder_has_supported_images(candidate):
            return candidate
        raise FileNotFoundError(
            f"MODEL_COMPARISON_ASSETS_DIR='{override}' has no supported images."
        )

    preferred_test_dir = os.path.join(ASSETS_DIR, MODEL_COMPARE_TEST_IMAGES_SUBDIR)
    if _folder_has_supported_images(preferred_test_dir):
        return preferred_test_dir

    if _folder_has_supported_images(ASSETS_DIR):
        return ASSETS_DIR

    raise FileNotFoundError(
        "No supported images found for model comparison. "
        "Add files under Assets/test-images (preferred) or Assets."
    )


def _extract_cifar10_label_from_filename(file_name: str) -> Optional[str]:
    stem = os.path.splitext(file_name)[0]
    direct = _normalize_cifar10_label(stem)
    if direct is not None:
        return direct

    tokens = [token for token in re.split(r"[^a-z0-9]+", stem.lower()) if token]
    for token in tokens:
        mapped = _normalize_cifar10_label(token)
        if mapped is not None:
            return mapped
    return None


class _FlatAssetsDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], classes: list[str], transform=None):
        self.samples = samples
        self.classes = classes
        self.transform = transform
        self.loader = self._pil_loader

    @staticmethod
    def _pil_loader(path: str):
        with Image.open(path) as image:
            return image.convert("RGB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, class_idx = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, class_idx


def _build_flat_assets_dataset(root: str, transform=None):
    image_files: list[str] = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in ASSET_IMAGE_EXTENSIONS:
            image_files.append(name)

    if not image_files:
        raise RuntimeError(
            f"No image files found in '{root}'. Supported formats: {sorted(ASSET_IMAGE_EXTENSIONS)}"
        )

    files_by_label: dict[str, list[str]] = {label: [] for label in CIFAR10_CLASSES}
    unmapped: list[str] = []

    for name in sorted(image_files):
        label = _extract_cifar10_label_from_filename(name)
        if label is None or label not in CIFAR10_CLASSES:
            unmapped.append(name)
            continue
        files_by_label[label].append(os.path.join(root, name))

    active_labels = [label for label in CIFAR10_CLASSES if files_by_label[label]]
    if not active_labels:
        preview = ", ".join(unmapped[:5])
        raise RuntimeError(
            "Could not infer CIFAR-10 labels from Assets filenames. "
            "Use names like cat_01.png, dog-image.jpg, truck-2.png. "
            f"Sample unmatched files: {preview or 'none'}."
        )

    samples: list[tuple[str, int]] = []
    for class_idx, class_name in enumerate(active_labels):
        for path in files_by_label[class_name]:
            samples.append((path, class_idx))

    return _FlatAssetsDataset(samples=samples, classes=active_labels, transform=transform)


def _build_assets_imagefolder(transform=None):
    root = _get_assets_root()

    class_dirs = [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]

    if class_dirs:
        dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    else:
        dataset = _build_flat_assets_dataset(root=root, transform=transform)

    if len(dataset.samples) == 0:
        raise RuntimeError(f"No images found in Assets folder '{root}'.")

    dataset_to_cifar_idx: dict[int, int] = {}
    dataset_to_cifar_label: dict[int, str] = {}
    seen_labels: dict[str, str] = {}

    for dataset_class_idx, class_name in enumerate(dataset.classes):
        normalized = _normalize_cifar10_label(class_name)
        if normalized is None or normalized not in CIFAR10_CLASSES:
            raise RuntimeError(
                f"Class folder '{class_name}' is not a valid CIFAR-10 class. "
                f"Allowed classes: {CIFAR10_CLASSES}."
            )
        if normalized in seen_labels:
            raise RuntimeError(
                f"Duplicate CIFAR-10 class mapping detected: '{class_name}' and '{seen_labels[normalized]}' both map to '{normalized}'."
            )
        seen_labels[normalized] = class_name
        dataset_to_cifar_idx[dataset_class_idx] = CIFAR10_CLASSES.index(normalized)
        dataset_to_cifar_label[dataset_class_idx] = normalized

    return dataset, dataset_to_cifar_idx, dataset_to_cifar_label


def _build_assets_inference_transform():
    # EXACT same preprocessing requested for CIFAR-10-style inference.
    return transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _build_assets_dataloader(sample_ids: Optional[list[int]] = None, batch_size: int = 16):
    dataset, dataset_to_cifar_idx, dataset_to_cifar_label = _build_assets_imagefolder(
        transform=_build_assets_inference_transform()
    )

    ordered_ids: list[int]
    if sample_ids is None:
        ordered_ids = list(range(len(dataset)))
        data_source = dataset
    else:
        ordered_ids = [int(i) for i in sample_ids]
        if not ordered_ids:
            raise ValueError("sample_ids cannot be empty for batch prediction.")
        for idx in ordered_ids:
            if idx < 0 or idx >= len(dataset):
                raise ValueError(f"Sample image id {idx} is out of range.")
        data_source = Subset(dataset, ordered_ids)

    loader = DataLoader(
        data_source,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, ordered_ids, dataset_to_cifar_idx, dataset_to_cifar_label


def _resize_for_display(image: Image.Image, target_size: int) -> Image.Image:
    """Resize with high-quality interpolation while preserving aspect ratio."""
    if target_size <= 0:
        return image

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        return image

    scale = min(target_size / float(src_w), target_size / float(src_h))
    resampling = getattr(Image, "Resampling", Image)
    resize_mode = resampling.LANCZOS if scale >= 1.0 else resampling.BICUBIC
    resized = ImageOps.contain(image, (target_size, target_size), method=resize_mode)

    # CIFAR images are tiny (32x32); use a stronger enhancement pass for demo clarity.
    if scale >= 1.0:
        cv2_enhanced = None
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            rgb = np.array(resized.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Local contrast enhancement in LAB space.
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(6, 6))
            l_chan = clahe.apply(l_chan)
            lab = cv2.merge((l_chan, a_chan, b_chan))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Edge-aware denoise + unsharp mask for a clearer look.
            enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=45, sigmaSpace=45)
            blur = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
            sharp = cv2.addWeighted(enhanced, 1.45, blur, -0.45, 0)

            sharp = np.clip(sharp, 0, 255).astype("uint8")
            rgb_out = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
            cv2_enhanced = Image.fromarray(rgb_out)
        except Exception:
            cv2_enhanced = None

        if cv2_enhanced is not None:
            resized = cv2_enhanced
        else:
            resized = ImageEnhance.Contrast(resized).enhance(1.12)
            resized = ImageEnhance.Color(resized).enhance(1.07)
            resized = ImageEnhance.Sharpness(resized).enhance(1.55)
            resized = resized.filter(ImageFilter.UnsharpMask(radius=1.4, percent=195, threshold=2))

    new_w, new_h = resized.size

    if new_w == target_size and new_h == target_size:
        return resized

    # Keep the original aspect ratio and avoid geometric distortion.
    canvas = Image.new("RGB", (target_size, target_size), (246, 248, 252))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def _pil_image_to_data_url(
    image,
    display_size: Optional[int] = None,
) -> str:
    render_image = image
    if display_size is not None:
        target_size = max(int(display_size), CIFAR10_INPUT_SIZE)
        render_image = _resize_for_display(render_image, target_size)

    buffer = io.BytesIO()
    render_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_cifar10_inference_transform(size: int = CIFAR10_INPUT_SIZE):
    target_size = max(int(size), CIFAR10_INPUT_SIZE)
    return transforms.Compose([
        transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _build_comparison_sample_images(limit: int = 10) -> list[dict]:
    global _COMPARISON_SAMPLE_CACHE

    dataset, dataset_to_cifar_idx, dataset_to_cifar_label = _build_assets_imagefolder(transform=None)

    ids_by_class: dict[int, list[int]] = {idx: [] for idx in range(len(CIFAR10_CLASSES))}
    for sample_id, (_, dataset_label_idx) in enumerate(dataset.samples):
        class_idx = dataset_to_cifar_idx[int(dataset_label_idx)]
        ids_by_class.setdefault(class_idx, []).append(sample_id)

    selected_ids: list[int] = []
    selected_set: set[int] = set()

    def _add_sample(sample_id: int):
        if sample_id in selected_set:
            return
        selected_set.add(sample_id)
        selected_ids.append(sample_id)

    focus_indices = [CIFAR10_CLASSES.index(label) for label in ("cat", "dog")]
    focus_quota_per_class = max(1, limit // 4)

    # Prioritize cat/dog coverage first when available.
    for class_idx in focus_indices:
        for sample_id in ids_by_class.get(class_idx, [])[:focus_quota_per_class]:
            _add_sample(sample_id)
            if len(selected_ids) >= limit:
                break
        if len(selected_ids) >= limit:
            break

    if len(selected_ids) < limit:
        for class_id in SAMPLE_CLASS_ORDER:
            if class_id in focus_indices:
                continue
            candidate_ids = ids_by_class.get(class_id, [])
            if candidate_ids:
                _add_sample(candidate_ids[0])
            if len(selected_ids) >= limit:
                break

    if len(selected_ids) < limit:
        fallback_priority = []
        for class_idx in focus_indices:
            fallback_priority.extend(ids_by_class.get(class_idx, []))
        fallback_priority.extend(range(len(dataset.samples)))
        for sample_id in fallback_priority:
            _add_sample(sample_id)
            if len(selected_ids) >= limit:
                break

    samples: list[dict] = []
    for sample_id in selected_ids:
        path, dataset_label_idx = dataset.samples[sample_id]
        image = dataset.loader(path)
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")

        class_label = dataset_to_cifar_label[int(dataset_label_idx)]
        class_idx = dataset_to_cifar_idx[int(dataset_label_idx)]
        rel_path = os.path.relpath(path, os.path.join(BACKEND_DIR, "..")).replace("\\", "/")
        is_focus_class = class_label in MODEL_COMPARE_FOCUS_CLASSES

        samples.append({
            "id": int(sample_id),
            "label": class_label,
            "class_index": int(class_idx),
            "is_focus_class": bool(is_focus_class),
            "source_path": rel_path,
            "image_data_url": _pil_image_to_data_url(image, display_size=CIFAR10_DISPLAY_SIZE),
        })

    _COMPARISON_SAMPLE_CACHE = samples
    return samples


def _get_sample_image_by_id(sample_id: int):
    dataset, dataset_to_cifar_idx, _ = _build_assets_imagefolder(transform=None)
    if sample_id < 0 or sample_id >= len(dataset.samples):
        raise ValueError(f"Sample image id {sample_id} is out of range.")
    path, dataset_label_idx = dataset.samples[sample_id]
    image = dataset.loader(path)
    if isinstance(image, Image.Image) and image.mode != "RGB":
        image = image.convert("RGB")
    label_idx = dataset_to_cifar_idx[int(dataset_label_idx)]
    return image, int(label_idx)


def _path_is_within(candidate_path: str, root_path: str) -> bool:
    try:
        candidate_abs = os.path.abspath(candidate_path)
        root_abs = os.path.abspath(root_path)
        return os.path.commonpath([candidate_abs, root_abs]) == root_abs
    except Exception:
        return False


def _resolve_true_label_from_path(image_path: str) -> Optional[str]:
    parent_token = os.path.basename(os.path.dirname(image_path))
    parent_label = _normalize_cifar10_label(parent_token)
    if parent_label in CIFAR10_CLASSES:
        return parent_label

    filename_label = _extract_cifar10_label_from_filename(os.path.basename(image_path))
    if filename_label in CIFAR10_CLASSES:
        return filename_label

    return None


def _resolve_assets_image_by_path(sample_image_path: str) -> tuple[Image.Image, Optional[str], str]:
    raw = str(sample_image_path or "").strip()
    if not raw:
        raise ValueError("sample_image_path is required.")

    candidate = raw
    if not os.path.isabs(candidate):
        candidate = os.path.abspath(os.path.join(PROJECT_ROOT, candidate))
    else:
        candidate = os.path.abspath(candidate)

    allowed_roots = {os.path.abspath(ASSETS_DIR)}
    try:
        allowed_roots.add(os.path.abspath(_get_assets_root()))
    except Exception:
        # If preferred assets root is unavailable, fall back to base Assets validation.
        pass

    if not any(_path_is_within(candidate, root) for root in allowed_roots):
        raise ValueError("sample_image_path must point to an image inside Assets.")

    if not os.path.isfile(candidate):
        raise ValueError(f"sample_image_path does not exist: {raw}")

    ext = os.path.splitext(candidate)[1].lower()
    if ext not in ASSET_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. Allowed: {sorted(ASSET_IMAGE_EXTENSIONS)}"
        )

    try:
        with Image.open(candidate) as img:
            image = img.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to open sample image: {exc}")

    inferred_label = _resolve_true_label_from_path(candidate)
    rel_path = os.path.relpath(candidate, PROJECT_ROOT).replace("\\", "/")
    return image, inferred_label, rel_path


def _decode_uploaded_image_bytes(image_bytes: bytes, filename_hint: str = "upload") -> Image.Image:
    if not image_bytes:
        raise ValueError("Uploaded image file is empty.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to decode uploaded image '{filename_hint}': {exc}")


def _parse_compressed_model_selection(compressed_key_raw: str) -> tuple[str, str]:
    selected = str(compressed_key_raw or "").strip()
    if "::" not in selected:
        raise ValueError(
            "Invalid compressed model selection format. Expected '<model_key>::<strategy>'."
        )

    model_key_raw, strategy_raw = selected.split("::", 1)
    model_key = _normalize_model_key(model_key_raw)
    strategy_key = _normalize_model_key(strategy_raw)

    if not model_key or not strategy_key:
        raise ValueError("Compressed model selection is incomplete.")

    return model_key, strategy_key


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

    candidate_paths = [
        os.path.join(PRETRAINED_DIR, f"{key}_baseline.pth"),
        os.path.join(MODELS_DIR, f"{key}_baseline.pth"),
    ]
    if key == "resnet18":
        candidate_paths.append(os.path.join(MODELS_DIR, "baseline_model.pth"))

    selected_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if selected_path is None:
        raise FileNotFoundError(
            (
                f"No baseline weights found for '{key}'. "
                f"Expected one of: {candidate_paths}"
            )
        )

    state = torch.load(selected_path, map_location=device)
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


def _resolve_comparison_temperature() -> float:
    try:
        temperature = float(os.getenv("MODEL_COMPARISON_TEMPERATURE", DEFAULT_COMPARISON_TEMPERATURE))
    except (TypeError, ValueError):
        temperature = DEFAULT_COMPARISON_TEMPERATURE
    if not math.isfinite(temperature) or temperature <= 0:
        temperature = DEFAULT_COMPARISON_TEMPERATURE
    return float(temperature)


def _prediction_from_probability_row(
    prob_row: torch.Tensor,
    input_size: int,
    temperature: float,
    top_k: int = MODEL_COMPARE_TOP_K,
) -> dict:
    confidence, pred_idx = torch.max(prob_row, dim=0)
    k = min(max(1, int(top_k)), int(prob_row.shape[0]))
    top_values, top_indices = torch.topk(prob_row, k=k)

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
        "input_size": int(input_size),
        "temperature": round(float(temperature), 3),
        "top_k": top_predictions,
    }


def _estimate_blur_edge_score(image: Image.Image) -> Optional[float]:
    try:
        gray = image.convert("L")
        edge_map = gray.filter(ImageFilter.FIND_EDGES)
        return float(ImageStat.Stat(edge_map).var[0])
    except Exception:
        return None


def _collect_preprocess_warnings(
    raw_tensor: torch.Tensor,
    normalized_tensor: torch.Tensor,
    blur_edge_score: Optional[float],
    expected_size: int,
) -> tuple[list[str], dict[str, float]]:
    warnings: list[str] = []

    shape = tuple(raw_tensor.shape)
    expected_shape = (3, expected_size, expected_size)
    if shape != expected_shape:
        warnings.append(
            f"Unexpected tensor shape {shape}; expected {expected_shape}."
        )

    raw_min = float(raw_tensor.min().item())
    raw_max = float(raw_tensor.max().item())
    if raw_min < -1e-6 or raw_max > 1.0 + 1e-6:
        warnings.append(
            f"Raw tensor range looks incorrect: [{raw_min:.4f}, {raw_max:.4f}] (expected [0, 1])."
        )

    normalized_mean = float(normalized_tensor.mean().item())
    normalized_std = float(normalized_tensor.std().item())
    if abs(normalized_mean) > 4.0:
        warnings.append(
            f"Normalized tensor mean is unusually high ({normalized_mean:.3f})."
        )
    if normalized_std < 0.05:
        warnings.append(
            f"Normalized tensor std is unusually low ({normalized_std:.3f})."
        )

    if blur_edge_score is not None and blur_edge_score < MODEL_COMPARE_BLUR_EDGE_THRESHOLD:
        warnings.append(
            f"Image may be blurry (edge score {blur_edge_score:.2f} < {MODEL_COMPARE_BLUR_EDGE_THRESHOLD:.2f})."
        )

    stats = {
        "raw_min": round(raw_min, 4),
        "raw_max": round(raw_max, 4),
        "normalized_mean": round(normalized_mean, 4),
        "normalized_std": round(normalized_std, 4),
        "blur_edge_score": round(float(blur_edge_score), 4) if blur_edge_score is not None else -1.0,
    }
    return warnings, stats


def _build_tta_images(image: Image.Image, enable_tta: bool) -> list[Image.Image]:
    variants = [image]
    if enable_tta:
        variants.extend([
            ImageOps.mirror(image),
            ImageEnhance.Brightness(image).enhance(1.15),
            ImageEnhance.Brightness(image).enhance(0.85),
        ])
    return variants


def _predict_probabilities_for_batch(
    model: nn.Module,
    batch_tensor: torch.Tensor,
    device,
    temperature: float,
) -> torch.Tensor:
    from compress import _extract_logits

    inputs = batch_tensor.to(device)
    with torch.no_grad():
        logits = _extract_logits(model(inputs))
        probs = F.softmax(logits / temperature, dim=1)
    return probs


def _format_topk_for_log(prediction: dict) -> str:
    entries = []
    for item in (prediction.get("top_k") or []):
        class_name = str(item.get("class_name", "unknown"))
        probability = float(item.get("probability", 0.0))
        entries.append(f"{class_name}:{probability:.2f}%")
    return ", ".join(entries) if entries else "n/a"


def _build_comparison_case_diagnostics(
    sample_id: int,
    true_label: str,
    baseline_pred: dict,
    compressed_pred: dict,
) -> dict[str, Any]:
    is_focus = true_label in MODEL_COMPARE_FOCUS_CLASSES
    baseline_conf = float(baseline_pred.get("confidence", 0.0))
    compressed_conf = float(compressed_pred.get("confidence", 0.0))
    confidence_drop = round(baseline_conf - compressed_conf, 2)
    significant_drop = confidence_drop >= MODEL_COMPARE_CONFIDENCE_DROP_ALERT_PCT

    prediction_match = int(baseline_pred.get("predicted_index", -1)) == int(compressed_pred.get("predicted_index", -1))
    baseline_correct_by_label = str(baseline_pred.get("predicted_class", "")).lower() == true_label.lower()
    compressed_correct_by_label = str(compressed_pred.get("predicted_class", "")).lower() == true_label.lower()

    if is_focus or significant_drop or (not prediction_match):
        logger.info(
            "[ModelCompare] sample=%s true=%s | baseline=%s (%.2f%%) [%s] | compressed=%s (%.2f%%) [%s] | delta=%+.2f%%",
            sample_id,
            true_label,
            baseline_pred.get("predicted_class"),
            baseline_conf,
            _format_topk_for_log(baseline_pred),
            compressed_pred.get("predicted_class"),
            compressed_conf,
            _format_topk_for_log(compressed_pred),
            compressed_conf - baseline_conf,
        )

    if significant_drop:
        logger.warning(
            "[ModelCompare] Significant compressed confidence drop for sample %s: baseline %.2f%% -> compressed %.2f%%",
            sample_id,
            baseline_conf,
            compressed_conf,
        )

    if is_focus and (not baseline_correct_by_label or not compressed_correct_by_label):
        logger.warning(
            "[ModelCompare] Focus-class mismatch sample=%s true=%s baseline=%s compressed=%s",
            sample_id,
            true_label,
            baseline_pred.get("predicted_class"),
            compressed_pred.get("predicted_class"),
        )

    return {
        "is_focus_class": bool(is_focus),
        "confidence_drop_percent": confidence_drop,
        "significant_confidence_drop": bool(significant_drop),
        "confidence_drop_threshold": MODEL_COMPARE_CONFIDENCE_DROP_ALERT_PCT,
        "prediction_match": bool(prediction_match),
        "baseline_correct_by_label": bool(baseline_correct_by_label),
        "compressed_correct_by_label": bool(compressed_correct_by_label),
    }


def _predict_model_on_image(model: nn.Module, image, device) -> dict:
    return _predict_model_on_asset_images_batch(model, [image], device=device)[0]


def _predict_model_on_preprocessed_batch(model: nn.Module, batch_tensor: torch.Tensor, device) -> list[dict]:
    """Predict on tensors already normalized for CIFAR-10; resize only if needed."""
    from compress import detect_input_shape

    model.eval()
    temperature = _resolve_comparison_temperature()

    model_input_shape = detect_input_shape(model)
    model_input_size = int(model_input_shape[-1]) if len(model_input_shape) >= 4 else CIFAR10_INPUT_SIZE
    if model_input_size <= 0:
        model_input_size = CIFAR10_INPUT_SIZE

    inputs = batch_tensor.to(device)
    if int(inputs.shape[-1]) != model_input_size:
        # Keep CIFAR preprocessing fixed, then adapt spatial size for model compatibility.
        inputs = F.interpolate(
            inputs,
            size=(model_input_size, model_input_size),
            mode='bilinear',
            align_corners=False,
        )

    probs = _predict_probabilities_for_batch(
        model,
        inputs,
        device=device,
        temperature=temperature,
    )

    predictions: list[dict] = []

    for row in probs:
        predictions.append(
            _prediction_from_probability_row(
                row,
                input_size=model_input_size,
                temperature=temperature,
                top_k=MODEL_COMPARE_TOP_K,
            )
        )

    return predictions


def _predict_model_on_asset_image(model: nn.Module, image: Image.Image, device) -> dict:
    return _predict_model_on_asset_images_batch(model, [image], device=device)[0]


def _get_model_input_size(model: nn.Module) -> int:
    from compress import detect_input_shape

    input_shape = detect_input_shape(model)
    model_input_size = int(input_shape[-1]) if len(input_shape) >= 4 else CIFAR10_INPUT_SIZE
    if model_input_size <= 0:
        model_input_size = CIFAR10_INPUT_SIZE
    return model_input_size


def _predict_model_on_asset_images_batch(
    model: nn.Module,
    images: list[Image.Image],
    device,
    enable_tta: bool = False,
) -> list[dict]:
    """Preprocess Assets images at model-native input size before batched inference."""
    if not images:
        return []

    model_input_size = _get_model_input_size(model)
    temperature = _resolve_comparison_temperature()

    resize = transforms.Resize(
        (model_input_size, model_input_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    predictions: list[dict] = []

    for image in images:
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")

        blur_edge_score = _estimate_blur_edge_score(image)
        variants = _build_tta_images(image, enable_tta=enable_tta)

        normalized_variants = []
        quality_warnings: list[str] = []
        normalization_stats: dict[str, float] = {}

        for idx, variant in enumerate(variants):
            raw_tensor = to_tensor(resize(variant))
            normalized_tensor = normalize(raw_tensor.clone())
            normalized_variants.append(normalized_tensor)

            if idx == 0:
                quality_warnings, normalization_stats = _collect_preprocess_warnings(
                    raw_tensor,
                    normalized_tensor,
                    blur_edge_score,
                    expected_size=model_input_size,
                )

        probs = _predict_probabilities_for_batch(
            model,
            torch.stack(normalized_variants, dim=0),
            device=device,
            temperature=temperature,
        )
        averaged_probs = probs.mean(dim=0)

        prediction = _prediction_from_probability_row(
            averaged_probs,
            input_size=model_input_size,
            temperature=temperature,
            top_k=MODEL_COMPARE_TOP_K,
        )
        prediction["quality_warnings"] = quality_warnings
        prediction["normalization_stats"] = normalization_stats
        prediction["blur_edge_score"] = (
            round(float(blur_edge_score), 4)
            if blur_edge_score is not None and math.isfinite(blur_edge_score)
            else None
        )
        prediction["tta_variants"] = len(variants)

        if quality_warnings:
            logger.warning(
                "[ModelCompare] Preprocess warnings for input_size=%s: %s",
                model_input_size,
                " | ".join(quality_warnings),
            )

        predictions.append(prediction)

    return predictions


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

            normalized_entry = _normalize_compression_result(model_key, entry)

            strategy = _normalize_model_key(
                normalized_entry.get("strategy") or normalized_entry.get("compression_method") or ""
            )
            if not strategy or strategy == "baseline":
                continue

            artifact = _find_compressed_artifact(model_key, strategy)
            if artifact is None:
                continue

            option_key = f"{model_key}::{strategy}"
            model_name = (
                normalized_entry.get("model_name")
                or PRELOADED_MODELS.get(model_key, {}).get("name")
                or model_key
            )
            size_mb = _first_numeric_value(normalized_entry, ("size_MB", "compressed_size_MB"))
            co2_kg = _first_numeric_value(
                normalized_entry,
                (
                    "compressed_total_emissions_kg",
                    "compressed_benchmark_total_emissions_kg",
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


def _baseline_training_snapshot(model_key: str) -> tuple[Optional[float], Optional[float]]:
    """Return immutable baseline training CO2/energy for a model from baseline source files."""
    key = _normalize_model_key(model_key)
    baseline_metrics = load_baseline_results().get(key, {})

    training_co2 = _first_numeric_value(
        baseline_metrics,
        ("training_co2_kg", "training_emissions_kg", "co2_kg", "emissions_kg"),
    )
    training_energy = _first_numeric_value(
        baseline_metrics,
        ("training_energy_kwh", "energy_kwh"),
    )

    return training_co2, training_energy


def _with_immutable_baseline_metrics(model_key: str, result: dict) -> dict:
    """Attach immutable baseline training metrics so baseline fields cannot drift after compression."""
    if not isinstance(result, dict):
        return result

    normalized = copy.deepcopy(result)
    baseline_co2, baseline_energy = _baseline_training_snapshot(model_key)

    if baseline_co2 is not None:
        normalized["baseline_training_co2_kg"] = baseline_co2
        if _first_numeric_value(normalized, ("baseline_total_emissions_kg", "baseline_benchmark_total_emissions_kg")) is None:
            normalized["baseline_total_emissions_kg"] = baseline_co2

    if baseline_energy is not None:
        normalized["baseline_training_energy_kwh"] = baseline_energy
        if _first_numeric_value(normalized, ("baseline_total_energy_kwh", "baseline_benchmark_total_energy_kwh")) is None:
            normalized["baseline_total_energy_kwh"] = baseline_energy

    return normalized


def _compute_percent_reduction(baseline_value, compressed_value) -> Optional[float]:
    """Compute percent reduction with robust numeric guards."""
    baseline = _to_float(baseline_value)
    compressed = _to_float(compressed_value)
    if baseline is None or compressed is None or baseline <= 0:
        return None
    return round(((baseline - compressed) / baseline) * 100.0, 2)


def _normalize_compression_result(model_key: str, result: dict) -> dict:
    """Normalize one compression result into stable baseline/compressed fields.

    This keeps immutable baseline-training metrics separate while preserving
    fair benchmark totals when they exist.
    """
    if not isinstance(result, dict):
        return result

    normalized = _with_immutable_baseline_metrics(model_key, result)

    baseline_size_mb = _first_numeric_value(
        normalized,
        ("baseline_size_MB", "original_size_MB", "baseline_model_size_MB"),
    )
    compressed_size_mb = _first_numeric_value(
        normalized,
        ("compressed_size_MB", "size_MB"),
    )

    if baseline_size_mb is not None:
        normalized["baseline_size_MB"] = round(baseline_size_mb, 2)
    if compressed_size_mb is not None:
        normalized["compressed_size_MB"] = round(compressed_size_mb, 2)
        if _first_numeric_value(normalized, ("size_MB",)) is None:
            normalized["size_MB"] = round(compressed_size_mb, 2)

    baseline_total_co2 = _first_numeric_value(
        normalized,
        (
            "baseline_total_emissions_kg",
            "baseline_benchmark_total_emissions_kg",
            "baseline_training_co2_kg",
        ),
    )
    compressed_total_co2 = _first_numeric_value(
        normalized,
        (
            "compressed_total_emissions_kg",
            "compressed_benchmark_total_emissions_kg",
            "co2_kg",
            "emissions_kg",
            "inference_co2_kg",
            "inference_emissions_kg",
        ),
    )

    projected_compressed_co2 = None
    if (
        baseline_total_co2 is not None
        and baseline_total_co2 > 0
        and baseline_size_mb is not None
        and baseline_size_mb > 0
        and compressed_size_mb is not None
        and compressed_size_mb >= 0
    ):
        projected_compressed_co2 = round(
            baseline_total_co2 * (compressed_size_mb / baseline_size_mb),
            12,
        )

    # Keep a projected value for diagnostics, but do not override measured emissions.
    if projected_compressed_co2 is not None:
        normalized["compressed_projected_emissions_kg"] = projected_compressed_co2
        if compressed_total_co2 is None:
            compressed_total_co2 = projected_compressed_co2

    if baseline_total_co2 is not None:
        normalized["baseline_total_emissions_kg"] = round(baseline_total_co2, 12)
    if compressed_total_co2 is not None:
        normalized["compressed_total_emissions_kg"] = round(compressed_total_co2, 12)
        normalized["co2_kg"] = round(compressed_total_co2, 12)
        normalized["emissions_kg"] = round(compressed_total_co2, 12)

    baseline_total_energy = _first_numeric_value(
        normalized,
        (
            "baseline_total_energy_kwh",
            "baseline_benchmark_total_energy_kwh",
            "baseline_training_energy_kwh",
        ),
    )
    compressed_total_energy = _first_numeric_value(
        normalized,
        (
            "compressed_total_energy_kwh",
            "compressed_benchmark_total_energy_kwh",
            "energy_kwh",
            "inference_energy_kwh",
            "training_energy_kwh",
        ),
    )

    if baseline_total_energy is not None:
        normalized["baseline_total_energy_kwh"] = round(baseline_total_energy, 12)
    if compressed_total_energy is not None:
        normalized["compressed_total_energy_kwh"] = round(compressed_total_energy, 12)
        normalized["energy_kwh"] = round(compressed_total_energy, 12)

    co2_reduction = _compute_percent_reduction(baseline_total_co2, compressed_total_co2)
    if co2_reduction is not None:
        normalized["emissions_reduction_percent"] = co2_reduction

    energy_reduction = _compute_percent_reduction(baseline_total_energy, compressed_total_energy)
    if energy_reduction is not None:
        normalized["energy_reduction_percent"] = energy_reduction

    size_reduction = _compute_percent_reduction(baseline_size_mb, compressed_size_mb)
    if size_reduction is not None:
        normalized["size_reduction_percent"] = size_reduction

    normalized["model_key"] = _normalize_model_key(model_key or normalized.get("model_key") or "")
    return normalized


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
    normalized_result = _normalize_compression_result(model_key, result)
    strategy = normalized_result.get("strategy", normalized_result.get("compression_method", "unknown"))
    history[model_key] = [
        r for r in history[model_key]
        if (r.get("strategy") or r.get("compression_method")) != strategy
    ]
    history[model_key].append(normalized_result)

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

    # Merge optional curated baseline catalog entries (fills gaps if per-model files are missing).
    catalog_path = os.path.join(RESULTS_DIR, "baseline_all_models.json")
    if os.path.exists(catalog_path):
        try:
            with open(catalog_path, "r") as f:
                catalog = json.load(f)
        except Exception:
            catalog = None

        if isinstance(catalog, dict):
            for raw_key, payload in catalog.items():
                if not isinstance(payload, dict):
                    continue

                model_key = _normalize_model_key(raw_key or payload.get("model_key") or payload.get("model_name") or "")
                if not model_key:
                    continue

                entry = baselines.get(model_key, {
                    "model_key": model_key,
                    "model_name": payload.get("model_name") or raw_key,
                    "params_label": payload.get("params_label"),
                    "total_params": _to_int(payload.get("total_params", payload.get("parameters"))),
                    "input_size": payload.get("input_size"),
                    "dataset": payload.get("dataset", "CIFAR10"),
                    "accuracy": _first_numeric_value(payload, ("accuracy", "baseline_accuracy", "accuracy_top1")),
                    "size_MB": _first_numeric_value(payload, ("size_MB", "original_size_MB", "baseline_size_MB")),
                    "latency_ms": payload.get("latency_ms"),
                    "training_co2_kg": None,
                    "training_energy_kwh": None,
                    "result_updated_at": None,
                    "result_file": os.path.basename(catalog_path),
                    "status": "ready" if payload.get("status", "ready") != "error" else "error",
                })

                if entry.get("training_co2_kg") is None:
                    entry["training_co2_kg"] = _first_numeric_value(
                        payload,
                        ("training_co2_kg", "training_emissions_kg", "co2_kg", "emissions_kg"),
                    )
                if entry.get("training_energy_kwh") is None:
                    entry["training_energy_kwh"] = _first_numeric_value(
                        payload,
                        ("training_energy_kwh", "energy_kwh"),
                    )

                if isinstance(entry.get("accuracy"), (int, float)):
                    entry["accuracy"] = round(float(entry["accuracy"]), 2)
                if isinstance(entry.get("size_MB"), (int, float)):
                    entry["size_MB"] = round(float(entry["size_MB"]), 2)

                baselines[model_key] = entry

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
    enable_tta: bool = False


class ModelComparisonBatchRequest(BaseModel):
    sample_ids: list[int]
    baseline_model_key: str
    compressed_model_key: str
    batch_size: int = 16
    enable_tta: bool = False


@app.get("/api/model-comparison/options")
async def get_model_comparison_options():
    """Return baseline and compressed model options for the image-comparison UI."""
    return _build_model_comparison_options()


@app.get("/api/model-comparison/sample-images")
async def get_model_comparison_sample_images(
    limit: int = Query(default=10, ge=1, le=200),
):
    """Return sample images from local Assets folder mapped to CIFAR-10 labels."""
    try:
        samples = _build_comparison_sample_images(limit=limit)
        return {"samples": samples}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load sample images: {exc}")


@app.post("/compare-image")
@app.post("/api/compare-image")
async def compare_image(
    baseline_model_key: str = Form(...),
    compressed_model_key: str = Form(...),
    sample_image_path: Optional[str] = Form(None),
    enable_tta: bool = Form(False),
    image_file: Optional[UploadFile] = File(None),
):
    """Compare baseline vs compressed model on one sample-path or uploaded image."""
    baseline_key = _normalize_model_key(baseline_model_key)
    if not baseline_key:
        raise HTTPException(status_code=400, detail="Baseline model key is required.")

    try:
        compressed_model_key_norm, compressed_strategy = _parse_compressed_model_selection(compressed_model_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    history_entry = _find_compression_history_entry(compressed_model_key_norm, compressed_strategy)
    if history_entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No compression history found for model '{compressed_model_key_norm}' "
                f"with strategy '{compressed_strategy}'."
            ),
        )

    has_upload = image_file is not None and bool(str(image_file.filename or "").strip())
    has_sample_path = bool(str(sample_image_path or "").strip())
    if has_upload == has_sample_path:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one image input: image_file (upload) OR sample_image_path.",
        )

    source = "upload" if has_upload else "sample"
    source_path = None
    upload_filename = None

    try:
        if has_upload:
            upload_filename = str(image_file.filename or "uploaded_image")
            content_type = str(image_file.content_type or "").lower().strip()
            if content_type and not content_type.startswith("image/"):
                raise ValueError("Uploaded file must be an image (content-type image/*).")

            image_bytes = await image_file.read()
            if len(image_bytes) > COMPARE_IMAGE_MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"Uploaded image is too large ({len(image_bytes)} bytes). "
                    f"Maximum allowed is {COMPARE_IMAGE_MAX_UPLOAD_MB} MB."
                )

            image = _decode_uploaded_image_bytes(image_bytes, upload_filename)
            inferred_true_label = _extract_cifar10_label_from_filename(upload_filename)
        else:
            image, inferred_true_label, source_path = _resolve_assets_image_by_path(str(sample_image_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load image input: {exc}")

    preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") if compressed_strategy == "quantization" else preferred_device

    try:
        baseline_model = _load_baseline_model_for_comparison(baseline_key, device=device)
    except Exception as exc:
        if device.type != "cuda":
            raise HTTPException(status_code=500, detail=f"Failed to load baseline model: {exc}")
        try:
            device = torch.device("cpu")
            baseline_model = _load_baseline_model_for_comparison(baseline_key, device=device)
        except Exception as cpu_exc:
            raise HTTPException(status_code=500, detail=f"Failed to load baseline model: {cpu_exc}")

    try:
        compressed_model, compressed_artifact = _load_compressed_model_for_comparison(
            compressed_model_key_norm,
            compressed_strategy,
            device=device,
        )
    except Exception as exc:
        if device.type != "cuda":
            raise HTTPException(status_code=500, detail=f"Failed to load compressed model: {exc}")
        try:
            device = torch.device("cpu")
            baseline_model = _load_baseline_model_for_comparison(baseline_key, device=device)
            compressed_model, compressed_artifact = _load_compressed_model_for_comparison(
                compressed_model_key_norm,
                compressed_strategy,
                device=device,
            )
        except Exception as cpu_exc:
            raise HTTPException(status_code=500, detail=f"Failed to load compressed model: {cpu_exc}")

    baseline_pred = _predict_model_on_asset_images_batch(
        baseline_model,
        [image],
        device=device,
        enable_tta=bool(enable_tta),
    )[0]
    compressed_pred = _predict_model_on_asset_images_batch(
        compressed_model,
        [image],
        device=device,
        enable_tta=bool(enable_tta),
    )[0]

    confidence_delta = round(
        float(compressed_pred.get("confidence", 0.0)) - float(baseline_pred.get("confidence", 0.0)),
        2,
    )
    prediction_match = int(baseline_pred.get("predicted_index", -1)) == int(compressed_pred.get("predicted_index", -1))

    true_label = inferred_true_label if inferred_true_label in CIFAR10_CLASSES else None
    true_label_index = CIFAR10_CLASSES.index(true_label) if true_label in CIFAR10_CLASSES else None

    baseline_correct = None
    compressed_correct = None
    if true_label_index is not None:
        baseline_correct = int(baseline_pred.get("predicted_index", -1)) == int(true_label_index)
        compressed_correct = int(compressed_pred.get("predicted_index", -1)) == int(true_label_index)

    diag_label = true_label if true_label is not None else "unknown"
    case_diagnostics = _build_comparison_case_diagnostics(
        sample_id=-1,
        true_label=diag_label,
        baseline_pred=baseline_pred,
        compressed_pred=compressed_pred,
    )

    quality_warnings = sorted(set(
        list(baseline_pred.get("quality_warnings") or []) +
        list(compressed_pred.get("quality_warnings") or [])
    ))

    mismatch_warning = None
    if not prediction_match:
        mismatch_warning = (
            f"Baseline predicted '{baseline_pred.get('predicted_class')}', while compressed predicted "
            f"'{compressed_pred.get('predicted_class')}'."
        )

    return {
        "input": {
            "source": source,
            "sample_image_path": source_path,
            "upload_filename": upload_filename,
            "true_label": true_label,
            "image_data_url": _pil_image_to_data_url(image, display_size=CIFAR10_DISPLAY_SIZE),
        },
        "baseline": {
            "model_key": baseline_key,
            "class": baseline_pred.get("predicted_class"),
            "confidence": baseline_pred.get("confidence"),
            "top3": baseline_pred.get("top_k", []),
            "prediction": baseline_pred,
        },
        "compressed": {
            "model_key": compressed_model_key_norm,
            "strategy": compressed_strategy,
            "class": compressed_pred.get("predicted_class"),
            "confidence": compressed_pred.get("confidence"),
            "top3": compressed_pred.get("top_k", []),
            "prediction": compressed_pred,
            "artifact": os.path.basename(compressed_artifact),
        },
        "comparison": {
            "prediction_match": bool(prediction_match),
            "prediction_mismatch_warning": mismatch_warning,
            "confidence_delta_percent": confidence_delta,
            "confidence_drop_alert": bool(case_diagnostics.get("significant_confidence_drop", False)),
            "confidence_drop_threshold_percent": MODEL_COMPARE_CONFIDENCE_DROP_ALERT_PCT,
            "baseline_correct": baseline_correct,
            "compressed_correct": compressed_correct,
        },
        "diagnostics": {
            "case": case_diagnostics,
            "quality_warnings": quality_warnings,
        },
        "preprocessing": {
            "resize": "model_native",
            "baseline_input_size": int(baseline_pred.get("input_size", _get_model_input_size(baseline_model))),
            "compressed_input_size": int(compressed_pred.get("input_size", _get_model_input_size(compressed_model))),
            "normalize_mean": list(CIFAR10_MEAN),
            "normalize_std": list(CIFAR10_STD),
            "tta_enabled": bool(enable_tta),
            "tta_variants": int(max(
                int(baseline_pred.get("tta_variants", 1)),
                int(compressed_pred.get("tta_variants", 1)),
            )),
        },
        "device": str(device),
    }


@app.post("/api/model-comparison/compare")
async def compare_models_on_image(req: ModelComparisonRequest):
    """Run baseline and compressed model inference on one selected Assets image."""
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
        sample_id = int(req.sample_id)
        image_dataset, image_dataset_to_cifar_idx, _ = _build_assets_imagefolder(transform=None)
        if sample_id < 0 or sample_id >= len(image_dataset.samples):
            raise ValueError(f"Sample image id {sample_id} is out of range.")

        image_path, image_dataset_label_idx = image_dataset.samples[sample_id]
        sample_rel_path = os.path.relpath(image_path, os.path.join(BACKEND_DIR, "..")).replace("\\", "/")
        image = image_dataset.loader(image_path)
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")

        label_idx = int(image_dataset_to_cifar_idx[int(image_dataset_label_idx)])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sample image selection: {exc}")

    preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") if compressed_strategy == "quantization" else preferred_device

    try:
        baseline_model = _load_baseline_model_for_comparison(baseline_key, device=device)
    except Exception as exc:
        if device.type != "cuda":
            raise HTTPException(status_code=500, detail=f"Failed to load baseline model: {exc}")
        try:
            device = torch.device("cpu")
            baseline_model = _load_baseline_model_for_comparison(baseline_key, device=device)
        except Exception as cpu_exc:
            raise HTTPException(status_code=500, detail=f"Failed to load baseline model: {cpu_exc}")

    try:
        compressed_model, compressed_artifact = _load_compressed_model_for_comparison(
            compressed_model_key,
            compressed_strategy,
            device=device,
        )
    except Exception as exc:
        if device.type != "cuda":
            raise HTTPException(status_code=500, detail=f"Failed to load compressed model: {exc}")
        try:
            device = torch.device("cpu")
            baseline_model = _load_baseline_model_for_comparison(baseline_key, device=device)
            compressed_model, compressed_artifact = _load_compressed_model_for_comparison(
                compressed_model_key,
                compressed_strategy,
                device=device,
            )
        except Exception as cpu_exc:
            raise HTTPException(status_code=500, detail=f"Failed to load compressed model: {cpu_exc}")

    baseline_pred = _predict_model_on_asset_images_batch(
        baseline_model,
        [image],
        device=device,
        enable_tta=bool(req.enable_tta),
    )[0]
    compressed_pred = _predict_model_on_asset_images_batch(
        compressed_model,
        [image],
        device=device,
        enable_tta=bool(req.enable_tta),
    )[0]

    baseline_metrics = load_baseline_results().get(baseline_key, {})
    baseline_accuracy = _first_numeric_value(
        baseline_metrics,
        ("accuracy", "accuracy_top1", "baseline_accuracy"),
    )
    baseline_latency = _first_numeric_value(baseline_metrics, ("latency_ms",))
    baseline_size = _first_numeric_value(
        baseline_metrics,
        ("size_MB", "original_size_MB", "baseline_size_MB"),
    )
    baseline_co2 = _first_numeric_value(
        baseline_metrics,
        ("training_co2_kg", "training_emissions_kg", "co2_kg", "emissions_kg"),
    )
    baseline_energy_kwh = _first_numeric_value(
        baseline_metrics,
        ("training_energy_kwh", "energy_kwh"),
    )

    if baseline_size is None:
        baseline_size = _first_numeric_value(history_entry, ("baseline_size_MB", "original_size_MB"))

    if baseline_accuracy is None:
        baseline_accuracy = _first_numeric_value(history_entry, ("baseline_accuracy",))

    if baseline_latency is None:
        baseline_latency = _first_numeric_value(history_entry, ("baseline_latency_ms",))

    if baseline_co2 is None:
        baseline_co2 = _first_numeric_value(
            history_entry,
            (
                "baseline_total_emissions_kg",
                "baseline_benchmark_total_emissions_kg",
                "baseline_training_co2_kg",
                "baseline_inference_co2_kg",
            ),
        )

    if baseline_energy_kwh is None:
        baseline_energy_kwh = _first_numeric_value(
            history_entry,
            (
                "baseline_total_energy_kwh",
                "baseline_benchmark_total_energy_kwh",
                "baseline_training_energy_kwh",
                "baseline_inference_energy_kwh",
            ),
        )

    compressed_accuracy = _first_numeric_value(
        history_entry,
        ("compressed_accuracy", "accuracy", "accuracy_top1"),
    )
    compressed_latency = _first_numeric_value(history_entry, ("latency_ms",))
    compressed_size = _first_numeric_value(history_entry, ("size_MB", "compressed_size_MB"))
    compressed_co2 = _first_numeric_value(
        history_entry,
        (
            "compressed_total_emissions_kg",
            "compressed_benchmark_total_emissions_kg",
            "co2_kg",
            "emissions_kg",
            "inference_co2_kg",
            "inference_emissions_kg",
        ),
    )
    compressed_energy_kwh = _first_numeric_value(
        history_entry,
        (
            "compressed_total_energy_kwh",
            "compressed_benchmark_total_energy_kwh",
            "training_energy_kwh",
            "energy_kwh",
            "inference_energy_kwh",
        ),
    )

    confidence_delta = round(compressed_pred["confidence"] - baseline_pred["confidence"], 2)
    prediction_match = baseline_pred["predicted_index"] == compressed_pred["predicted_index"]
    baseline_correct = baseline_pred["predicted_index"] == int(label_idx)
    compressed_correct = compressed_pred["predicted_index"] == int(label_idx)

    accuracy_delta = None
    if baseline_accuracy is not None and compressed_accuracy is not None:
        accuracy_delta = round(compressed_accuracy - baseline_accuracy, 2)

    latency_reduction = None
    if baseline_latency is not None and baseline_latency > 0 and compressed_latency is not None:
        latency_reduction = round(((baseline_latency - compressed_latency) / baseline_latency) * 100, 2)

    size_reduction = None
    if baseline_size is not None and baseline_size > 0 and compressed_size is not None:
        size_reduction = round(((baseline_size - compressed_size) / baseline_size) * 100, 2)

    if (
        compressed_co2 is None and
        baseline_co2 is not None and
        baseline_co2 > 0 and
        baseline_size is not None and
        baseline_size > 0 and
        compressed_size is not None and
        compressed_size >= 0
    ):
        compressed_co2 = round(baseline_co2 * (compressed_size / baseline_size), 12)

    co2_reduction = None
    if baseline_co2 is not None and baseline_co2 > 0 and compressed_co2 is not None:
        co2_reduction = round(((baseline_co2 - compressed_co2) / baseline_co2) * 100, 2)
        if co2_reduction > 80:
            print(
                "[SanityWarning] CO2 reduction above 80% in model comparison. "
                "Check size metadata and emissions inputs."
            )

    energy_reduction = None
    if baseline_energy_kwh is not None and baseline_energy_kwh > 0 and compressed_energy_kwh is not None:
        energy_reduction = round(((baseline_energy_kwh - compressed_energy_kwh) / baseline_energy_kwh) * 100, 2)

    size_reduction_text = f"{size_reduction:.2f}%" if size_reduction is not None else "N/A"
    summary_parts = [
        f"Prediction agreement: {'Yes' if prediction_match else 'No'}",
        f"Size reduction: {size_reduction_text}",
        f"Confidence delta: {confidence_delta:+.2f}%",
    ]
    if accuracy_delta is not None:
        summary_parts.append(f"Dataset accuracy delta: {accuracy_delta:+.2f}%")
    if latency_reduction is not None:
        summary_parts.append(f"Latency reduction: {latency_reduction:.2f}%")
    if co2_reduction is not None:
        summary_parts.append(f"CO2 reduction: {co2_reduction:.2f}%")
    summary = " | ".join(summary_parts)

    true_label = (
        CIFAR10_CLASSES[label_idx]
        if 0 <= int(label_idx) < len(CIFAR10_CLASSES)
        else f"class_{int(label_idx)}"
    )

    case_diagnostics = _build_comparison_case_diagnostics(
        sample_id=sample_id,
        true_label=true_label,
        baseline_pred=baseline_pred,
        compressed_pred=compressed_pred,
    )
    aggregated_quality_warnings = sorted(set(
        list(baseline_pred.get("quality_warnings") or []) +
        list(compressed_pred.get("quality_warnings") or [])
    ))
    confidence_drop_alert = bool(case_diagnostics.get("significant_confidence_drop", False))

    return {
        "sample": {
            "id": int(sample_id),
            "true_label": true_label,
            "is_focus_class": bool(true_label in MODEL_COMPARE_FOCUS_CLASSES),
            "source_path": sample_rel_path,
            "image_data_url": _pil_image_to_data_url(image, display_size=CIFAR10_DISPLAY_SIZE),
        },
        "baseline": {
            "model_key": baseline_key,
            "model_name": baseline_metrics.get("model_name") or baseline_key,
            "prediction": baseline_pred,
            "accuracy": baseline_accuracy,
            "latency_ms": baseline_latency,
            "size_MB": baseline_size,
            "co2_kg": baseline_co2,
            "energy_kwh": baseline_energy_kwh,
        },
        "compressed": {
            "model_key": compressed_model_key,
            "strategy": compressed_strategy,
            "model_name": history_entry.get("model_name") or compressed_model_key,
            "strategy_label": _strategy_label(compressed_strategy),
            "prediction": compressed_pred,
            "accuracy": compressed_accuracy,
            "latency_ms": compressed_latency,
            "size_MB": compressed_size,
            "co2_kg": compressed_co2,
            "energy_kwh": compressed_energy_kwh,
            "artifact": os.path.basename(compressed_artifact),
        },
        "comparison": {
            "confidence_delta_percent": confidence_delta,
            "size_reduction_percent": size_reduction,
            "co2_reduction_percent": co2_reduction,
            "energy_reduction_percent": energy_reduction,
            "accuracy_delta_percent": accuracy_delta,
            "latency_reduction_percent": latency_reduction,
            "prediction_match": prediction_match,
            "baseline_correct": baseline_correct,
            "compressed_correct": compressed_correct,
            "confidence_drop_alert": confidence_drop_alert,
            "confidence_drop_threshold_percent": MODEL_COMPARE_CONFIDENCE_DROP_ALERT_PCT,
            "full_dataset_metrics": {
                "baseline_accuracy_percent": baseline_accuracy,
                "compressed_accuracy_percent": compressed_accuracy,
                "accuracy_delta_percent": accuracy_delta,
                "baseline_latency_ms": baseline_latency,
                "compressed_latency_ms": compressed_latency,
                "latency_reduction_percent": latency_reduction,
                "baseline_size_MB": baseline_size,
                "compressed_size_MB": compressed_size,
                "size_reduction_percent": size_reduction,
                "baseline_co2_kg": baseline_co2,
                "compressed_co2_kg": compressed_co2,
                "co2_reduction_percent": co2_reduction,
                "baseline_energy_kwh": baseline_energy_kwh,
                "compressed_energy_kwh": compressed_energy_kwh,
                "energy_reduction_percent": energy_reduction,
            },
            "summary": summary,
        },
        "diagnostics": {
            "case": case_diagnostics,
            "quality_warnings": aggregated_quality_warnings,
            "baseline_top3": baseline_pred.get("top_k", []),
            "compressed_top3": compressed_pred.get("top_k", []),
        },
        "preprocessing": {
            "resize": "model_native",
            "baseline_input_size": int(baseline_pred.get("input_size", _get_model_input_size(baseline_model))),
            "compressed_input_size": int(compressed_pred.get("input_size", _get_model_input_size(compressed_model))),
            "normalize_mean": list(CIFAR10_MEAN),
            "normalize_std": list(CIFAR10_STD),
            "dataset_loader": "Assets + model-specific transform",
            "tta_enabled": bool(req.enable_tta),
            "tta_variants": int(max(
                int(baseline_pred.get("tta_variants", 1)),
                int(compressed_pred.get("tta_variants", 1)),
            )),
        },
        "device": str(device),
    }


@app.post("/api/model-comparison/compare-batch")
async def compare_models_on_batch(req: ModelComparisonBatchRequest):
    """Run baseline and compressed inference on a selected batch of Assets images."""
    sample_ids = [int(i) for i in (req.sample_ids or [])]
    if not sample_ids:
        raise HTTPException(status_code=400, detail="sample_ids cannot be empty.")
    if len(sample_ids) > 256:
        raise HTTPException(status_code=400, detail="sample_ids exceeds limit (256).")

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

    history_entry = _find_compression_history_entry(compressed_model_key, compressed_strategy)
    if history_entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No compression history found for model '{compressed_model_key}' "
                f"with strategy '{compressed_strategy}'."
            ),
        )

    preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") if compressed_strategy == "quantization" else preferred_device

    try:
        baseline_model = _load_baseline_model_for_comparison(baseline_key, device=device)
    except Exception as exc:
        if device.type != "cuda":
            raise HTTPException(status_code=500, detail=f"Failed to load baseline model: {exc}")
        try:
            device = torch.device("cpu")
            baseline_model = _load_baseline_model_for_comparison(baseline_key, device=device)
        except Exception as cpu_exc:
            raise HTTPException(status_code=500, detail=f"Failed to load baseline model: {cpu_exc}")

    try:
        compressed_model, compressed_artifact = _load_compressed_model_for_comparison(
            compressed_model_key,
            compressed_strategy,
            device=device,
        )
    except Exception as exc:
        if device.type != "cuda":
            raise HTTPException(status_code=500, detail=f"Failed to load compressed model: {exc}")
        try:
            device = torch.device("cpu")
            baseline_model = _load_baseline_model_for_comparison(baseline_key, device=device)
            compressed_model, compressed_artifact = _load_compressed_model_for_comparison(
                compressed_model_key,
                compressed_strategy,
                device=device,
            )
        except Exception as cpu_exc:
            raise HTTPException(status_code=500, detail=f"Failed to load compressed model: {cpu_exc}")

    try:
        raw_dataset, raw_dataset_to_cifar_idx, _ = _build_assets_imagefolder(transform=None)
        active_assets_root = _get_assets_root()
        ordered_ids = [int(i) for i in sample_ids]
        for idx in ordered_ids:
            if idx < 0 or idx >= len(raw_dataset.samples):
                raise ValueError(f"Sample image id {idx} is out of range.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to build Assets dataloader: {exc}")

    batch_size = max(1, min(int(req.batch_size), 64))
    baseline_model_input_size = _get_model_input_size(baseline_model)
    compressed_model_input_size = _get_model_input_size(compressed_model)

    baseline_correct = 0
    compressed_correct = 0
    prediction_match = 0
    baseline_latency_total_ms = 0.0
    compressed_latency_total_ms = 0.0
    total = 0
    results = []
    per_class_totals = {label: 0 for label in CIFAR10_CLASSES}
    per_class_baseline_correct = {label: 0 for label in CIFAR10_CLASSES}
    per_class_compressed_correct = {label: 0 for label in CIFAR10_CLASSES}
    focus_misclassifications: list[dict[str, Any]] = []
    significant_confidence_drop_cases: list[dict[str, Any]] = []

    for start in range(0, len(ordered_ids), batch_size):
        current_ids = ordered_ids[start:start + batch_size]
        images: list[Image.Image] = []
        true_indices: list[int] = []

        for sample_id in current_ids:
            image_path, raw_dataset_label_idx = raw_dataset.samples[sample_id]
            image = raw_dataset.loader(image_path)
            if isinstance(image, Image.Image) and image.mode != "RGB":
                image = image.convert("RGB")

            images.append(image)
            true_indices.append(int(raw_dataset_to_cifar_idx[int(raw_dataset_label_idx)]))

        t0 = time.perf_counter()
        baseline_preds = _predict_model_on_asset_images_batch(
            baseline_model,
            images,
            device=device,
            enable_tta=bool(req.enable_tta),
        )
        baseline_latency_total_ms += (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        compressed_preds = _predict_model_on_asset_images_batch(
            compressed_model,
            images,
            device=device,
            enable_tta=bool(req.enable_tta),
        )
        compressed_latency_total_ms += (time.perf_counter() - t1) * 1000.0

        for i, sample_id in enumerate(current_ids):
            true_idx = int(true_indices[i])
            true_label = CIFAR10_CLASSES[int(true_idx)]
            per_class_totals[true_label] += 1

            b_pred = baseline_preds[i]
            c_pred = compressed_preds[i]

            if b_pred["predicted_index"] == true_idx:
                baseline_correct += 1
                per_class_baseline_correct[true_label] += 1
            if c_pred["predicted_index"] == true_idx:
                compressed_correct += 1
                per_class_compressed_correct[true_label] += 1
            if b_pred["predicted_index"] == c_pred["predicted_index"]:
                prediction_match += 1

            mapped_true_label = true_label
            confidence_delta = round(float(c_pred.get("confidence", 0.0)) - float(b_pred.get("confidence", 0.0)), 2)
            case_diagnostics = _build_comparison_case_diagnostics(
                sample_id=int(sample_id),
                true_label=true_label,
                baseline_pred=b_pred,
                compressed_pred=c_pred,
            )

            if case_diagnostics.get("significant_confidence_drop"):
                significant_confidence_drop_cases.append({
                    "sample_id": int(sample_id),
                    "true_label": true_label,
                    "baseline_confidence": float(b_pred.get("confidence", 0.0)),
                    "compressed_confidence": float(c_pred.get("confidence", 0.0)),
                    "confidence_drop_percent": float(case_diagnostics.get("confidence_drop_percent", 0.0)),
                })

            is_focus_class = mapped_true_label in MODEL_COMPARE_FOCUS_CLASSES
            if is_focus_class and (
                int(b_pred.get("predicted_index", -1)) != true_idx or
                int(c_pred.get("predicted_index", -1)) != true_idx
            ):
                focus_misclassifications.append({
                    "sample_id": int(sample_id),
                    "true_label": mapped_true_label,
                    "baseline_predicted_class": b_pred.get("predicted_class"),
                    "compressed_predicted_class": c_pred.get("predicted_class"),
                    "baseline_confidence": b_pred.get("confidence"),
                    "compressed_confidence": c_pred.get("confidence"),
                })

            sample_quality_warnings = sorted(set(
                list(b_pred.get("quality_warnings") or []) +
                list(c_pred.get("quality_warnings") or [])
            ))

            results.append({
                "sample_id": int(sample_id),
                "true_label": mapped_true_label,
                "is_focus_class": bool(is_focus_class),
                "confidence_delta_percent": confidence_delta,
                "quality_warnings": sample_quality_warnings,
                "image_data_url": _pil_image_to_data_url(images[i], display_size=CIFAR10_DISPLAY_SIZE),
                "baseline_prediction": b_pred,
                "compressed_prediction": c_pred,
                "diagnostics": case_diagnostics,
            })

            total += 1

    baseline_acc = round(100.0 * baseline_correct / max(total, 1), 2)
    compressed_acc = round(100.0 * compressed_correct / max(total, 1), 2)
    agreement = round(100.0 * prediction_match / max(total, 1), 2)
    baseline_latency_per_image = round(baseline_latency_total_ms / max(total, 1), 3)
    compressed_latency_per_image = round(compressed_latency_total_ms / max(total, 1), 3)

    per_class_accuracy: dict[str, dict[str, float]] = {}
    for class_name in CIFAR10_CLASSES:
        class_total = int(per_class_totals.get(class_name, 0))
        if class_total <= 0:
            continue
        per_class_accuracy[class_name] = {
            "count": class_total,
            "baseline_accuracy_percent": round(100.0 * per_class_baseline_correct[class_name] / class_total, 2),
            "compressed_accuracy_percent": round(100.0 * per_class_compressed_correct[class_name] / class_total, 2),
        }

    focus_class_accuracy = {
        label: per_class_accuracy.get(
            label,
            {
                "count": 0,
                "baseline_accuracy_percent": 0.0,
                "compressed_accuracy_percent": 0.0,
            },
        )
        for label in sorted(MODEL_COMPARE_FOCUS_CLASSES)
    }

    return {
        "count": total,
        "batch_size": batch_size,
        "baseline_model_key": baseline_key,
        "compressed_model_key": compressed_model_key,
        "compressed_strategy": compressed_strategy,
        "compressed_artifact": os.path.basename(compressed_artifact),
        "results": results,
        "summary": {
            "baseline_accuracy_percent": baseline_acc,
            "compressed_accuracy_percent": compressed_acc,
            "prediction_agreement_percent": agreement,
            "baseline_latency_ms_per_image": baseline_latency_per_image,
            "compressed_latency_ms_per_image": compressed_latency_per_image,
            "per_class_accuracy": per_class_accuracy,
            "focus_class_accuracy": focus_class_accuracy,
            "focus_misclassification_count": len(focus_misclassifications),
            "significant_confidence_drop_count": len(significant_confidence_drop_cases),
        },
        "diagnostics": {
            "focus_misclassifications": focus_misclassifications,
            "significant_confidence_drop_cases": significant_confidence_drop_cases,
        },
        "preprocessing": {
            "resize": "model_native",
            "baseline_input_size": int(baseline_model_input_size),
            "compressed_input_size": int(compressed_model_input_size),
            "normalize_mean": list(CIFAR10_MEAN),
            "normalize_std": list(CIFAR10_STD),
            "dataset_loader": "Assets + model-specific transform",
            "tta_enabled": bool(req.enable_tta),
            "assets_dir": os.path.relpath(active_assets_root, os.path.join(BACKEND_DIR, "..")).replace("\\", "/"),
        },
        "device": str(device),
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
        history = json.load(f)

    if not isinstance(history, dict):
        return {"history": {}}

    normalized_history = {}
    for model_key, entries in history.items():
        if not isinstance(entries, list):
            continue
        canonical_key = _normalize_model_key(model_key)
        normalized_history[canonical_key] = [
            _normalize_compression_result(canonical_key, entry)
            for entry in entries
            if isinstance(entry, dict)
        ]

    return {"history": normalized_history}


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
    history = load_json("compression_history.json")
    summary = load_json("compression_summary.json")
    energy = load_json("energy_report.json")
    evaluation = load_json("evaluation_report.json")

    if summary is None and not isinstance(history, dict):
        raise HTTPException(status_code=404, detail="No results found")

    comparison = []
    if isinstance(history, dict) and history:
        baselines = load_baseline_results()
        for model_key, entries in history.items():
            canonical_key = _normalize_model_key(model_key)
            if not isinstance(entries, list):
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                metrics = _normalize_compression_result(canonical_key, entry)
                strategy_key = _normalize_model_key(
                    metrics.get("strategy") or metrics.get("compression_method") or ""
                )
                if not strategy_key:
                    continue

                baseline_size = _first_numeric_value(
                    baselines.get(canonical_key, {}),
                    ("size_MB",),
                )
                compressed_size = _first_numeric_value(metrics, ("size_MB", "compressed_size_MB"))
                size_reduction = _first_numeric_value(metrics, ("size_reduction_percent",))
                if size_reduction is None:
                    size_reduction = _compute_percent_reduction(baseline_size, compressed_size)

                normalized_key = f"{canonical_key}::{strategy_key}"
                model_name = metrics.get("model_name") or baselines.get(canonical_key, {}).get("model_name") or canonical_key

                entry_obj = {
                    "key": normalized_key,
                    "model_key": canonical_key,
                    "model_name": model_name,
                    "strategy": strategy_key,
                    "strategy_label": _strategy_label(strategy_key),
                    "accuracy": _first_numeric_value(metrics, ("compressed_accuracy", "accuracy", "accuracy_top1")),
                    "size_MB": compressed_size,
                    "size_reduction_percent": size_reduction,
                    "latency_ms": _first_numeric_value(metrics, ("latency_ms", "compressed_latency_ms")),
                    "params": _to_int(metrics.get("params", metrics.get("student_params", metrics.get("total_params")))),
                    "training_co2_kg": _first_numeric_value(metrics, ("training_co2_kg", "training_emissions_kg")),
                    "inference_co2_kg": _first_numeric_value(metrics, ("inference_co2_kg", "inference_emissions_kg")),
                    "baseline_co2_kg": _first_numeric_value(metrics, ("baseline_training_co2_kg", "baseline_total_emissions_kg")),
                    "compressed_co2_kg": _first_numeric_value(metrics, ("compressed_total_emissions_kg", "co2_kg", "emissions_kg")),
                    "co2_kg": _first_numeric_value(metrics, ("compressed_total_emissions_kg", "co2_kg", "emissions_kg")),
                }

                # Attach optional evaluation metrics when keys happen to align.
                if evaluation and normalized_key in evaluation:
                    eval_data = evaluation[normalized_key]
                    entry_obj["accuracy_top5"] = eval_data.get("accuracy_top5", None)
                    entry_obj["flops_M"] = eval_data.get("flops_M", None)
                    entry_obj["sparsity_percent"] = eval_data.get("sparsity_percent", None)

                comparison.append(entry_obj)

        comparison.sort(key=lambda item: (str(item.get("model_name", "")).lower(), str(item.get("strategy", "")).lower()))

    elif isinstance(summary, dict):
        for key, metrics in summary.items():
            entry = {
                "strategy": key,
                "accuracy": metrics.get("accuracy", metrics.get("accuracy_top1", None)),
                "size_MB": metrics.get("size_MB", metrics.get("size_MB_compressed", metrics.get("size_MB_sparse", None))),
                "size_reduction_percent": metrics.get("size_reduction_percent", metrics.get("size_reduction_compressed_percent", 0)),
                "latency_ms": metrics.get("latency_ms", None),
                "params": metrics.get("params", metrics.get("student_params", metrics.get("total_params", None))),
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
        )
        if result.returncode != 0:
            task_status[task_key]["error"] = result.stderr[-500:]
        task_status[task_key]["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        task_status[task_key]["error"] = str(e)
    finally:
        task_status[task_key]["running"] = False


@app.post("/api/compress")
async def trigger_compression():
    """
    Deprecated legacy endpoint.
    Use /api/compress/preloaded for curated models or /api/compress/dynamic for uploads.
    """
    raise HTTPException(
        status_code=410,
        detail=(
            "Deprecated endpoint '/api/compress'. "
            "Use '/api/compress/preloaded' for baseline-model compression "
            "or '/api/compress/dynamic' for uploaded models."
        ),
    )


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
    valid_methods = VALID_STRATEGIES
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
            import torch as _torch
            from compress import run_compression

            # Explicitly pick the best available device so GPU is always used
            # when CUDA is present — never rely on the default fallback inside run_compression.
            _device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
            _device_label = (
                f"CUDA ({_torch.cuda.get_device_name(_device)})"
                if _device.type == 'cuda'
                else 'CPU'
            )

            # Surface device info in the very first status update
            preloaded_task["detail"] = f"Loading {req.model_name} on {_device_label}..."
            preloaded_task["progress"] = preloaded_task["detail"]

            def progress_cb(step, detail=''):
                preloaded_task["step"] = step
                preloaded_task["detail"] = detail
                preloaded_task["progress"] = detail

            result = run_compression(
                model_name=model_key,
                method=method,
                dataset=dataset,
                fine_tune_epochs=req.fine_tune_epochs,
                device=_device,
                progress_cb=progress_cb,
            )

            # Remove internal path info
            result.pop("saved_path", None)
            result = _normalize_compression_result(model_key, result)

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
    - strategy: smart | maximize_speed | minimize_size | preserve_accuracy | pruning | quantization | hybrid | kd
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
    valid_strategies = VALID_STRATEGIES
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
        dynamic_model_key = _normalize_model_key(result.get("model_key") or os.path.splitext(filename)[0])
        result = _normalize_compression_result(dynamic_model_key, result)

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
    history = load_json("compression_history.json") or {}
    summary = load_json("compression_summary.json") or {}
    energy = load_json("energy_report.json") or {}
    evaluation = load_json("evaluation_report.json") or {}
    models = list_models()
    baselines = load_baseline_results()

    # Build strategy cards from normalized compression history (multi-model aware).
    strategies = []
    latest_entries = {}
    if isinstance(history, dict):
        for model_key, entries in history.items():
            canonical_model_key = _normalize_model_key(model_key)
            if not isinstance(entries, list):
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                normalized_entry = _normalize_compression_result(canonical_model_key, entry)
                strategy_key = _normalize_model_key(
                    normalized_entry.get("strategy") or normalized_entry.get("compression_method") or ""
                )
                if not strategy_key or strategy_key == "baseline":
                    continue

                latest_entries[f"{canonical_model_key}::{strategy_key}"] = normalized_entry

    for composite_key in sorted(latest_entries.keys()):
        model_key, strategy_key = composite_key.split("::", 1)
        m = latest_entries[composite_key]
        baseline_meta = baselines.get(model_key, {})

        baseline_size = _first_numeric_value(baseline_meta, ("size_MB",))
        if baseline_size is None:
            baseline_size = _first_numeric_value(m, ("baseline_size_MB", "original_size_MB"))

        size_mb = _first_numeric_value(m, ("size_MB", "compressed_size_MB"))
        if size_mb is None:
            continue

        size_reduction = _compute_percent_reduction(baseline_size, size_mb)
        model_label = m.get("model_name") or baseline_meta.get("model_name") or model_key

        strategies.append({
            "key": composite_key,
            "name": f"{model_label} · {_strategy_label(strategy_key)}",
            "accuracy": _first_numeric_value(m, ("compressed_accuracy", "accuracy", "accuracy_top1")) or 0,
            "size_MB": round(size_mb, 2),
            "size_reduction": size_reduction if size_reduction is not None else 0,
            "latency_ms": _first_numeric_value(m, ("latency_ms", "compressed_latency_ms")),
            "params": _to_int(m.get("params", m.get("student_params", m.get("total_params")))) or 0,
        })

    # Backward-compatible fallback for older single-model summary files.
    if not strategies and isinstance(summary, dict):
        baseline_data = summary.get("baseline", {})
        baseline_size = _first_numeric_value(baseline_data, ("size_MB",)) or 44.81
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
            size = m.get("size_MB", m.get("size_MB_compressed", m.get("size_MB_sparse", baseline_size)))
            strategies.append({
                "key": key,
                "name": display_name,
                "accuracy": m.get("accuracy", 0),
                "size_MB": size,
                "size_reduction": round(100 * (baseline_size - size) / baseline_size, 2) if baseline_size > 0 else 0,
                "latency_ms": m.get("latency_ms", 0),
                "params": m.get("params", m.get("student_params", m.get("total_params", 0))),
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
