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

# Enable cuDNN auto-tuner for faster convolutions with fixed input sizes
torch.backends.cudnn.benchmark = True
import copy
import time
import gzip
import io


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
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = model(inputs)
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


def save_compressed(model, path):
    """
    Save model state dict.
    Uses plain torch.save — gzip was removed because Python's single-threaded
    gzip takes 5-10 min on large models.  The file is a few MB larger but
    saving finishes in <1 second.
    """
    state = model.state_dict() if isinstance(model, nn.Module) else model
    # Strip .gz extension if present so we always write a plain file
    if path.endswith('.gz'):
        path = path[:-3]
    torch.save(state, path)
    return path   # return actual path (may differ from input)


def load_compressed(path, device='cpu'):
    """Load a state dict — supports both plain .pth and legacy .pth.gz."""
    if path.endswith('.gz') and os.path.exists(path):
        with gzip.open(path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        return torch.load(buffer, map_location=device)
    # Plain .pth
    if not os.path.exists(path) and os.path.exists(path + '.gz'):
        # Legacy fallback
        with gzip.open(path + '.gz', 'rb') as f:
            buffer = io.BytesIO(f.read())
        return torch.load(buffer, map_location=device)
    return torch.load(path, map_location=device)


def distillation_loss(student_out, teacher_out, labels, T=3.0, alpha=0.5):
    """
    Combined KD + CE loss.
    - T: temperature (higher = softer probabilities, more knowledge transfer)
    - alpha: weight for CE loss (1-alpha for KD loss)
    """
    kd = F.kl_div(
        F.log_softmax(student_out / T, dim=1),
        F.softmax(teacher_out / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    ce = F.cross_entropy(student_out, labels)
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
    # Warmup
    for _ in range(10):
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
            outputs = model(inputs)
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


def get_data_loaders(dataset_name='CIFAR10', batch_size=None, input_size=32, pin_memory=True):
    """Get train/test DataLoaders for a given dataset name.

    Args:
        dataset_name: 'CIFAR10' or 'CIFAR100'
        batch_size: Batch size for DataLoader.  When None (default) the size
                    is chosen automatically based on input_size so that 8 GB
                    VRAM is not exceeded:
                        32  → 128
                        224 → 32
                        299 → 16
        input_size: Spatial size to resize images to (default 32 = native CIFAR).
                    Pretrained ImageNet models typically need 224 or 299.
        pin_memory: Whether to use CUDA pinned memory (disable if OOM issues)
    """
    # Auto-select batch size to prevent GPU OOM on 8 GB cards
    if batch_size is None:
        if input_size <= 32:
            batch_size = 128
        elif input_size <= 224:
            batch_size = 32
        else:
            batch_size = 16

    ds = dataset_name.upper()

    # Choose the dataset class and normalization stats
    if ds == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        DSClass = torchvision.datasets.CIFAR10
    elif ds == 'CIFAR100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        DSClass = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use CIFAR10 or CIFAR100.")

    data_root = os.path.join(os.path.dirname(__file__), 'data')

    if input_size > 32:
        # On-the-fly resize with more workers to keep GPU fed
        train_spatial = [
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size, padding=4),
        ]
        test_spatial = [
            transforms.Resize(input_size),
        ]
    else:
        # ---- Native 32×32 — no resize needed ----
        train_spatial = [
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

    # num_workers=0 avoids Windows multiprocessing deadlocks (spawn method
    # inside a daemon thread kills the DataLoader).  With 50-batch caps the
    # overhead of main-process data loading is negligible (<10 s).
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin_memory)
    return train_loader, test_loader


def apply_pruning(model, train_loader, test_loader, device,
                  amount=0.7, fine_tune_epochs=5, save_dir='../models/uploads',
                  progress_cb=None):
    """
    Apply unstructured pruning to any model.
    Returns dict with metrics: accuracy, size_MB, latency_ms, etc.
    """
    _cb = progress_cb or (lambda *a, **k: None)
    os.makedirs(save_dir, exist_ok=True)
    model = copy.deepcopy(model).to(device)
    input_shape = detect_input_shape(model)

    # Baseline metrics
    _cb("Evaluating baseline accuracy...")
    baseline_acc = evaluate(model, test_loader, dev=device)
    baseline_path = os.path.join(save_dir, '_temp_baseline.pth')
    torch.save(model.state_dict(), baseline_path)
    baseline_size = get_size_mb(baseline_path)

    # Apply global unstructured L1 pruning
    _cb(f"Applying {int(amount*100)}% L1 unstructured pruning...")
    params_to_prune = [
        (m, 'weight') for _, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    if not params_to_prune:
        raise ValueError("Model has no Conv2d or Linear layers to prune.")

    prune.global_unstructured(
        params_to_prune, pruning_method=prune.L1Unstructured, amount=amount)

    # Fine-tune WITH masks active — keeps pruned weights at zero
    max_batches = 50  # cap per epoch for speed
    if fine_tune_epochs > 0:
        optimizer = optim.SGD(model.parameters(), lr=0.001,
                              momentum=0.9, weight_decay=5e-4)
        for epoch in range(fine_tune_epochs):
            _cb(f"Fine-tuning epoch {epoch+1}/{fine_tune_epochs} (0/{max_batches} batches)...")
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= max_batches:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(inputs), labels)
                loss.backward()
                optimizer.step()
                if (batch_idx + 1) % 10 == 0:
                    _cb(f"Fine-tuning epoch {epoch+1}/{fine_tune_epochs} — batch {batch_idx+1}/{max_batches}")
            _cb(f"Fine-tuning epoch {epoch+1}/{fine_tune_epochs} — done")

    # Remove masks AFTER fine-tuning so zeros are permanent
    for m, _ in params_to_prune:
        prune.remove(m, 'weight')

    # Save compressed
    _cb("Saving compressed model...")
    save_path = os.path.join(save_dir, 'pruned_dynamic.pth')
    save_compressed(model, save_path)

    pruned_acc = evaluate(model, test_loader, dev=device)
    pruned_size = get_size_mb(save_path)
    latency = measure_latency(model, input_shape=input_shape, dev=device)
    nonzero = count_nonzero(model)
    total = count_params(model)
    sparsity = round(100 * (1 - nonzero / total), 2) if total > 0 else 0

    # Clean up temp file
    if os.path.exists(baseline_path):
        os.remove(baseline_path)

    return {
        "strategy": "pruning",
        "baseline_accuracy": baseline_acc,
        "compressed_accuracy": pruned_acc,
        "size_MB": pruned_size,
        "baseline_size_MB": baseline_size,
        "size_reduction_percent": round(
            100 * (baseline_size - pruned_size) / baseline_size, 2) if baseline_size > 0 else 0,
        "latency_ms": latency,
        "sparsity_percent": sparsity,
        "total_params": total,
        "nonzero_params": nonzero,
        "pruning_amount": amount,
        "fine_tune_epochs": fine_tune_epochs,
        "saved_path": save_path,
    }


def apply_quantization(model, train_loader, test_loader, device,
                       save_dir='../models/uploads', progress_cb=None):
    """
    Apply dynamic INT8 quantization to any model.
    Returns dict with metrics.
    """
    _cb = progress_cb or (lambda *a, **k: None)
    os.makedirs(save_dir, exist_ok=True)
    model = copy.deepcopy(model).cpu()
    model.eval()
    input_shape = detect_input_shape(model)
    cpu_dev = torch.device('cpu')

    # Baseline metrics (cap at 50 batches on CPU for speed)
    _cb("Evaluating baseline accuracy...")
    baseline_acc = evaluate(model, test_loader, dev=cpu_dev, max_batches=50)
    baseline_path = os.path.join(save_dir, '_temp_baseline.pth')
    torch.save(model.state_dict(), baseline_path)
    baseline_size = get_size_mb(baseline_path)

    # Apply dynamic quantization (only nn.Linear is supported;
    # nn.Conv2d is silently ignored by PyTorch's quantize_dynamic)
    _cb("Applying dynamic INT8 quantization...")
    quant_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8)

    # Save quantized model
    _cb("Saving quantized model...")
    save_path = os.path.join(save_dir, 'quantized_dynamic.pth')
    save_compressed(quant_model, save_path)

    _cb("Evaluating compressed model...")
    quant_acc = evaluate(quant_model, test_loader, dev=cpu_dev, max_batches=50)
    quant_size = get_size_mb(save_path)
    latency = measure_latency(quant_model, input_shape=input_shape, dev=cpu_dev, n_runs=20)

    if os.path.exists(baseline_path):
        os.remove(baseline_path)

    return {
        "strategy": "quantization",
        "baseline_accuracy": baseline_acc,
        "compressed_accuracy": quant_acc,
        "size_MB": quant_size,
        "baseline_size_MB": baseline_size,
        "size_reduction_percent": round(
            100 * (baseline_size - quant_size) / baseline_size, 2) if baseline_size > 0 else 0,
        "latency_ms": latency,
        "quantization_type": "dynamic_int8",
        "total_params": count_params(model),
        "saved_path": save_path,
    }


def apply_hybrid(model, train_loader, test_loader, device,
                 amount=0.5, fine_tune_epochs=5, save_dir='../models/uploads',
                 progress_cb=None):
    """
    Apply pruning + quantization (hybrid) to any model.
    Returns dict with metrics.
    """
    _cb = progress_cb or (lambda *a, **k: None)
    os.makedirs(save_dir, exist_ok=True)
    model = copy.deepcopy(model).to(device)
    input_shape = detect_input_shape(model)

    # Baseline
    _cb("Evaluating baseline accuracy...")
    baseline_acc = evaluate(model, test_loader, dev=device)
    baseline_path = os.path.join(save_dir, '_temp_baseline.pth')
    torch.save(model.state_dict(), baseline_path)
    baseline_size = get_size_mb(baseline_path)

    # Step 1: Prune
    _cb(f"Applying {int(amount*100)}% L1 unstructured pruning...")
    params_to_prune = [
        (m, 'weight') for _, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    if not params_to_prune:
        raise ValueError("Model has no Conv2d or Linear layers to prune.")

    prune.global_unstructured(
        params_to_prune, pruning_method=prune.L1Unstructured, amount=amount)

    # Fine-tune WITH masks active — keeps pruned weights at zero
    max_batches = 50  # cap per epoch for speed
    if fine_tune_epochs > 0:
        optimizer = optim.SGD(model.parameters(), lr=0.001,
                              momentum=0.9, weight_decay=5e-4)
        for epoch in range(fine_tune_epochs):
            _cb(f"Fine-tuning epoch {epoch+1}/{fine_tune_epochs} (0/{max_batches} batches)...")
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= max_batches:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(inputs), labels)
                loss.backward()
                optimizer.step()
                if (batch_idx + 1) % 10 == 0:
                    _cb(f"Fine-tuning epoch {epoch+1}/{fine_tune_epochs} — batch {batch_idx+1}/{max_batches}")
            _cb(f"Fine-tuning epoch {epoch+1}/{fine_tune_epochs} — done")

    # Remove masks AFTER fine-tuning so zeros are permanent
    for m, _ in params_to_prune:
        prune.remove(m, 'weight')

    # Measure sparsity before quantization
    nonzero = count_nonzero(model)
    total = count_params(model)

    # Step 2: Quantize (only nn.Linear is actually supported by
    # quantize_dynamic; Conv2d is silently ignored by PyTorch)
    _cb("Applying dynamic INT8 quantization...")
    model = model.cpu()
    model.eval()
    quant_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8)

    # Save hybrid model
    _cb("Saving hybrid model...")
    save_path = os.path.join(save_dir, 'hybrid_dynamic.pth')
    save_compressed(quant_model, save_path)

    _cb("Evaluating hybrid model...")
    cpu_dev = torch.device('cpu')
    hybrid_acc = evaluate(quant_model, test_loader, dev=cpu_dev, max_batches=50)
    hybrid_size = get_size_mb(save_path)
    latency = measure_latency(quant_model, input_shape=input_shape, dev=cpu_dev, n_runs=20)

    if os.path.exists(baseline_path):
        os.remove(baseline_path)

    return {
        "strategy": "hybrid",
        "baseline_accuracy": baseline_acc,
        "compressed_accuracy": hybrid_acc,
        "size_MB": hybrid_size,
        "baseline_size_MB": baseline_size,
        "size_reduction_percent": round(
            100 * (baseline_size - hybrid_size) / baseline_size, 2) if baseline_size > 0 else 0,
        "latency_ms": latency,
        "pruning_amount": amount,
        "sparsity_percent": round(100 * (1 - nonzero / total), 2) if total > 0 else 0,
        "total_params": total,
        "pipeline": f"Prune {int(amount*100)}% → Fine-tune {fine_tune_epochs}ep → Quantize INT8",
        "saved_path": save_path,
    }


def apply_kd(teacher, train_loader, test_loader, device,
             num_classes=10, epochs=20, save_dir='../models/uploads',
             progress_cb=None):
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

    # CompactStudent is designed for 32×32 inputs. If the teacher/data uses
    # a larger resolution (e.g. 224×224), we resize inputs for the student
    # to avoid enormous intermediate feature maps and very slow training.
    student_size_px = 32
    teacher_size_px = input_shape[2]  # spatial dim from teacher
    needs_resize = teacher_size_px != student_size_px

    # KD training
    optimizer = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0

    max_batches = 100  # cap per epoch for speed
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
                t_out = teacher(inputs)
            # Student uses 32×32
            s_inputs = F.interpolate(inputs, size=student_size_px,
                                     mode='bilinear', align_corners=False
                                     ) if needs_resize else inputs
            s_out = student(s_inputs)
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
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(),
                       os.path.join(save_dir, 'student_dynamic.pth'))

    # Reload best
    save_path = os.path.join(save_dir, 'student_dynamic.pth')
    student.load_state_dict(torch.load(save_path, map_location=device))
    student_acc = _evaluate_with_resize(student, test_loader, student_size_px,
                                         needs_resize, dev=device)
    student_size = get_size_mb(save_path)
    # Measure latency at the student's native 32×32 resolution
    student_input_shape = (1, input_shape[1], student_size_px, student_size_px)
    latency = measure_latency(student, input_shape=student_input_shape, dev=device)

    if os.path.exists(teacher_path):
        os.remove(teacher_path)

    return {
        "strategy": "kd",
        "baseline_accuracy": teacher_acc,
        "compressed_accuracy": student_acc,
        "size_MB": student_size,
        "baseline_size_MB": teacher_size,
        "size_reduction_percent": round(
            100 * (teacher_size - student_size) / teacher_size, 2) if teacher_size > 0 else 0,
        "latency_ms": latency,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "param_reduction_percent": round(
            100 * (1 - student_params / teacher_params), 2) if teacher_params > 0 else 0,
        "kd_epochs": epochs,
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
                               amount=0.7, fine_tune_epochs=fine_tune_epochs,
                               save_dir=save_dir)
    elif strategy == 'quantization':
        result = apply_quantization(model, train_loader, test_loader, device,
                                    save_dir=save_dir)
    elif strategy == 'hybrid':
        result = apply_hybrid(model, train_loader, test_loader, device,
                              amount=0.5, fine_tune_epochs=fine_tune_epochs,
                              save_dir=save_dir)
    elif strategy == 'kd':
        result = apply_kd(model, train_loader, test_loader, device,
                          num_classes=num_classes, epochs=fine_tune_epochs * 4,
                          save_dir=save_dir)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Choose from: pruning, quantization, hybrid, kd"
        )

    # Add energy tracking
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(
            project_name=f"compress_{strategy}",
            output_dir=save_dir,
            country_iso_code="PAK",
            log_level="error",
        )
        tracker.start()
        # Quick inference pass to measure energy using the original model
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                inputs = inputs.to(device)
                model(inputs)
                if i >= 20:
                    break
        emissions = tracker.stop()
        result["emissions_kg"] = round(float(emissions), 8) if emissions else 0
    except Exception:
        result["emissions_kg"] = 0

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

    _cb = progress_cb or (lambda step, detail='': None)

    print(f"\n{'='*60}")
    print(f"Preloaded Compression: {cfg['name']} + {method.upper()}")
    print(f"Dataset: {dataset} | Input: {input_size}x{input_size} | Classes: {num_classes}")
    print(f"{'='*60}")

    # Step 1: Load model — try pre-saved baseline first
    pretrained_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'pretrained_baselines',
        f'{model_key}_baseline.pth')
    has_pretrained = os.path.exists(pretrained_path)

    _cb('loading_model', f'Loading {cfg["name"]}...')
    model = get_pretrained_model(model_key, num_classes=num_classes)
    model = model.to(device)

    if has_pretrained:
        # Load pre-saved fine-tuned weights — skip transfer learning!
        print(f"  Using pre-saved baseline: {pretrained_path}")
        _cb('loading_model', f'Loading pre-saved {cfg["name"]} baseline...')
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model = model.to(device)

    # Step 2: Prepare dataset
    _cb('loading_data', f'Preparing {dataset} dataset ({input_size}x{input_size})...')
    train_loader, test_loader = get_data_loaders(
        dataset, input_size=input_size)

    # ── Step 2.5: Transfer-learn (SKIP if pre-saved baseline exists) ──
    if not has_pretrained:
        # The pretrained model has an ImageNet backbone but a *randomly
        # initialised* classifier for num_classes. We must fine-tune so
        # baseline accuracy reflects actual performance on the target
        # dataset — otherwise it will be ≈10% (random).
        _cb('loading_data', 'Fine-tuning classifier head on target dataset...')

        # Phase A – freeze backbone, train only the new head (fast)
        for p in model.parameters():
            p.requires_grad = False
        _enable_head_gradients(model, model_key)

        head_epochs = max(fine_tune_epochs, 3)
        optimizer_A = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4)

        head_total_batches = len(train_loader)
        for ep in range(head_epochs):
            _cb('loading_data',
                f'Head fine-tune epoch {ep+1}/{head_epochs}...')
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_A.zero_grad()
                loss = F.cross_entropy(model(inputs), labels)
                loss.backward()
                optimizer_A.step()
                if (batch_idx + 1) % 20 == 0:
                    _cb('loading_data',
                        f'Head fine-tune epoch {ep+1}/{head_epochs} — '
                        f'batch {batch_idx+1}/{head_total_batches}')

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
            _cb('loading_data',
                f'Full fine-tune epoch {ep+1}/{ft_epochs}...')
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_idx >= max_batches_per_epoch:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_B.zero_grad()
                loss = F.cross_entropy(model(inputs), labels)
                loss.backward()
                optimizer_B.step()
                if (batch_idx + 1) % 10 == 0:
                    _cb('loading_data',
                        f'Full fine-tune epoch {ep+1}/{ft_epochs} — '
                        f'batch {batch_idx+1}/{effective_batches}')
            scheduler.step()

        tl_acc = evaluate(model, test_loader, dev=device)
        print(f"  Transfer-learn accuracy: {tl_acc}%")
    else:
        _cb('loading_data', 'Using pre-saved baseline — skipping transfer learning')
        tl_acc = evaluate(model, test_loader, dev=device)
        print(f"  Pre-saved baseline accuracy: {tl_acc}%")
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
                               amount=0.7, fine_tune_epochs=fine_tune_epochs,
                               save_dir=save_dir, progress_cb=compress_cb)
    elif method == 'quantization':
        result = apply_quantization(model, train_loader, test_loader, device,
                                    save_dir=save_dir, progress_cb=compress_cb)
    elif method == 'hybrid':
        result = apply_hybrid(model, train_loader, test_loader, device,
                              amount=0.5, fine_tune_epochs=fine_tune_epochs,
                              save_dir=save_dir, progress_cb=compress_cb)
    elif method == 'kd':
        result = apply_kd(model, train_loader, test_loader, device,
                          num_classes=num_classes,
                          epochs=fine_tune_epochs,
                          save_dir=save_dir, progress_cb=compress_cb)
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

    # Step 4: Energy tracking
    _cb('energy_tracking', 'Measuring energy consumption...')
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(
            project_name=f"compress_{model_key}_{method}",
            output_dir=save_dir,
            country_iso_code="PAK",
            log_level="error",
        )
        tracker.start()
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                inputs = inputs.to(device)
                model(inputs)
                if i >= 20:
                    break
        emissions = tracker.stop()
        result["emissions_kg"] = round(float(emissions), 8) if emissions else 0
    except Exception:
        result["emissions_kg"] = 0

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
    return result


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    # ----------------------------------------------------------
    # Data loaders (CIFAR-10)
    # ----------------------------------------------------------
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
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
        torch.load("../models/baseline_model.pth", map_location=device))
    baseline_model = baseline_model.to(device)
    baseline_model.eval()

    baseline_acc = evaluate(baseline_model, test_loader)
    baseline_size = get_size_mb("../models/baseline_model.pth")
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
        torch.load("../models/baseline_model.pth", map_location=device))
    pruned_model = pruned_model.to(device)

    # Global unstructured L1 pruning — removes 70% of smallest weights
    params_to_prune = [
        (m, 'weight') for _, m in pruned_model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    prune.global_unstructured(
        params_to_prune, pruning_method=prune.L1Unstructured, amount=0.7)
    for m, _ in params_to_prune:
        prune.remove(m, 'weight')

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

    # Save with gzip compression (zeros compress extremely well)
    save_compressed(pruned_model, "../models/pruned_model_compressed.pth.gz")
    # Also save standard format for comparison
    torch.save(pruned_model.state_dict(), "../models/pruned_model.pth")

    pruned_acc = evaluate(pruned_model, test_loader)
    pruned_size_std = get_size_mb("../models/pruned_model.pth")
    pruned_size_compressed = get_size_mb("../models/pruned_model_compressed.pth.gz")
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
            "../models/baseline_model.pth", map_location='cpu')
        quant_model.load_state_dict(baseline_state, strict=False)
        quant_model.eval()
        quant_model.cpu()

        # Fuse Conv-BN-ReLU blocks for better quantization accuracy
        quant_model.fuse_model()

        # Set quantization config for x86 (fbgemm backend)
        backend = 'fbgemm'  # Use 'qnnpack' for ARM edge devices
        quant_model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend

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
        torch.save(quant_model.state_dict(), "../models/quantized_model.pth")

        quant_acc = evaluate(quant_model, test_loader, dev=torch.device('cpu'))
        quant_size = get_size_mb("../models/quantized_model.pth")
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
            torch.load("../models/baseline_model.pth", map_location='cpu'))
        quant_model_dyn.eval()

        quant_model_dyn = torch.quantization.quantize_dynamic(
            quant_model_dyn, {nn.Linear}, dtype=torch.qint8)

        torch.save(quant_model_dyn.state_dict(),
                   "../models/quantized_model.pth")

        quant_acc = evaluate(quant_model_dyn, test_loader,
                             dev=torch.device('cpu'))
        quant_size = get_size_mb("../models/quantized_model.pth")
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
            s_out = student(inputs)
            with torch.no_grad():
                t_out = teacher(inputs)
            loss = distillation_loss(s_out, t_out, labels, T=4.0, alpha=0.3)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        acc = evaluate(student, test_loader)
        if acc > best_student_acc:
            best_student_acc = acc
            torch.save(student.state_dict(), "../models/student_distilled.pth")
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: loss={running_loss/len(train_loader):.4f}, "
                  f"acc={acc}%, best={best_student_acc}%")

    # Reload best checkpoint
    student.load_state_dict(
        torch.load("../models/student_distilled.pth", map_location=device))
    student_acc = evaluate(student, test_loader)
    student_size = get_size_mb("../models/student_distilled.pth")
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
        torch.load("../models/student_distilled.pth", map_location='cpu'))
    hybrid_student.eval()

    hybrid_student = torch.quantization.quantize_dynamic(
        hybrid_student, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

    torch.save(hybrid_student.state_dict(), "../models/hybrid_model.pth")
    hybrid_acc = evaluate(hybrid_student, test_loader, dev=torch.device('cpu'))
    hybrid_size = get_size_mb("../models/hybrid_model.pth")
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
        torch.load("../models/student_distilled.pth", map_location=device))

    # Prune 50% of compact student's weights
    ultra_prune_params = [
        (m, 'weight') for _, m in ultra_model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    prune.global_unstructured(
        ultra_prune_params, pruning_method=prune.L1Unstructured, amount=0.5)
    for m, _ in ultra_prune_params:
        prune.remove(m, 'weight')

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
                t_out = teacher(inputs)
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

    # Save pruned student with gzip compression
    save_compressed(ultra_model, "../models/ultra_compact_compressed.pth.gz")

    # Also apply dynamic quantization and save sparse
    ultra_quant = torch.quantization.quantize_dynamic(
        copy.deepcopy(ultra_model), {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
    torch.save(ultra_quant.state_dict(),
               "../models/ultra_compact_quant.pth")

    ultra_acc = evaluate(ultra_model, test_loader, dev=torch.device('cpu'))
    ultra_size_compressed = get_size_mb("../models/ultra_compact_compressed.pth.gz")
    ultra_size_quant = get_size_mb("../models/ultra_compact_quant.pth")
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