# evaluate.py
"""
Green AI FYP — Phase 5: Comprehensive Model Evaluation
=======================================================
Loads all saved models (baseline + 5 compression strategies) and evaluates:
  - Top-1 / Top-5 accuracy
  - Model size (disk)
  - Parameter count & sparsity
  - Inference latency
  - FLOPs estimate (multiply-accumulate operations)
  - Per-class accuracy breakdown

Outputs a unified evaluation report to ../results/evaluation_report.json
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import sys
import json
import time
import gzip
import io

# Add parent dir so we can import CompactStudent from compress.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compress import (
    CompactStudent,
    load_compressed,
    _extract_logits,
    _configure_quantized_backend,
)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def evaluate_accuracy(model, loader, device, topk=(1, 5)):
    """Compute top-k accuracy."""
    model.eval()
    maxk = max(topk)
    correct = {k: 0 for k in topk}
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = _extract_logits(model(inputs))
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct_mask = pred.eq(labels.view(1, -1).expand_as(pred))
            for k in topk:
                correct[k] += correct_mask[:k].reshape(-1).float().sum().item()
            total += labels.size(0)

    return {f"top{k}": round(100.0 * correct[k] / total, 2) for k in topk}


def per_class_accuracy(model, loader, device, num_classes=10):
    """Compute per-class accuracy."""
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = _extract_logits(model(inputs))
            _, predicted = outputs.max(1)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    return {
        CIFAR10_CLASSES[i]: round(100.0 * class_correct[i] / max(class_total[i], 1), 2)
        for i in range(num_classes)
    }


def count_params(model):
    """Count total and non-zero parameters."""
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero


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


def estimate_flops(model, input_shape=None, device='cpu'):
    """
    Estimate FLOPs using a forward hook approach.
    Counts multiply-accumulate operations for Conv2d and Linear layers.
    If input_shape is None, auto-detects from model's first Conv2d layer.
    """
    if input_shape is None:
        input_shape = detect_input_shape(model)

    flops = [0]

    def conv_hook(module, inp, out):
        batch_size = inp[0].size(0)
        out_channels, out_h, out_w = out.size(1), out.size(2), out.size(3)
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
            module.in_channels // module.groups)
        flops[0] += batch_size * out_channels * out_h * out_w * kernel_ops

    def linear_hook(module, inp, out):
        batch_size = inp[0].size(0)
        flops[0] += batch_size * module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return flops[0]


def evaluate_uploaded_model(model, test_loader, device, model_path=None):
    """
    Evaluate any uploaded model dynamically.
    Returns a dict with accuracy, size, params, FLOPs, latency.
    """
    model.eval()
    model = model.to(device)
    input_shape = detect_input_shape(model)

    acc = evaluate_accuracy(model, test_loader, device)
    total_params, nonzero_params = count_params(model)
    sparsity = round(100 * (1 - nonzero_params / total_params), 2) if total_params > 0 else 0
    flops = estimate_flops(model, input_shape=input_shape, device=device)
    latency = measure_latency(model, device, input_shape=input_shape)

    result = {
        "model_name": os.path.basename(model_path).replace('.pth', '').replace('.pt', '') if model_path else "uploaded_model",
        "accuracy": acc.get("top1", 0),
        "accuracy_top1": acc.get("top1", 0),
        "accuracy_top5": acc.get("top5", 0),
        "parameters": total_params,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "sparsity_percent": sparsity,
        "flops": flops,
        "flops_M": round(flops / 1e6, 2),
        "latency_ms": latency,
        "energy_kwh": 0,
        "co2_kg": 0,
    }

    if model_path and os.path.exists(model_path):
        result["original_size_MB"] = get_size_mb(model_path)
        result["size_MB"] = get_size_mb(model_path)

    return result


def measure_latency(model, device, input_shape=(1, 3, 32, 32), n_runs=200):
    """Measure average inference latency in milliseconds."""
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    return round((time.time() - start) / n_runs * 1000, 2)


def get_size_mb(path):
    """Get file size in MB."""
    return round(os.path.getsize(path) / 1e6, 2)


def load_model(arch, path, device):
    """Load a model from path, handling different architectures."""
    if arch == 'resnet18':
        model = resnet18(weights=None, num_classes=10)
        model.load_state_dict(torch.load(path, map_location=device))
    elif arch == 'compact_student':
        model = CompactStudent(num_classes=10)
        model.load_state_dict(torch.load(path, map_location=device))
    elif arch == 'compact_student_compressed':
        model = CompactStudent(num_classes=10)
        state = load_compressed(path, device=str(device))
        model.load_state_dict(state)
    elif arch == 'quantized_resnet18':
        from torchvision.models.quantization import resnet18 as qresnet18
        model = qresnet18(weights=None, num_classes=10, quantize=False)
        model.eval()
        model.fuse_model()
        backend = _configure_quantized_backend()
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        device = torch.device('cpu')  # quantized models run on CPU
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    model = model.to(device)
    model.eval()
    return model, device


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("../results", exist_ok=True)

    # Data loader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Models to evaluate
    models_config = [
        {
            "name": "Baseline (ResNet18)",
            "key": "baseline",
            "arch": "resnet18",
            "path": "../models/baseline_model.pth",
        },
        {
            "name": "Pruned 70% (Standard Save)",
            "key": "pruned_standard",
            "arch": "resnet18",
            "path": "../models/pruned_model.pth",
        },
        {
            "name": "Pruned 70% (Gzip Compressed)",
            "key": "pruned_compressed",
            "arch": "resnet18",
            "path": "../models/pruned_model_compressed.pth.gz",
            "compressed": True,
        },
        {
            "name": "Static Quantization INT8",
            "key": "quantized_static",
            "arch": "quantized_resnet18",
            "path": "../models/quantized_model.pth",
        },
        {
            "name": "KD → Compact Student",
            "key": "kd_compact_student",
            "arch": "compact_student",
            "path": "../models/student_distilled.pth",
        },
        {
            "name": "Compact Student + Quant",
            "key": "hybrid_student_quant",
            "arch": "compact_student",
            "path": "../models/hybrid_model.pth",
        },
        {
            "name": "Ultra-Compact (Compressed)",
            "key": "ultra_compact",
            "arch": "compact_student_compressed",
            "path": "../models/ultra_compact_compressed.pth.gz",
            "compressed": True,
        },
    ]

    evaluation_report = {}
    baseline_size = None

    print("\n" + "=" * 75)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 75)
    print(f"{'Model':<35} {'Top1':>6} {'Top5':>6} {'Size MB':>8} "
          f"{'↓Size%':>7} {'Params':>12} {'FLOPs':>12} {'Lat ms':>7}")
    print("-" * 95)

    for cfg in models_config:
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"  [SKIP] {cfg['name']}: {path} not found")
            continue

        try:
            # Load model
            if cfg.get("compressed"):
                if cfg["arch"] == "compact_student_compressed":
                    model = CompactStudent(num_classes=10)
                    state = load_compressed(path, device=str(device))
                    model.load_state_dict(state)
                    model = model.to(device)
                    eval_device = device
                else:
                    model = resnet18(weights=None, num_classes=10)
                    state = load_compressed(path, device=str(device))
                    model.load_state_dict(state)
                    model = model.to(device)
                    eval_device = device
            else:
                model, eval_device = load_model(
                    cfg["arch"], path, device)
            model.eval()

            # Size
            size_mb = get_size_mb(path)
            if baseline_size is None:
                baseline_size = size_mb
            size_reduction = round(
                100 * (baseline_size - size_mb) / baseline_size, 2
            ) if baseline_size > 0 else 0.0

            # Accuracy
            acc = evaluate_accuracy(model, test_loader, eval_device)

            # Parameters
            total_params, nonzero_params = count_params(model)
            sparsity = round(100 * (1 - nonzero_params / total_params), 2)

            # FLOPs
            flops = estimate_flops(model, device=eval_device)

            # Latency
            latency = measure_latency(model, eval_device)

            # Per-class accuracy
            pca = per_class_accuracy(model, test_loader, eval_device)

            result = {
                "model_name": cfg["name"],
                "accuracy": acc["top1"],
                "accuracy_top1": acc["top1"],
                "accuracy_top5": acc["top5"],
                "parameters": total_params,
                "original_size_MB": size_mb,
                "compressed_size_MB": size_mb,
                "size_MB": size_mb,
                "size_reduction_percent": size_reduction,
                "total_params": total_params,
                "nonzero_params": nonzero_params,
                "sparsity_percent": sparsity,
                "flops": flops,
                "flops_M": round(flops / 1e6, 2),
                "latency_ms": latency,
                "per_class_accuracy": pca,
                "energy_kwh": 0,
                "co2_kg": 0,
            }
            evaluation_report[cfg["key"]] = result

            print(f"{cfg['name']:<35} {acc['top1']:>6} {acc['top5']:>6} "
                  f"{size_mb:>8} {size_reduction:>6}% "
                  f"{total_params:>12,} {round(flops/1e6, 1):>10.1f}M "
                  f"{latency:>6}ms")

        except Exception as e:
            print(f"  [ERROR] {cfg['name']}: {e}")

    print("-" * 95)

    # Save report
    with open("../results/evaluation_report.json", "w") as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"\nEvaluation report saved to ../results/evaluation_report.json")
    print("✅ Evaluation complete!")
