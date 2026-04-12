"""
Simple reference snippets for Green AI CIFAR-10 experiments.
These are intentionally minimal and suitable for final-year project reports.
"""

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights, resnet18


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_cifar10_transforms(input_size: int = 224):
    """Train/test transforms with proper CIFAR-10 normalization."""
    train_tfms = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ]
    if input_size > 32:
        train_tfms.append(T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC))
    train_tfms.extend([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_tfms = []
    if input_size > 32:
        test_tfms.append(T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC))
    test_tfms.extend([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return T.Compose(train_tfms), T.Compose(test_tfms)


def build_resnet18_for_cifar(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """ResNet18 setup for CIFAR-10 using ImageNet pretrained weights."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model._input_size = 224  # convenience flag for downstream inference code
    return model


@dataclass
class EvalResult:
    top1_acc: float
    avg_latency_ms: float
    model_size_mb: float


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    """Simple full-dataset top-1 evaluation loop."""
    model.eval()
    model.to(device)

    total = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def benchmark_latency_ms(model: nn.Module, device: torch.device,
                         input_shape=(1, 3, 224, 224), warmup=20, runs=200):
    """Average single-batch inference latency in milliseconds."""
    model.eval()
    model.to(device)
    x = torch.randn(*input_shape, device=device)

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(runs):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.time() - t0) * 1000.0 / runs


def get_model_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


def compute_size_reduction_percent(baseline_mb: float, compressed_mb: float) -> float:
    if baseline_mb <= 0:
        return 0.0
    return 100.0 * (baseline_mb - compressed_mb) / baseline_mb


def compute_efficiency_report(
    baseline_acc: float,
    compressed_acc: float,
    baseline_latency_ms: float,
    compressed_latency_ms: float,
    baseline_size_mb: float,
    compressed_size_mb: float,
    baseline_co2_kg: float,
    compressed_co2_kg: float,
):
    """Compact metrics dictionary for dashboard/UI."""
    return {
        "baseline_accuracy": round(baseline_acc, 2),
        "compressed_accuracy": round(compressed_acc, 2),
        "accuracy_delta": round(compressed_acc - baseline_acc, 2),
        "baseline_latency_ms": round(baseline_latency_ms, 2),
        "compressed_latency_ms": round(compressed_latency_ms, 2),
        "latency_reduction_percent": round(
            100.0 * (baseline_latency_ms - compressed_latency_ms) / max(baseline_latency_ms, 1e-9), 2
        ),
        "baseline_size_mb": round(baseline_size_mb, 2),
        "compressed_size_mb": round(compressed_size_mb, 2),
        "size_reduction_percent": round(
            compute_size_reduction_percent(baseline_size_mb, compressed_size_mb), 2
        ),
        "baseline_co2_kg": baseline_co2_kg,
        "compressed_co2_kg": compressed_co2_kg,
        "co2_reduction_percent": round(
            100.0 * (baseline_co2_kg - compressed_co2_kg) / max(baseline_co2_kg, 1e-12), 2
        ),
    }
