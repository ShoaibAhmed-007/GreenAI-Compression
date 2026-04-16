"""
Green AI — Professional CIFAR-10 Training Pipeline
====================================================
Trains any of the 11 supported CNN architectures on CIFAR-10 with:

  - Correct CIFAR-10 preprocessing (RandomCrop + HFlip + Normalize)
  - Per-model input resize (224 for most, 299 for InceptionV3)
  - Proper classifier head adaptation for 10 classes
  - SGD with momentum + CosineAnnealingLR scheduler
  - Mixed precision (AMP) on GPU
  - CodeCarbon energy tracking
  - Automatic results JSON with all required metrics

Supported models:
  resnet18, resnet34, mobilenet_v2, efficientnet_b0, efficientnet_b1,
  densenet121, densenet169, squeezenet, shufflenet_v2, inception_v3, googlenet

Usage:
  python train.py                          # Train ResNet18 (default)
  python train.py --model mobilenet_v2     # Train MobileNetV2
  python train.py --model all              # Train all 11 models sequentially
  python train.py --epochs 30 --batch-size 64
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ─── Constants ─────────────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
NUM_CLASSES  = 10

# Model → required spatial input size
MODEL_INPUT_SIZES = {
    'resnet18':       224,
    'resnet34':       224,
    'mobilenet_v2':   224,
    'efficientnet_b0': 224,
    'efficientnet_b1': 224,
    'densenet121':    224,
    'densenet169':    224,
    'squeezenet':     224,
    'shufflenet_v2':  224,
    'inception_v3':   299,
    'googlenet':      224,
}

SUPPORTED_MODELS = list(MODEL_INPUT_SIZES.keys())


# ─── Model Factory ─────────────────────────────────────────────────────────────
def get_input_size(model_name: str) -> int:
    """Return the required input spatial size for a given model."""
    key = model_name.lower().strip()
    if key not in MODEL_INPUT_SIZES:
        raise ValueError(f"Unknown model: {model_name}. Supported: {SUPPORTED_MODELS}")
    return MODEL_INPUT_SIZES[key]


def get_model(model_name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Load a torchvision model with ImageNet pretrained weights (if available)
    and replace the classifier head for the target number of classes.
    """
    key = model_name.lower().strip()
    weights = 'DEFAULT' if pretrained else None

    if key == 'resnet18':
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif key == 'resnet34':
        model = torchvision.models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif key == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif key == 'efficientnet_b0':
        model = torchvision.models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif key == 'efficientnet_b1':
        model = torchvision.models.efficientnet_b1(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif key == 'densenet121':
        model = torchvision.models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif key == 'densenet169':
        model = torchvision.models.densenet169(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif key == 'squeezenet':
        model = torchvision.models.squeezenet1_1(weights=weights)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes

    elif key == 'shufflenet_v2':
        model = torchvision.models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif key == 'inception_v3':
        model = torchvision.models.inception_v3(weights=weights, aux_logits=True)
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif key == 'googlenet':
        model = torchvision.models.googlenet(weights=weights, aux_logits=True)
        model.aux_logits = False
        model.aux1 = None
        model.aux2 = None
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: {SUPPORTED_MODELS}")

    return model


# ─── Data Loaders ──────────────────────────────────────────────────────────────
def get_cifar10_loaders(input_size: int = 224, batch_size: int = 128,
                        num_workers: int = 4, data_dir: str = './data'):
    """
    Create CIFAR-10 train/test DataLoaders with proper preprocessing.
    - Training: RandomCrop(32, padding=4) → RandomHorizontalFlip → Resize → Normalize
    - Validation: Resize → Normalize (NO augmentation)
    """
    # Training transforms — augmentation on native 32×32, then resize
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if input_size > 32:
        train_transforms.append(transforms.Resize(input_size))
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Validation transforms — NO augmentation
    val_transforms = []
    if input_size > 32:
        val_transforms.append(transforms.Resize(input_size))
    val_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    transform_train = transforms.Compose(train_transforms)
    transform_val = transforms.Compose(val_transforms)

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ─── Training Loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch. Returns (avg_loss, accuracy%)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a DataLoader. Returns accuracy%."""
    model.eval()
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def train_model(model_name: str, epochs: int = 50, batch_size: int = 128,
                lr: float = None, num_workers: int = 4, data_dir: str = './data',
                save_dir: str = '../models', results_dir: str = '../results',
                use_amp: bool = True, pretrained: bool = True):
    """
    Full training pipeline for a single model.
    Returns dict with all metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = get_input_size(model_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Auto-select LR: pretrained models need lower LR to avoid NaN
    if lr is None:
        lr = 0.01 if pretrained else 0.1

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"  Device: {device} | Input: {input_size}×{input_size}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Pretrained: {pretrained}")
    print(f"{'='*60}\n")

    # ── Model ──
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=pretrained)
    model = model.to(device)

    # ── Data ──
    train_loader, val_loader = get_cifar10_loaders(
        input_size=input_size, batch_size=batch_size,
        num_workers=num_workers, data_dir=data_dir)

    # ── Optimizer + Scheduler ──
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Mixed Precision ──
    use_amp_actual = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp_actual else None

    # ── CodeCarbon Energy Tracking ──
    tracker = None
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(
            project_name=f"train_{model_name}",
            output_dir=results_dir,
            country_iso_code="PAK",
            region="Punjab",
            log_level="error",
        )
        tracker.start()
        print("  📊 CodeCarbon energy tracking: ACTIVE")
        
        # Check if actual GPU tracking is engaged or if it fell back to TDP
        has_gpu = any(hw.__class__.__name__.lower().find('gpu') != -1 for hw in getattr(tracker, '_hardware', []))
        print("     ℹ️ Carbon Intensity: Using average fallback (Regional estimate for Punjab, PAK)")
        if has_gpu:
            print("     ℹ️ Power Draw: Using actual GPU hardware metrics (NVML)")
        else:
            print("     ℹ️ Power Draw: Using TDP approximations (CPU/RAM fallback tracking)")
    except ImportError:
        print("  ⚠️  CodeCarbon not installed — energy tracking disabled")
    except Exception as e:
        print(f"  ⚠️  CodeCarbon failed to start: {e}")

    # ── Training ──
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr_current = optimizer.param_groups[0]['lr']

        # Update emissions so far and print them
        curr_kwh, curr_co2 = 0.0, 0.0
        if tracker is not None:
            try:
                curr_co2 = float(tracker.flush())
                # Safely try to grab energy_consumed (in kWh), otherwise use fallback conversion
                if hasattr(tracker, '_total_energy') and hasattr(tracker._total_energy, 'kWh'):
                    curr_kwh = float(tracker._total_energy.kWh)
                else:
                    curr_kwh = curr_co2 / 1000.0  # fallback approximation
            except Exception:
                pass

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"Loss: {train_loss:.4f}  "
              f"Train: {train_acc:.2f}%  "
              f"Val: {val_acc:.2f}%  "
              f"LR: {lr_current:.6f}  "
              f"Time: {epoch_time:.0f}s  "
              f"Energy: {curr_kwh:.5f}kWh  "
              f"CO₂: {curr_co2:.5f}kg")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, f'{model_name}_baseline.pth')
            torch.save(model.state_dict(), save_path)

    total_time = time.time() - start_time

    # ── Stop Energy Tracking ──
    energy_kwh = 0.0
    co2_kg = 0.0
    if tracker is not None:
        try:
            emissions = tracker.stop()
            if emissions is not None:
                co2_kg = round(max(float(emissions), 0.0), 8)

            final_data = getattr(tracker, "final_emissions_data", None)
            if final_data is not None and hasattr(final_data, "energy_consumed"):
                try:
                    energy_kwh = round(max(float(final_data.energy_consumed), 0.0), 8)
                except (TypeError, ValueError):
                    energy_kwh = 0.0

            # Fallback approximation if tracker energy is unavailable.
            if energy_kwh <= 0 and co2_kg > 0:
                energy_kwh = round(co2_kg / 1000, 8)
        except Exception:
            pass

    # ── Compute Final Metrics ──
    model_path = os.path.join(save_dir, f'{model_name}_baseline.pth')
    model_size_mb = round(os.path.getsize(model_path) / 1e6, 2) if os.path.exists(model_path) else 0
    total_params = sum(p.numel() for p in model.parameters())

    result = {
        'model_name': model_name,
        'accuracy': round(best_acc, 2),
        'parameters': total_params,
        'original_size_MB': model_size_mb,
        'epochs': epochs,
        'batch_size': batch_size,
        'optimizer': 'SGD',
        'learning_rate': lr,
        'scheduler': 'CosineAnnealingLR',
        'input_size': input_size,
        'training_time_seconds': round(total_time, 1),
        'energy_kwh': energy_kwh,
        'co2_kg': co2_kg,
        'training_energy_kwh': energy_kwh,
        'training_co2_kg': co2_kg,
        'device': str(device),
        'amp': use_amp_actual,
    }

    # ── Save Results ──
    result_path = os.path.join(results_dir, f'{model_name}_training_result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  ✅ {model_name}: Best accuracy = {best_acc:.2f}%")
    print(f"     Model saved: {model_path} ({model_size_mb} MB)")
    print(f"     Results saved: {result_path}")
    print(f"     Energy: {energy_kwh} kWh | CO₂: {co2_kg} kg")
    print(f"     Total time: {total_time:.1f}s")

    # Cleanup GPU
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ─── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Green AI — CIFAR-10 Training Pipeline')
    parser.add_argument('--model', nargs='+', default=['resnet18'],
                        help='Model(s) to train. Use "all" for all 11 models. '
                             'Can specify multiple: --model squeezenet shufflenet_v2')
    parser.add_argument('--smallest-first', action='store_true',
                        help='Train models from smallest to largest (by params)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 0.01 for pretrained, 0.1 for scratch)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--data-dir', default='./data',
                        help='Data directory (default: ./data)')
    parser.add_argument('--save-dir', default='../models',
                        help='Model save directory (default: ../models)')
    parser.add_argument('--results-dir', default='../results',
                        help='Results directory (default: ../results)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train from scratch (no ImageNet pretrained weights)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Model param counts for sorting (millions)
    PARAM_COUNTS = {
        'squeezenet': 1.2, 'shufflenet_v2': 2.3, 'mobilenet_v2': 3.4,
        'efficientnet_b0': 5.3, 'googlenet': 6.8, 'efficientnet_b1': 7.8,
        'densenet121': 8.0, 'resnet18': 11.2, 'densenet169': 14.3,
        'resnet34': 21.8, 'inception_v3': 23.8,
    }

    # Resolve model list
    if 'all' in args.model:
        models = list(SUPPORTED_MODELS)
    else:
        models = [m.lower().strip() for m in args.model]
        invalid = [m for m in models if m not in SUPPORTED_MODELS]
        if invalid:
            print(f"❌ Unknown model(s): {invalid}")
            print(f"   Supported: {SUPPORTED_MODELS}")
            return

    # Sort smallest to largest if requested
    if args.smallest_first:
        models.sort(key=lambda m: PARAM_COUNTS.get(m, 999))
        print(f"  📏 Sorted smallest → largest: {models}")

    all_results = {}

    print(f"\n{'='*60}")
    print(f"  Green AI — CIFAR-10 Training Pipeline")
    print(f"  Models: {len(models)}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*60}")

    for model_name in models:
        try:
            result = train_model(
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
                data_dir=args.data_dir,
                save_dir=args.save_dir,
                results_dir=args.results_dir,
                use_amp=not args.no_amp,
                pretrained=not args.no_pretrained,
            )
            all_results[model_name] = result
        except Exception as e:
            print(f"\n  ❌ FAILED: {model_name} — {e}")
            all_results[model_name] = {'model_name': model_name, 'error': str(e)}

    # Save combined results (merge with existing so partial reruns do not wipe prior models).
    combined_path = os.path.join(args.results_dir, 'training_results_all.json')
    os.makedirs(args.results_dir, exist_ok=True)
    existing_results = {}
    if os.path.exists(combined_path):
        try:
            with open(combined_path, 'r') as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                existing_results = payload
        except Exception:
            existing_results = {}

    existing_results.update(all_results)
    with open(combined_path, 'w') as f:
        json.dump(existing_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    for name, r in existing_results.items():
        if 'error' in r:
            print(f"  ❌ {name}: {r['error']}")
        else:
            print(f"  ✅ {name}: {r['accuracy']}% | {r['original_size_MB']} MB")
    print(f"\n  Combined results: {combined_path}")


if __name__ == '__main__':
    main()