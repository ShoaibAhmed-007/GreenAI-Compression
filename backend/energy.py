# energy.py
"""
Green AI FYP — Phase 7: Energy & Carbon Emissions Tracking
===========================================================
Uses CodeCarbon to measure energy consumption and CO2 emissions
during model training and inference for all compression strategies.

Tracks:
  - Training energy (kWh) and CO2 (kg) for baseline + compression
  - Inference energy per 10,000 samples for each strategy
  - Energy savings percentage vs baseline

Outputs: ../results/energy_report.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import sys
import json
import time

# CodeCarbon for energy tracking
from codecarbon import OfflineEmissionsTracker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compress import (
    CompactStudent,
    evaluate,
    distillation_loss,
    load_compressed,
    _extract_logits,
    _configure_quantized_backend,
)


def track_dynamic_energy(model, test_loader, device, model_name="uploaded_model",
                         n_batches=40):
    """
    Track energy consumed during inference for any uploaded model.
    Returns a dict with energy_kWh and co2_kg.
    """
    model.eval()
    model = model.to(device)

    tracker = OfflineEmissionsTracker(
        project_name=f"dynamic_{model_name}",
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'results'),
        country_iso_code="PAK",
        region="Punjab",
        log_level="error",
    )

    tracker.start()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            _ = _extract_logits(model(inputs))
            if i >= n_batches - 1:
                break
    emissions = tracker.stop()

    return {
        "model": model_name,
        "phase": "inference",
        "samples": n_batches * test_loader.batch_size,
        "energy_kWh": round(float(emissions) / 1000, 8) if emissions else 0,
        "co2_kg": round(float(emissions), 8) if emissions else 0,
    }


def track_inference_energy(model, loader, device, model_name, n_batches=80):
    """
    Track energy consumed during inference on n_batches (~10,000 samples).
    Returns emissions data dict.
    """
    model.eval()
    model = model.to(device)

    tracker = OfflineEmissionsTracker(
        project_name=f"inference_{model_name}",
        output_dir="../results",
        output_file=f"emissions_inference_{model_name}.csv",
        country_iso_code="PAK",
        region="Punjab",
        log_level="error",
    )

    tracker.start()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            _ = _extract_logits(model(inputs))
            if i >= n_batches - 1:
                break
    emissions = tracker.stop()

    return {
        "model": model_name,
        "phase": "inference",
        "samples": n_batches * loader.batch_size,
        "energy_kWh": round(float(emissions) / 1000, 8) if emissions else 0,
        "co2_kg": round(float(emissions), 8) if emissions else 0,
    }


def track_training_energy(model, train_loader, device, model_name,
                          epochs=3, lr=0.001, teacher=None):
    """
    Track energy consumed during a short training/fine-tuning run.
    If teacher is provided, uses knowledge distillation loss.
    """
    model.train()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    tracker = OfflineEmissionsTracker(
        project_name=f"training_{model_name}",
        output_dir="../results",
        output_file=f"emissions_training_{model_name}.csv",
        country_iso_code="PAK",
        region="Punjab",
        log_level="error",
    )

    tracker.start()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = _extract_logits(model(inputs))
            if teacher is not None:
                teacher.eval()
                with torch.no_grad():
                    t_out = _extract_logits(teacher(inputs))
                loss = distillation_loss(outputs, t_out, labels, T=4.0, alpha=0.3)
            else:
                loss = F.cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"    [{model_name}] Epoch {epoch+1}/{epochs}, "
              f"Loss: {running_loss/len(train_loader):.4f}")

    emissions = tracker.stop()

    return {
        "model": model_name,
        "phase": "training",
        "epochs": epochs,
        "energy_kWh": round(float(emissions) / 1000, 8) if emissions else 0,
        "co2_kg": round(float(emissions), 8) if emissions else 0,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("../results", exist_ok=True)

    # ----------------------------------------------------------
    # Data
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

    energy_report = {
        "inference": {},
        "training": {},
    }

    # ----------------------------------------------------------
    # Load baseline teacher
    # ----------------------------------------------------------
    baseline = resnet18(weights=None, num_classes=10)
    baseline.load_state_dict(
        torch.load("../models/baseline_model.pth", map_location=device))
    baseline = baseline.to(device)
    baseline.eval()

    # ==========================================================
    # 1. INFERENCE ENERGY — all models
    # ==========================================================
    print("\n" + "=" * 65)
    print("PHASE 7a: INFERENCE ENERGY TRACKING")
    print("=" * 65)

    inference_models = [
        ("baseline_resnet18", lambda: baseline),
    ]

    # Add pruned model if available
    if os.path.exists("../models/pruned_model.pth"):
        def load_pruned():
            m = resnet18(weights=None, num_classes=10)
            m.load_state_dict(torch.load("../models/pruned_model.pth",
                                         map_location=device))
            return m.to(device)
        inference_models.append(("pruned_70pct", load_pruned))

    # Add quantized model (CPU only)
    if os.path.exists("../models/quantized_model.pth"):
        def load_quantized():
            from torchvision.models.quantization import resnet18 as qresnet18
            m = qresnet18(weights=None, num_classes=10, quantize=False)
            m.eval()
            m.fuse_model()
            backend = _configure_quantized_backend()
            m.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.quantization.prepare(m, inplace=True)
            torch.quantization.convert(m, inplace=True)
            m.load_state_dict(torch.load("../models/quantized_model.pth",
                                         map_location='cpu'))
            return m  # stays on CPU
        inference_models.append(("quantized_int8", load_quantized))

    # Add compact student
    if os.path.exists("../models/student_distilled.pth"):
        def load_student():
            m = CompactStudent(num_classes=10)
            m.load_state_dict(torch.load("../models/student_distilled.pth",
                                         map_location=device))
            return m.to(device)
        inference_models.append(("compact_student", load_student))

    # Add hybrid
    if os.path.exists("../models/hybrid_model.pth"):
        def load_hybrid():
            m = CompactStudent(num_classes=10)
            m.load_state_dict(torch.load("../models/hybrid_model.pth",
                                         map_location='cpu'))
            return m  # quantized, stays on CPU
        inference_models.append(("hybrid_student_quant", load_hybrid))

    for model_name, loader_fn in inference_models:
        print(f"\n  Measuring inference energy: {model_name}...")
        try:
            model = loader_fn()
            model.eval()
            dev = next(model.parameters()).device
            result = track_inference_energy(
                model, test_loader, dev, model_name)
            energy_report["inference"][model_name] = result
            print(f"    Energy: {result['energy_kWh']:.8f} kWh, "
                  f"CO2: {result['co2_kg']:.8f} kg")
        except Exception as e:
            print(f"    [ERROR] {model_name}: {e}")

    # ==========================================================
    # 2. TRAINING ENERGY — baseline vs compact student
    # ==========================================================
    print("\n" + "=" * 65)
    print("PHASE 7b: TRAINING ENERGY TRACKING (3 epochs each)")
    print("=" * 65)

    # Baseline training energy
    print("\n  Tracking baseline ResNet18 training energy...")
    baseline_train = resnet18(weights=None, num_classes=10).to(device)
    baseline_train_result = track_training_energy(
        baseline_train, train_loader, device,
        "baseline_resnet18", epochs=3, lr=0.01)
    energy_report["training"]["baseline_resnet18"] = baseline_train_result
    print(f"    Energy: {baseline_train_result['energy_kWh']:.8f} kWh, "
          f"CO2: {baseline_train_result['co2_kg']:.8f} kg")

    # Compact student training energy (with KD)
    print("\n  Tracking compact student KD training energy...")
    student_train = CompactStudent(num_classes=10).to(device)
    student_train_result = track_training_energy(
        student_train, train_loader, device,
        "compact_student_kd", epochs=3, lr=0.001, teacher=baseline)
    energy_report["training"]["compact_student_kd"] = student_train_result
    print(f"    Energy: {student_train_result['energy_kWh']:.8f} kWh, "
          f"CO2: {student_train_result['co2_kg']:.8f} kg")

    # ==========================================================
    # 3. ENERGY SAVINGS SUMMARY
    # ==========================================================
    print("\n" + "=" * 65)
    print("ENERGY SAVINGS SUMMARY")
    print("=" * 65)

    baseline_inf = energy_report["inference"].get("baseline_resnet18", {})
    baseline_inf_energy = baseline_inf.get("energy_kWh", 0)
    baseline_train_energy = baseline_train_result.get("energy_kWh", 0)

    savings = {}
    for name, data in energy_report["inference"].items():
        if name == "baseline_resnet18":
            continue
        inf_energy = data.get("energy_kWh", 0)
        if baseline_inf_energy > 0 and inf_energy > 0:
            pct = round(100 * (1 - inf_energy / baseline_inf_energy), 2)
        else:
            pct = 0.0
        savings[name] = {
            "inference_energy_kWh": inf_energy,
            "baseline_energy_kWh": baseline_inf_energy,
            "energy_saving_percent": pct,
        }
        print(f"  {name:<30} "
              f"Energy: {inf_energy:.8f} kWh  "
              f"Saving: {pct}%")

    if baseline_train_energy > 0:
        student_energy = student_train_result.get("energy_kWh", 0)
        if student_energy > 0:
            train_saving = round(
                100 * (1 - student_energy / baseline_train_energy), 2)
        else:
            train_saving = 0.0
        savings["training_compact_vs_baseline"] = {
            "baseline_energy_kWh": baseline_train_energy,
            "student_energy_kWh": student_energy,
            "energy_saving_percent": train_saving,
        }
        print(f"\n  Training: Compact Student uses "
              f"{train_saving}% less energy than baseline")

    energy_report["savings"] = savings

    # Save report
    with open("../results/energy_report.json", "w") as f:
        json.dump(energy_report, f, indent=2)

    print(f"\nEnergy report saved to ../results/energy_report.json")
    print("✅ Phase 7: Energy tracking complete!")
