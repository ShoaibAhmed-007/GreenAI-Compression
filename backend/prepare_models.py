"""
GreenAI — Pre-download & Evaluate All 15 Pretrained Models
============================================================
Run this ONCE to download all model weights and compute baseline metrics.
Results are saved to ../results/baseline_all_models.json

Usage:
    python prepare_models.py
    python prepare_models.py --dataset CIFAR10
    python prepare_models.py --models resnet18 vgg16 squeezenet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import time
import gc
import argparse
import signal
import threading

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compress import (
    PRELOADED_MODELS, get_pretrained_model, get_data_loaders,
    evaluate, count_params, measure_latency, detect_input_shape,
    _enable_head_gradients,
)
import torch.optim as optim


RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'results', 'baseline_all_models.json')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'pretrained_baselines')


def prepare_single_model(model_key, dataset='CIFAR10', device=None, save_weights=True,
                         train_loader=None, test_loader=None, timeout_sec=300):
    """
    Download, fine-tune, evaluate and save baseline metrics for a single model.
    Returns dict with all baseline metrics.
    timeout_sec: max seconds per model (default 300 = 5 min). Raises TimeoutError if exceeded.
    """
    model_start = time.time()

    def _check_timeout(step_name):
        elapsed = time.time() - model_start
        if elapsed > timeout_sec:
            raise TimeoutError(f"Model {model_key} exceeded {timeout_sec}s timeout at {step_name} ({elapsed:.0f}s elapsed)")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = PRELOADED_MODELS[model_key]
    input_size = cfg['input_size']
    num_classes = 10 if dataset.upper() == 'CIFAR10' else 100

    print(f"\n{'='*60}", flush=True)
    print(f"  Preparing: {cfg['name']} ({cfg['params']} params)", flush=True)
    print(f"  Input: {input_size}x{input_size} | Dataset: {dataset}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Download pretrained model
    print(f"  [1/5] Downloading pretrained weights...", flush=True)
    t0 = time.time()
    model = get_pretrained_model(model_key, num_classes=num_classes)
    model = model.to(device)
    download_time = time.time() - t0
    print(f"         Done in {download_time:.1f}s", flush=True)
    _check_timeout('download')

    # 2. Prepare data (reuse if provided)
    if train_loader is None or test_loader is None:
        print(f"  [2/5] Preparing {dataset} data ({input_size}x{input_size})...", flush=True)
        train_loader, test_loader = get_data_loaders(dataset, input_size=input_size, pin_memory=False)
    else:
        print(f"  [2/5] Using shared data loaders...", flush=True)

    # 3. Fine-tune: head only (fast)
    head_epochs = 2
    max_head_batches = 100
    print(f"  [3/5] Fine-tuning classifier head ({head_epochs} epochs, {max_head_batches} batches)...", flush=True)
    for p in model.parameters():
        p.requires_grad = False
    _enable_head_gradients(model, model_key)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    for ep in range(head_epochs):
        model.train()
        batch_count = 0
        for inputs, labels in train_loader:
            if batch_count >= max_head_batches:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(inputs), labels)
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % 25 == 0:
                print(f"         Epoch {ep+1}/{head_epochs} — batch {batch_count}/{max_head_batches} ({time.time()-model_start:.0f}s)", flush=True)
        _check_timeout(f'head-tune epoch {ep+1}')

    # 4. (Skipped) Full fine-tune is too slow for 224×224 pretrained models.
    #    Head-only tune gives good-enough baselines for compression comparison.
    print(f"  [4/5] Skipping full fine-tune (head-only is sufficient for baselines)...", flush=True)

    # 5. Evaluate
    print(f"  [5/5] Evaluating...", flush=True)
    accuracy = evaluate(model, test_loader, dev=device)
    total_params = count_params(model)
    input_shape = detect_input_shape(model)
    latency = measure_latency(model, input_shape=input_shape, dev=device)

    # Save weights
    if save_weights:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        weight_path = os.path.join(WEIGHTS_DIR, f'{model_key}_baseline.pth')
        torch.save(model.state_dict(), weight_path)
        size_mb = round(os.path.getsize(weight_path) / 1e6, 2)
        print(f"         Saved to {weight_path} ({size_mb} MB)", flush=True)
    else:
        size_mb = round(sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6, 2)

    result = {
        'model_key': model_key,
        'model_name': cfg['name'],
        'params_label': cfg['params'],
        'total_params': total_params,
        'input_size': input_size,
        'dataset': dataset,
        'accuracy': accuracy,
        'size_MB': size_mb,
        'latency_ms': latency,
        'status': 'ready',
    }

    total_time = time.time() - model_start
    print(f"\n  ✅ {cfg['name']}: Accuracy={accuracy}% | Size={size_mb}MB | Latency={latency}ms | Time={total_time:.0f}s", flush=True)
    # Cleanup GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result


def prepare_all_models(dataset='CIFAR10', model_keys=None, device=None):
    """
    Pre-download and evaluate all (or selected) pretrained models.
    Saves results to baseline_all_models.json.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keys = model_keys or list(PRELOADED_MODELS.keys())

    # Load existing results (to avoid re-doing already completed models)
    existing = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {RESULTS_FILE}")

    print(f"\nDevice: {device}", flush=True)
    print(f"Models to prepare: {len(keys)}", flush=True)
    print(f"Dataset: {dataset}", flush=True)
    print(f"{'='*60}", flush=True)

    # Lazily create data loaders per input_size (not all at once to save memory)
    shared_loaders = {}

    results = dict(existing)  # preserve existing
    failed = []

    for i, key in enumerate(keys):
        if key in results and results[key].get('status') == 'ready':
            print(f"\n[{i+1}/{len(keys)}] {key} — already prepared, skipping.", flush=True)
            continue

        try:
            # Clear GPU memory before each model
            torch.cuda.empty_cache()
            gc.collect()

            print(f"\n[{i+1}/{len(keys)}] Preparing {key}...", flush=True)
            isz = PRELOADED_MODELS[key]['input_size']

            # Lazily create loaders per input_size (pin_memory=False to avoid OOM)
            if isz not in shared_loaders:
                print(f"  Loading {dataset} data at {isz}x{isz}...", flush=True)
                shared_loaders[isz] = get_data_loaders(dataset, input_size=isz, pin_memory=False)

            tl, tel = shared_loaders[isz]
            result = prepare_single_model(key, dataset=dataset, device=device,
                                          train_loader=tl, test_loader=tel,
                                          timeout_sec=300)
            results[key] = result

            # Save incrementally (in case of crash)
            os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"\n  ❌ FAILED: {key} — {e}", flush=True)
            # Cleanup GPU after failure
            torch.cuda.empty_cache()
            gc.collect()
            results[key] = {
                'model_key': key,
                'model_name': PRELOADED_MODELS[key]['name'],
                'params_label': PRELOADED_MODELS[key]['params'],
                'input_size': PRELOADED_MODELS[key]['input_size'],
                'dataset': dataset,
                'status': 'error',
                'error': str(e),
            }
            failed.append(key)
            # Save even failures
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # Final summary
    ready = [k for k, v in results.items() if v.get('status') == 'ready']
    errors = [k for k, v in results.items() if v.get('status') == 'error']

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Ready:  {len(ready)}/{len(keys)} models")
    if errors:
        print(f"  Errors: {len(errors)} — {errors}")
    print(f"  Results saved to: {RESULTS_FILE}")
    print(f"  Weights saved to: {WEIGHTS_DIR}")

    return results


def _parse_params_label(label: str) -> float:
    """Convert params label like '138M' or '1.2M' to numeric millions."""
    label = label.strip().upper().replace(',', '')
    if label.endswith('M'):
        return float(label[:-1])
    if label.endswith('K'):
        return float(label[:-1]) / 1000
    return float(label)


def filter_models_by_params(keys: list, max_params_m: float) -> list:
    """Filter model keys to only those with ≤ max_params_m million parameters."""
    kept, skipped = [], []
    for k in keys:
        params_m = _parse_params_label(PRELOADED_MODELS[k]['params'])
        if params_m <= max_params_m:
            kept.append(k)
        else:
            skipped.append((k, PRELOADED_MODELS[k]['params']))
    if skipped:
        print(f"\nSkipping {len(skipped)} models exceeding {max_params_m}M params:")
        for k, p in skipped:
            print(f"  ✗ {k} ({p})")
    return kept


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-download and evaluate all pretrained models')
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific model keys to prepare (default: all)')
    parser.add_argument('--max-params', type=float, default=None,
                        help='Max params in millions (e.g. 25). Models above this are skipped.')
    args = parser.parse_args()

    model_keys = args.models or list(PRELOADED_MODELS.keys())
    if args.max_params is not None:
        model_keys = filter_models_by_params(model_keys, args.max_params)

    prepare_all_models(dataset=args.dataset, model_keys=model_keys)
