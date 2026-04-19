# GreenAI Compression Implementation Summary (Code-Level)

This document explains how compression is currently implemented in code with a three-layer interaction model:
- Layer 1: Auto-Green (Smart Router)
- Layer 2: Intent Presets
- Layer 3: Manual (Advanced)

Primary implementation file:
- backend/compress.py

API integration file:
- backend/main.py


## 1) End-to-End Compression Entry Points

There are two production entry points.

1. Preloaded model compression
- Endpoint: POST /api/compress/preloaded
- Backend call path:
  - backend/main.py -> run_compression(...) in backend/compress.py
- Uses curated torchvision models (resnet18, densenet, efficientnet, etc.), optionally loading pre-saved baseline checkpoints first.

2. Dynamic uploaded-model compression
- Endpoint: POST /api/compress/dynamic
- Backend call path:
  - backend/main.py -> compress_dynamic(...) in backend/compress.py
- Accepts uploaded .pt/.pth and auto-detects architecture when possible.

Supported strategy keys for both entry points:
- smart
- maximize_speed
- minimize_size
- preserve_accuracy
- pruning
- quantization
- hybrid
- kd


## 2) Shared Pipeline Building Blocks

These are used by all strategies.

### 2.1 Data and preprocessing
- Function: get_data_loaders(...)
- Datasets: CIFAR10
- Uses dataset-specific normalization.
- Applies augmentation on train split:
  - color jitter (optional)
  - random horizontal flip
  - random crop
- Resizes if input_size > 32 (for ImageNet-style models).
- Includes hardware-aware batch sizing when model parameter counts are available:
  - <5M params -> batch 1024
  - <15M params -> batch 512
  - >=15M params -> batch 256
  - large input sizes clamp batch for VRAM safety
- On Windows, DataLoader workers are set to 0 to avoid spawn deadlocks.

### 2.2 Model loading (dynamic uploads)
- Function: load_uploaded_model(...)
- Load priority:
  1. Explicit architecture if provided
  2. TorchScript model
  3. Full torch-saved nn.Module
  4. state_dict auto-detection by key patterns
- Supports state_dict unwrapping keys like model_state_dict/state_dict and strips DataParallel prefix module.

### 2.3 Core metric helpers
- evaluate(...): computes classification accuracy.
- measure_latency(...): warmup + timed average inference latency.
- count_params(...), count_nonzero(...): used in pruning/hybrid reporting.

### 2.4 Artifact saving and size optimization
- build_compressed_model_path(...): normalized filename builder.
- save_sparse_state_dict(...): converts zero-heavy tensors (>50% zeros) to sparse tensors before save.
- save_compressed(...): auto-switch sparse saving when model sparsity is high.
- save_smallest_artifact(...): writes multiple variants and keeps only smallest:
  - dense .pth
  - optional sparse .pth
  - optional fp16 .pth
  - gzipped variants of each

### 2.5 CO2 / energy tracking and fair comparison
- _start_emissions_tracker(...), _stop_emissions_tracker(...): CodeCarbon online with offline fallback.
- _track_inference_emissions(...): steady-state temporal inference emissions.
- _track_training_emissions(...): standardized training benchmark emissions with pre-tracking warmup.
- _build_fair_comparison_metrics(...): compares baseline vs compressed under controlled benchmark workloads.

Important fair-comparison behavior:
- Measures baseline and compressed train + inference emissions under comparable settings.
- Uses pre-tracking warmup (2 seconds) for both inference and training benchmarks.
- Enables cuDNN autotuning during benchmark tracking.
- Inference benchmarking uses a strict time window (5 seconds), not a fixed batch quota.
- Inference loader iteration is cycled continuously so the benchmark keeps running until time expires.
- Inference benchmark metadata is exported (images processed, iterations, warmup/window durations).
- CO2 reporting method label:
  - Temporal Steady-State Measurement (5s window)
- Uses latency-scaled compressed batch counts:
  - scale = clamp(compressed_latency / baseline_latency, 0.5, 1.0)
- Rebuilds benchmark train/test loaders with model-aware batch sizing to better saturate high-end GPUs.
- If compressed CO2 measurement is zero, fallback projection is:
  - compressed_co2 = baseline_co2 * (0.6 * latency_ratio + 0.4 * size_ratio)
- Adds sanity warnings for suspicious values.


## 3) Manual Technique 1: Pruning (apply_pruning)

Function:
- apply_pruning(model, train_loader, test_loader, device, amount=0.15, fine_tune_epochs=8, ...)

What the code does:
1. Deep-copies model and computes baseline accuracy/size.
2. Starts CodeCarbon tracking for compression phase.
3. Performs gradual structured pruning:
   - prune_steps = 2
   - step_amount = amount / prune_steps
   - Conv2d: L2 structured pruning over output channels
   - Linear: L1 unstructured fallback
   - applied through _apply_structured_pruning_step(...)
4. Recovery training after each pruning step:
   - SGD (lr=1e-3, momentum=0.9, weight_decay=5e-4)
   - capped to 100 batches per step for speed
5. Final fine-tune stage:
   - SGD (lr=5e-4, momentum=0.9, weight_decay=5e-4)
   - CosineAnnealingLR
   - keeps best checkpoint by validation accuracy
6. Removes pruning reparameterization masks with _remove_pruning_from_model(...).
7. Saves artifact using save_smallest_artifact(..., prefer_sparse=True).
8. Runs final full evaluation, latency, inference emissions, and fair baseline/compressed benchmark metrics.

Manual dispatch defaults in entry points:
- Preloaded path: amount=0.70, fine_tune_epochs=max(10, requested_epochs)
- Dynamic path: amount=0.70, fine_tune_epochs=max(10, requested_epochs)


## 4) Manual Technique 2: Quantization (apply_quantization)

Function:
- apply_quantization(model, train_loader, test_loader, device, fine_tune_epochs=10, ...)

High-level behavior:
- Multi-stage quantization with runtime-aware fallbacks.
- Designed to keep a working path even for fragile architectures.

Detailed flow:
1. Baseline eval and baseline artifact size capture.
2. Pre-finetune before quantization:
   - 5 epochs
   - SGD + StepLR
   - capped to 100 batches/epoch
3. Quantization backend selection:
   - chooses available backend from x86/fbgemm/qnnpack/onednn (depending on build)
4. Architecture safety routing:
   - lightweight/fragile families can trigger FP16 safeguard preference.

Primary quantization attempts:
- Attempt 1: QAT path
- Attempt 2: Post-Training Static Quantization
- Attempt 3: Dynamic quantization fallback

CUDA/TensorRT runtime logic:
- On CUDA, tries TensorRT export/compile via export_to_tensorrt(...)
- Fallback path continues with float runtime when TensorRT is unavailable.

Manual dispatch defaults in entry points:
- Preloaded path: fine_tune_epochs=max(1, requested_epochs // 2)
- Dynamic path: fine_tune_epochs=max(1, requested_epochs // 2)


## 5) Manual Technique 3: Hybrid (apply_hybrid)

Function:
- apply_hybrid(model, train_loader, test_loader, device, amount=0.25, fine_tune_epochs=5, ...)

Pipeline:
1. Baseline eval and size capture.
2. Accuracy guard tracking initialized (default threshold 2.5% drop).
3. Structured pruning stage with retry behavior if guard fails.
4. Fine-tune while masks remain active (preserves zeros).
5. Guarded recovery phase before quantization if needed.
6. Remove pruning masks.
7. Quantization stage with fallback chain (QAT -> static -> dynamic).
8. Save hybrid artifact with save_smallest_artifact(..., prefer_sparse=True).
9. Evaluate quantized hybrid model, latency, emissions, and fair metrics.

Manual dispatch defaults in entry points:
- Preloaded path: amount=0.25, fine_tune_epochs=requested_epochs
- Dynamic path: amount=0.25, fine_tune_epochs=requested_epochs


## 6) Manual Technique 4: Knowledge Distillation (apply_kd)

Function:
- apply_kd(teacher, train_loader, test_loader, device, num_classes=10, epochs=20, ...)

Student architecture:
- CompactStudent class (MobileNet-style depthwise separable CNN).

KD flow summary:
1. Evaluate teacher baseline accuracy/size/params.
2. Build CompactStudent and compute student params.
3. Train student with distillation loss while teacher is no_grad.
4. Apply accuracy guard behavior and adaptive learning-rate responses.
5. Save/reload best student checkpoint.
6. Save artifact, evaluate, track emissions, compute fair metrics.

Manual dispatch defaults in entry points:
- Preloaded path: epochs=requested_epochs
- Dynamic path: epochs=requested_epochs * 4


## 7) Layer 1: Auto-Green Smart Router (run_smart_compression)

Function:
- run_smart_compression(model_key, model, train_loader, test_loader, device, ...)

Current routing logic:
- Heavy-model allowlist:
  - resnet18, resnet34, densenet121, densenet169, inception_v3, googlenet
  - route: apply_hybrid(..., amount=0.35, fine_tune_epochs=max(1, fine_tune_epochs))
- Else (compact/lightweight default):
  - route: apply_kd(..., epochs=15)

Returned metadata for smart:
- strategy = smart
- user_intent_layer = Smart
- resolved_technique = apply_hybrid or apply_kd
- resolved_strategy = preserved pre-annotation strategy
- smart_router_enabled = true
- smart_router_group = heavy | lightweight
- smart_router_strategy = hybrid | kd
- runtime_precision default set to int8 for hybrid route, fp16 for kd route (if not already set)


## 8) Layer 2: Intent-Based Presets (Wrapper Layer)

These wrappers map user intent to existing technical pipelines.

1. maximize_speed_preset(...)
- Intent: Real-time inference
- Strategy: apply_quantization(..., fine_tune_epochs=5)
- Metadata:
  - strategy = maximize_speed
  - user_intent_layer = Preset
  - resolved_technique = apply_quantization

2. minimize_size_preset(...)
- Intent: Edge/mobile storage
- Strategy: apply_hybrid(..., amount=0.50, fine_tune_epochs=10)
- Metadata:
  - strategy = minimize_size
  - user_intent_layer = Preset
  - resolved_technique = apply_hybrid

3. preserve_accuracy_preset(...)
- Intent: High-fidelity AI
- Strategy: apply_kd(..., epochs=20)
- Metadata:
  - strategy = preserve_accuracy
  - user_intent_layer = Preset
  - resolved_technique = apply_kd


## 9) Layer 3: Manual Override (Advanced)

Manual mode directly uses:
- apply_pruning(...)
- apply_quantization(...)
- apply_hybrid(...)
- apply_kd(...)

For manual routes, metadata annotation is enforced:
- user_intent_layer = Manual
- resolved_technique = apply_pruning | apply_quantization | apply_hybrid | apply_kd


## 10) Accuracy Guard and Stability Controls

Global default:
- DEFAULT_ACCURACY_DROP_THRESHOLD = 2.5

Helper behavior:
- _accuracy_guard(current_acc, baseline_acc, allowed_drop)
- _record_accuracy_checkpoint(...)
- _log_guard(...)

Used mainly in hybrid and KD loops to prevent over-aggressive compression and trigger adaptive behavior (retry with lower prune amount, reduce LR, or early stop).


## 11) Persistence and Returned Metrics

### 11.1 File outputs
- Compressed artifacts:
  - Preloaded: models/compressed/
  - Dynamic uploads: models/uploads/compressed/
- Result JSON snapshots:
  - results/<model>_<method>_compression_result.json
  - API-level result files for latest preloaded/dynamic

### 11.2 Common returned metric families
- Accuracy:
  - baseline_accuracy, compressed_accuracy
- Size:
  - original_size_MB, compressed_size_MB, size_reduction_percent, compression_ratio
- Latency:
  - baseline_latency_ms, latency_ms, latency_speedup_percent
- Emissions/Energy:
  - training_emissions_kg, inference_emissions_kg
  - baseline_total_emissions_kg, compressed_total_emissions_kg
  - emissions_reduction_percent
  - training_energy_kwh, inference_energy_kwh, total energy fields
- Defense/reporting fields:
  - co2_method
  - inference_images_processed
  - inference_iterations
  - benchmark_window_seconds / benchmark_warmup_seconds
  - energy_per_1k_images
  - hardware_saturation_level
  - bottleneck_analysis
  - runtime_warning / runtime_warnings
- Intent-layer metadata:
  - strategy (requested user strategy)
  - user_intent_layer (Smart | Preset | Manual)
  - resolved_technique (actual apply_* function used)
  - resolved_strategy (original internal strategy before annotation)


## 12) API-Level Validation and Status Reporting

In backend/main.py:
- Shared VALID_STRATEGIES is used by both preloaded and dynamic endpoints:
  - smart, maximize_speed, minimize_size, preserve_accuracy, pruning, quantization, hybrid, kd
- _strategy_label(...) maps these to user-facing labels.
- Preloaded compression runs in a background thread and exposes step-wise status:
  - loading_model -> loading_data -> compressing -> energy_tracking -> evaluating -> complete
- Dynamic compression runs synchronously per request with status/error tracking.


## 13) Practical Notes

1. Batch caps are intentional
- Many loops cap to 100 or 50 batches per epoch to keep runtime manageable.

2. Quantized kernels vs runtime target
- PyTorch quantized ops are CPU-centric; CUDA runtime can route through TensorRT or float fallbacks while still preserving compressed artifacts.

3. Pruning and actual size reduction
- Sparse-aware artifact saving remains important for file-size reduction.
- Structured Conv2d pruning targets hardware-meaningful channel/filter structure.

4. Fair metrics are benchmarked, not only projected
- The code prefers measured benchmark emissions and only projects when measurements are unavailable.
- Inference measurement is steady-state time based (5s), with 2s warmup excluded from tracked energy.

5. VRAM vs. activations
- We prioritize activation-memory stability over theoretical maximum batch size during FairMetrics benchmarking.
- For models resized to 224x224 (or higher), benchmark batch size is capped at 512 to avoid OOM in the 5-second steady-state window.


## 14) Quick Function Map

- Orchestration:
  - run_compression(...)
  - compress_dynamic(...)
  - run_smart_compression(...)
- Intent wrappers:
  - maximize_speed_preset(...)
  - minimize_size_preset(...)
  - preserve_accuracy_preset(...)
- Manual compression methods:
  - apply_pruning(...)
  - apply_quantization(...)
  - apply_hybrid(...)
  - apply_kd(...)
- Metadata helper:
  - _annotate_intent_result(...)
- Artifact/loading helpers:
  - save_smallest_artifact(...), save_sparse_state_dict(...)
  - load_uploaded_model(...)
- Pruning helper:
  - _apply_structured_pruning_step(...)
- Emissions/fairness:
  - _track_inference_emissions(...)
  - _track_training_emissions(...)
  - _build_fair_comparison_metrics(...)
  - _select_benchmark_batch_size(...), _rebuild_loader_for_benchmark(...)
