# Agent Task: Final Green AI Pipeline Optimizations for compress.py

## Objective
Update the `compress.py` pipeline to ensure hardware-aware benchmarking, true structural compression, and valid energy metrics for high-end NVIDIA GPUs (RTX 3070 / RTX 5090). Quantization is already handled; focus on Pruning, Benchmarking Stability, and Batch Saturation.

## 1. Shift to Structured Pruning (Hardware-Friendly)
Unstructured L1 pruning creates masks but does not reduce compute overhead. We must shift to Structured Pruning (removing entire filters/channels) to see actual latency and energy reductions.

**Update `apply_pruning` and the Prune Phase of `apply_hybrid`:**
- Replace `prune.global_unstructured` and `prune.L1Unstructured`.
- Implement **L2 Structured Pruning** across the output channels (`dim=0`) of `nn.Conv2d` layers.
- Example replacement logic:
  ```python
  import torch.nn.utils.prune as prune

  for module in prunable_modules:
      if isinstance(module, torch.nn.Conv2d):
          # Prune entire filters based on L2 norm
          prune.ln_structured(module, name="weight", amount=step_amount, n=2, dim=0)
      elif isinstance(module, torch.nn.Linear):
          # Fallback to unstructured for linear layers if structured is too destructive
          prune.l1_unstructured(module, name="weight", amount=step_amount)
  ```

## 2. Implement Temporal Benchmarking (Steady-State Emissions)
Short inference bursts (e.g., < 1 second) produce noise in CodeCarbon. We must enforce a time-based steady-state loop instead of a fixed batch limit.

**Update `_track_inference_emissions`:**
- Remove `max_batches` as the primary loop break condition.
- Implement a strict 5.0 - 10.0 second temporal window.
- Ensure the data loader cycles infinitely if it runs out of batches before the time limit.
- Example logic:
  ```python
  benchmark_duration = 5.0  # Force 5 seconds of continuous GPU compute
  start_bench = time.time()
  iterations = 0

  model.eval()
  with torch.no_grad():
      while (time.time() - start_bench) < benchmark_duration:
          for inputs, _ in loader:
              if (time.time() - start_bench) >= benchmark_duration:
                  break
              inputs = inputs.to(dev)
              _ = model(inputs)
              iterations += 1
  torch.cuda.synchronize()
  ```

## 3. Dynamic Batch Sizing (Hardware Saturation)
To overcome the orchestration overhead on powerful GPUs, we must saturate the Tensor Cores. Lightweight models need massive batches to show green efficiency.

**Update Data Loader Initialization (`get_data_loaders` or caller):**
- Dynamically set the `benchmark_batch_size` based on model complexity (Parameter count).
- Logic:
  ```python
  if total_params < 5_000_000:
      target_batch_size = 1024  # Saturate lightweight models (MobileNet, ShuffleNet)
  elif total_params < 15_000_000:
      target_batch_size = 512   # Medium models (ResNet18)
  else:
      target_batch_size = 256   # Heavy models (ResNet34, Inception)
  ```

## 4. Hardware Warmup & cuDNN Optimization
Ensure the GPU is prepared and using the most efficient convolution algorithms before CodeCarbon tracking begins.

**Global Updates before tracking starts:**
- Set `torch.backends.cudnn.benchmark = True` at the start of the benchmarking phase to allow the GPU to auto-select the lowest-power convolution algorithms.
- Add a **2-second warmup loop** immediately before calling `_start_emissions_tracker(...)` in both inference and training evaluation blocks. Do not track the energy of the first few batches where JIT compilation and memory allocation occur.

## 5. Hybrid Accuracy Guard Enhancement
In `apply_hybrid`, ensure the `_accuracy_guard` aggressively triggers a recovery fine-tuning cycle if the structured pruning step causes the accuracy to drop by more than `DEFAULT_ACCURACY_DROP_THRESHOLD` (e.g., 2.5%), *before* moving to the quantization phase.
