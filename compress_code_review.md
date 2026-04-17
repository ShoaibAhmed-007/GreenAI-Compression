# compress.py — Deep Code Review & Recommendations

## Overview

Your pipeline implements 4 active compression methods (Pruning, QAT Quantization, Knowledge Distillation, Hybrid) across 11 pretrained models. The code is **overall solid** and well-structured. Below is a detailed analysis of what's working well, what has real bugs/issues, and targeted improvements for your FYP goals.

---

## ✅ What's Done Well

| Area | Assessment |
|---|---|
| **Sparse saving** | `save_smallest_artifact` tries dense, fp16, gzip variants and keeps the smallest — genuinely clever |
| **GPU/CPU fallback** | Auto-detects CUDA, falls back to CPU for QAT correctly (required by PyTorch quantized kernels) |
| **Fair CO2 comparison** | `_build_fair_comparison_metrics` runs identical train+inference workloads on baseline and compressed models — this is methodologically correct for research |
| **Accuracy guard** | `_accuracy_guard` with configurable `allowed_drop` prevents runaway accuracy loss during pruning |
| **Evaluation correctness** | All evaluations use the **test set**, not training data — no data leakage |
| **KD setup** | Temperature T=4.0, alpha=0.3 (70% KD / 30% CE) is a proven configuration |
| **Transfer learning** | Two-phase: freeze backbone → train head, then unfreeze full model. This is textbook fine-tuning |
| **Distillation loss** | Correctly applies the T² scaling factor to the KL divergence term |
| **CodeCarbon tracking** | Online+offline fallback, per-phase tracking (compression vs. inference), PAK region set |

---

## 🔴 Real Issues / Bugs Found

### 1. Pruning: `apply_pruning` uses **layer-wise L1** but `apply_hybrid` uses **global L1** — inconsistent

- `apply_pruning` (line 1225–1227) loops and calls `prune.l1_unstructured(module, ...)` layer by layer
- `apply_hybrid` (line 1620–1624) uses `prune.global_unstructured(...)` which finds the globally smallest weights

**Why it matters:** Global unstructured is strictly better — it prunes the least important weights across the *entire* network, not forcing equal sparsity on each layer. Small layers (like the final FC) may lose important weights while large conv layers are barely touched with local pruning.

**Fix:** Change `apply_pruning` to use `prune.global_unstructured` as well.

```diff
- for module in model.modules():
-     if isinstance(module, (nn.Conv2d, nn.Linear)):
-         prune.l1_unstructured(module, name='weight', amount=step_amount)
+ params_to_prune = _collect_prunable_modules(model)
+ prune.global_unstructured(
+     params_to_prune,
+     pruning_method=prune.L1Unstructured,
+     amount=step_amount,
+ )
```

---

### 2. Pruning: only 100 batches per fine-tune epoch, but **full dataset** in QAT pre-finetuning (inconsistency)

- `apply_pruning` fine-tune: `if batch_idx >= 100: break` (line 1235, 1257)
- `apply_quantization` pre-finetune: **no batch cap** — runs the entire training dataset (line 1406)

**Why it matters:** For a fair FYP comparison, you want consistent fine-tuning budgets across strategies. This creates an unfair CO2 comparison — QAT gets more compute for recovery.

**Fix:** Add `max_batches` cap consistently across all strategies, or document the difference.

---

### 3. QAT: `evaluate()` during QAT training uses `dev=qat_dev` (CPU), but **final `evaluate` at line 1796** in hybrid uses `max_batches=50` instead of the full test set

```python
# Line 1796 — Hybrid strategy
hybrid_acc = evaluate(quant_model, test_loader, dev=cpu_dev, max_batches=50)
```

For the reported accuracy in your FYP, `max_batches=50` means you're testing on only 50*batch_size ≈ 6,400 samples, not the full CIFAR-10 test set (10,000 samples). This **underestimates accuracy variability** and is inconsistent with other strategies that use the full test set.

**Fix:**
```diff
- hybrid_acc = evaluate(quant_model, test_loader, dev=cpu_dev, max_batches=50)
+ hybrid_acc = evaluate(quant_model, test_loader, dev=cpu_dev)
```

---

### 4. KD: `save_path` uses `.pth` but `save_smallest_artifact` may rename it (with `_fp16`, etc.)

At line 1994, `torch.save(student.state_dict(), save_path)` is called inside the training loop. Then at line 2023, `save_smallest_artifact` is called — which may write a different file and delete the original. The subsequent `torch.load(save_path, ...)` at line 2020 may try to load the original path which no longer exists if the best artifact used a different suffix.

However, looking more carefully: line 2020 (`student.load_state_dict(...)`) runs **before** `save_smallest_artifact` at line 2023, so the order is correct. But there's still a risk that the intermediate `save_path` gets deleted by `save_smallest_artifact`.

**Low risk** — actually okay as long as `save_path` stays valid before line 2023. Just worth noting.

---

### 5. CO2 projection formula is **linear by model size** — academically weak

In `_build_fair_comparison_metrics` (line 695–701):

```python
compressed_total_co2 = _linear_co2_from_size_ratio(
    baseline_total_co2,
    baseline_size_mb,
    compressed_size_mb,
)
```

This assumes CO2 ∝ model size linearly. **This is not accurate** — small quantized models running on CPU may use *more* energy than the same FP32 model on GPU. CO2 is determined by:
- Hardware (CPU vs GPU TDP)
- Batch size
- Real inference throughput
- Grid carbon intensity

**For your FYP:** You should acknowledge this limitation in your thesis. The projections are useful for estimation but not precisely measured. You're already tracking actual benchmark emissions, which is the right thing to report.

---

### 6. Legacy `__main__` block: fine-tuning after 70% pruning uses `optim.SGD` **without** weight decay

```python
# Line 2807-2808 (legacy block)
optimizer = optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9)
```

No `weight_decay`. This risks overfitting during fine-tune recovery. The dynamic `apply_pruning` function correctly uses `weight_decay=5e-4` — so the API path is fine, but the legacy `__main__` path has a bug.

---

## 🟡 Technique-Level Recommendations

### A. Pruning: Consider Structured Pruning for Real Latency Gains

Your current L1 **unstructured** pruning only reduces file size (sparse tensors), not actual compute unless you have sparse inference hardware. For real latency gains, consider adding an option for **channel/filter pruning** (structured):

```python
# L2 norm structred filter pruning:
prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)
```

Unstructured pruning is fine for disk size and CO2 comparison, but your thesis should note that **inference speedup requires structured pruning or sparse acceleration hardware**.

### B. Quantization: QAT needs `torch.nn.intrinsic` fusion for Conv→BN→ReLU blocks

You only call `model.fuse_model()` when `hasattr(model, 'fuse_model')` (line 1425). For custom models (CompactStudent) and general architectures, this is never available. Without fusion, QAT's quantization observers are placed between BN and ReLU, leading to worse accuracy and larger quantized size.

**For maximum QAT accuracy**, you should manually fuse modules:
```python
torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']], inplace=True)
```

### C. KD: Temperature T=4.0 is good — but alpha=0.3 means only 30% hard-label CE

For CIFAR-10, T=4.0 and alpha=0.3 is a reasonable choice. If your student accuracy is below teacher by more than 3%, try:
- `T=3.0, alpha=0.5` (more balanced CE/KD)
- Increase KD epochs to 30+ for better convergence

### D. Hybrid: Pruning then QAT order is correct — don't reverse it

Pruning → QAT is the right order. QAT quantizes pruned weights more accurately because zero weights are cleanly represented in INT8. You're doing this correctly.

### E. CO2 Tracking: `measure_power_secs=1` may be too coarse for short compressions

For workloads under 5 seconds, CodeCarbon's 1-second sampling may miss the entire compute burst. For your FYP experiments that run in seconds, CodeCarbon will often report 0 kg CO2.

**Already handled in your code** with `_finalize_emissions_tracking` logging a warning when `emissions_kg <= 0`. This is acceptable for a research project — just document it in your thesis.

---

## 📊 For Your FYP Thesis — Key Claims to Make

| Compression Method | Typical Size Reduction | Accuracy Drop | CO2 Benefit |
|---|---|---|---|
| Unstructured Pruning (30%) | ~30-50% (sparse) | <1% | Proportional to size |
| QAT INT8 | ~75% (4x) | <0.5% | Significant (less memory bandwidth) |
| Knowledge Distillation | ~95% (parameter reduction) | 2-5% | Largest CO2 reduction |
| Hybrid (Pruning+QAT) | ~80-85% | <1.5% | Strong combination |

---

## 🔧 Priority Fixes for Best Results

1. **[High]** Fix `apply_pruning` to use `global_unstructured` instead of layer-wise — better accuracy preservation
2. **[High]** Remove `max_batches=50` cap in hybrid final evaluation — get full test set accuracy  
3. **[Medium]** Cap QAT pre-finetune batches consistently with pruning fine-tune (fairness in CO2 comparison)
4. **[Medium]** In thesis: clarify that unstructured pruning reduces disk size but not real-time latency without sparse kernel support
5. **[Low]** Add `weight_decay=5e-4` in legacy `__main__` fine-tune block (line 2807)

---

## Summary

The code is **research-quality and academically sound** for your FYP. The methodology is correct (train/test split, fair benchmarking, per-region CO2 tracking). The biggest actionable fix for better results is switching `apply_pruning` to use global unstructured pruning and removing the partial evaluation cap in the hybrid strategy's final accuracy check.
