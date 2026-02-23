# GreenAI FYP — Claude Instructions

## 🔹 Context

I am working on my **Final Year Project (FYP)** called **GreenAI**, focused on **compressing deep learning models** for edge devices to save energy and reduce carbon emissions.

Currently, **Phase 6 — Model Compression** has been completed with the following results:

---

## 🟢 Phase 6 Results

**Using device: CUDA**

### Baseline Model (ResNet18)

| Metric  | Value          |
|---------|----------------|
| Accuracy | 85.72%        |
| Size     | 44.81 MB      |
| Params   | 11,181,642    |
| Latency  | 3.74 ms       |

---

### Strategy 1 — Unstructured Pruning (70%) + Fine-tune + Sparse Save

* **Why:** Setting 70% of smallest-magnitude weights to zero, then saving in sparse format.  
* **Fine-tuning:** 5 epochs

| Epoch | Loss   | Acc%  |
|-------|--------|-------|
| 1     | 0.2003 | 85.51 |
| 2     | 0.2013 | 85.6  |
| 3     | 0.1956 | 85.43 |
| 4     | 0.1965 | 85.69 |
| 5     | 0.1917 | 85.7  |

**Results:**

| Metric       | Value          |
|--------------|----------------|
| Accuracy     | 85.7%          |
| Std Save Size| 44.81 MB       |
| Sparse Save  | 72.51 MB (↓ -61.82%) |
| Sparsity     | 58.79%         |
| Latency      | 3.64 ms        |

---

### Strategy 2 — Post-Training Static Quantization (INT8)

* **Why:** Convert weights and activations from float32 → int8 (~4× compression).  
* **Calibration:** 50 batches

**Results:**

| Metric  | Value          |
|---------|----------------|
| Accuracy| 85.67%         |
| Size    | 11.31 MB (↓ 74.76%) |
| Latency | 2.2 ms         |

---

### Strategy 3 — Knowledge Distillation → Compact Student Model

* **Why:** Distill ResNet18 teacher → MobileNet-style student (~543K params vs 11M params).  
* **Training:** 20 epochs

| Epoch | Loss   | Acc%  | Best Acc% |
|-------|--------|-------|------------|
| 5     | 1.2418 | 78.99 | 78.99      |
| 10    | 0.8163 | 83.17 | 83.17      |
| 15    | 0.6521 | 85.39 | 85.39      |
| 20    | 0.6002 | 85.85 | 85.85      |

**Results:**

| Metric   | Value          |
|----------|----------------|
| Accuracy | 85.85%         |
| Size     | 2.23 MB (↓ 95.02%) |
| Params   | 543,050        |
| Latency  | 2.16 ms        |

---

### Strategy 4 — Hybrid: Compact Student + Dynamic Quantization

**Results:**

| Metric  | Value          |
|---------|----------------|
| Accuracy| 85.86%         |
| Size    | 2.21 MB (↓ 95.07%) |
| Latency | 2.27 ms        |

---

### Strategy 5 — Ultra-Compact: Pruned Student + Quantization + Sparse

* **Why:** Maximum compression by combining all techniques.  
* **Fine-tuning:** 10 epochs

| Epoch | Loss   | Acc%  |
|-------|--------|-------|
| 5     | 0.6257 | 85.71 |
| 10    | 0.5744 | 86.56 |

**Results:**

| Metric     | Value          |
|------------|----------------|
| Accuracy   | 86.57%         |
| Sparse Size| 2.23 MB (↓ 95.02%) |
| Quant Size | 2.21 MB (↓ 95.07%) |
| Sparsity   | 0%             |
| Latency    | 2.37 ms        |

---

### Compression Results Summary

| Strategy                          | Acc%  | Size MB | ↓ Size% | Latency |
|----------------------------------|-------|---------|---------|---------|
| Baseline (ResNet18)              | 85.72 | 44.81   | —       | 3.74ms |
| 1. Pruning 70% + Sparse          | 85.7  | 72.51   | -61.82% | 3.64ms |
| 2. Static Quantization INT8       | 85.67 | 11.31   | 74.76%  | 2.2ms  |
| 3. KD → Compact Student           | 85.85 | 2.23    | 95.02%  | 2.16ms |
| 4. Compact Student + Quant        | 85.86 | 2.21    | 95.07%  | 2.27ms |
| 5. Ultra: Prune+Quant+Sparse      | 86.57 | 2.23    | 95.02%  | 2.37ms |

All results are saved to `../results/`  
All models are saved to `../models/`

---

## 🔹 Objective for Claude Code

1. **Implement all remaining phases (7–10)**:

   **Phase 7 — Energy Tracking**  
   - Track energy and CO2 emissions during training/inference (CodeCarbon).  

   **Phase 8 — FastAPI Backend**  
   - Endpoint to receive model + compression method  
   - Apply compression  
   - Return JSON metrics: accuracy, size reduction, FLOPs reduction, emissions  

   **Phase 9 — Next.js Frontend**  
   - Upload model, select compression, display metrics/charts  

   **Phase 10 — Demo Flow**  
   - Full end-to-end working demo: upload → compress → results → emissions  

2. **Review and improve Phases 1–6**:

   * Improve pruning/quantization for **actual size reduction**  
   * Avoid CPU-only quantization errors  
   * Optimize GPU usage  
   * Ensure energy tracking is accurate  
   * Suggest improvements in folder structure, scripts, or workflow  

3. **Provide ready-to-run Python code**:

   - `train.py`, `compress.py`, `evaluate.py`, `energy.py`, `main.py`  
   - FastAPI backend template  
   - Next.js frontend template  

4. **Constraints**:

   * Use **ResNet-18** and **CIFAR-10** for experiments  
   * Keep **fast, GPU-accelerated** workflow  
   * **JSON outputs** for frontend  
   * Track **energy / emissions**  

5. **Output Expected**:

   * Full implementation for Phases 1–10  
   * Suggestions for compression improvements  
   * Ready-to-run backend + frontend  
   * Experiment workflow reproducible on GPU laptop  