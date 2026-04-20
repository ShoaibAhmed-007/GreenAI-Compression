# FYP Report Methodology Update (April 2026)

## 1. Why This Update Is Needed

The earlier FYP draft described compression using only four manual techniques:

- pruning
- quantization
- knowledge distillation (KD)
- hybrid

The implementation has now evolved into a layered methodology that still includes these four techniques, but also adds smarter, intent-driven orchestration.

This document can be used to update the Methodology chapter so it matches the current system behavior.

## 2. Old vs Updated Methodology

### Previous Version (Technique-Only)

- Manual Pruning
- Manual Quantization
- Manual Hybrid
- Manual Knowledge Distillation

### Current Version (3-Layer Framework)

1. Layer 1: Auto-Green Smart Router
2. Layer 2: Intent Presets
3. Layer 3: Manual Advanced Techniques

The manual methods are not removed. They remain available as direct controls in Layer 3.

## 3. Current Compression Strategies (Canonical Keys)

The backend now validates and supports these eight strategy keys:

- smart
- maximize_speed
- minimize_size
- preserve_accuracy
- pruning
- quantization
- hybrid
- kd

Note for report wording:

- "Smallest Size" in user-facing explanation corresponds to `minimize_size` in implementation.

## 4. Layer-Wise Methodology for the Report

### Layer 1: Auto-Green Smart Router (`smart`)

Purpose:
- Automatically selects the most suitable compression path based on model characteristics.

Current routing behavior:
- Heavy CNN group -> Hybrid route (pruning + quantization)
- Lightweight/compact CNN group -> KD route

Result:
- Reduces manual trial-and-error.
- Provides architecture-aware automatic compression.

### Layer 2: Intent Presets (User Goal Driven)

These presets convert high-level deployment goals into compression pipelines.

1. Maximize Speed (`maximize_speed`)
- Goal: fastest inference runtime.
- Pipeline: quantization-focused compression.

2. Minimize Size / Smallest Size (`minimize_size`)
- Goal: minimum model storage footprint.
- Pipeline: aggressive hybrid compression (strong pruning + quantization).

3. Preserve Accuracy (`preserve_accuracy`)
- Goal: retain predictive performance as much as possible.
- Pipeline: knowledge distillation with conservative settings.

### Layer 3: Manual Advanced Methods

For expert users and ablation studies, direct techniques remain available:

- Pruning (`pruning`)
- Quantization (`quantization`)
- Hybrid (`hybrid`)
- Knowledge Distillation (`kd`)

This layer is especially useful for controlled experiments where each method must be tested independently.

## 5. Suggested FYP Report Text (Ready to Paste)

"The compression framework was upgraded from a technique-only approach to a three-layer methodology. In the initial version, only four direct techniques were available: pruning, quantization, hybrid compression, and knowledge distillation. In the updated system, these techniques are retained under a Manual Advanced layer, while two higher-level orchestration layers were introduced. The first is an Auto-Green Smart Router that selects a suitable compression path based on model characteristics. The second is an Intent Preset layer that exposes deployment-oriented choices: Maximize Speed, Minimize Size (Smallest Size), and Preserve Accuracy. This redesign improves usability and deployment relevance while preserving full experimental control for manual benchmarking." 

## 6. What Changed Methodologically

- Added architecture-aware automatic routing (`smart`).
- Added goal-driven preset layer (`maximize_speed`, `minimize_size`, `preserve_accuracy`).
- Kept existing manual techniques for reproducibility and scientific comparison.
- Shifted user interaction from "pick one low-level algorithm" to "choose deployment intent or manual control."

## 7. Recommended Figure/Table Update in Report

If your report includes a methodology table or flowchart, update it to show:

1. Input model
2. Strategy selection interface:
   - Smart Router
   - Presets (Speed / Smallest Size / Accuracy)
   - Manual (Pruning / Quantization / Hybrid / KD)
3. Compression execution
4. Evaluation outputs (accuracy, size, latency, energy, CO2)
