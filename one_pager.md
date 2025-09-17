# Energy-Efficient Deep Learning: Reducing Carbon Footprint with Model Compression

## Problem Statement
Training deep neural networks requires high computational power, which leads to significant energy consumption 
and carbon emissions. While existing work focuses on accuracy improvements, fewer studies emphasize making 
AI models greener and sustainable.

## Research Gap
Most studies do not quantify the energy impact of compression methods or compare them on a standardized benchmark.

## Objectives
1. Train a baseline model (ResNet-18 on CIFAR-10).
2. Apply compression techniques (quantization, pruning, knowledge distillation).
3. Measure trade-offs between accuracy, efficiency, and energy.
4. Identify the most sustainable approach for small-scale deployment.

## Methodology (High-level)
- Dataset: CIFAR-10
- Baseline: ResNet-18
- Compression: Pruning, Quantization, Knowledge Distillation
- Metrics: Accuracy, FLOPs, Model Size, Training Time, Carbon Emissions (via CodeCarbon)

## Expected Outcomes
- A comparative study of compressed vs baseline models
- A guideline for energy-efficient model deployment
- Open-source code repository
