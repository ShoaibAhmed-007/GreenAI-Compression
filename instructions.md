# Green AI Platform — Full Pipeline Refactor Instructions

## Context

We have built a Green AI model compression platform that:

1. Trains CNN models on CIFAR-10
2. Applies compression techniques
3. Evaluates performance
4. Measures energy consumption using CodeCarbon
5. Returns results to the frontend

However several serious problems exist:

• Some models have **very low CIFAR-10 accuracy**
• Some models **do not shrink after compression**
• **CodeCarbon emissions are not recorded**
• Training pipeline is **not optimized**
• Dataset preprocessing is inconsistent
• Model classifiers may not be correctly adapted to CIFAR-10
• Compression pipeline does not work consistently across models

The goal is to **refactor the training and compression pipeline so results are reliable and reproducible**.

---

# Task 1 — Fix CIFAR-10 Training Pipeline

Rewrite `train.py` so all models train properly on CIFAR-10.

### Requirements

1. Correct CIFAR-10 preprocessing

```
RandomCrop(32, padding=4)
RandomHorizontalFlip()
Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
```

Validation set must NOT use augmentation.

---

### Resize Inputs

Most ImageNet models require larger inputs.

Use:

| Model                         | Input |
| ----------------------------- | ----- |
| ResNet / DenseNet / MobileNet | 224   |
| EfficientNet                  | 224   |
| InceptionV3                   | 299   |

---

### Fix Classifier Layers

All models must output **10 classes**.

Examples:

ResNet

```
model.fc = nn.Linear(model.fc.in_features, 10)
```

DenseNet

```
model.classifier = nn.Linear(model.classifier.in_features, 10)
```

MobileNet

```
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
```

EfficientNet

```
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
```

GoogLeNet

```
model.fc = nn.Linear(model.fc.in_features, 10)
```

---

### Optimizer

Use SGD with momentum.

```
optimizer = SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)
```

---

### Learning Rate Scheduler

Use cosine schedule.

```
CosineAnnealingLR
```

---

### Training Parameters

```
epochs = 50
batch_size = 128
```

---

### Mixed Precision

Use AMP for faster training.

```
torch.cuda.amp.autocast()
GradScaler
```

---

### DataLoader Optimization

```
num_workers = 4
pin_memory = True
```

---

# Task 2 — Achieve Proper CIFAR-10 Accuracy

Expected accuracy ranges:

| Model          | Expected Accuracy |
| -------------- | ----------------- |
| ResNet18       | 92–94%            |
| ResNet34       | 93–95%            |
| MobileNetV2    | 88–91%            |
| EfficientNetB0 | 90–93%            |
| EfficientNetB1 | 91–93%            |
| DenseNet121    | 92–94%            |
| DenseNet169    | 93–95%            |
| SqueezeNet     | 85–88%            |
| ShuffleNetV2   | 88–91%            |
| InceptionV3    | 92–94%            |
| GoogLeNet      | 90–92%            |

If accuracy is significantly lower, training must be adjusted.

---

# Task 3 — Fix Compression Pipeline

Current issue: pruning sets weights to zero but **model size does not change**.

Fix compression process.

### Step 1 — Pruning

Use magnitude pruning.

After pruning remove masks.

```
prune.remove(module, 'weight')
```

---

### Step 2 — Sparse Saving

Convert tensors to sparse format.

```
tensor.to_sparse()
```

Save sparse state_dict.

---

### Step 3 — Quantization

Apply dynamic quantization.

```
torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
```

---

### Step 4 — Verify Compression

Calculate:

```
original_size_MB
compressed_size_MB
compression_ratio
```

---

# Task 4 — Compression Methods

Ensure the following strategies work for all models:

1️⃣ Pruning
2️⃣ Quantization
3️⃣ Knowledge Distillation
4️⃣ Hybrid (student + quantization)

---

# Task 5 — CodeCarbon Integration

Energy usage must be tracked.

Wrap training and compression code:

```
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()

tracker.start()

# training or compression

emissions = tracker.stop()
```

Store:

```
energy_kwh
co2_kg
```

---

# Task 6 — Location Based Emissions

CodeCarbon should attempt geographic fallback.

Priority:

1️⃣ DHA Phase 6
2️⃣ Lahore
3️⃣ Punjab
4️⃣ Pakistan

Implementation logic:

```
region="Punjab"
country_iso_code="PAK"
```

---

# Task 7 — Results JSON

Each experiment must save metrics:

```
{
  model_name,
  accuracy,
  parameters,
  original_size_MB,
  compressed_size_MB,
  latency_ms,
  energy_kwh,
  co2_kg
}
```

---

# Task 8 — Evaluation Pipeline

Create a consistent evaluation function that measures:

• Accuracy
• Model size
• Latency
• Energy
• CO₂ emissions

---

# Final Goal

After refactoring:

Example expected results:

ResNet18 → ~93% accuracy
MobileNetV2 → ~90% accuracy
DenseNet121 → ~93% accuracy

Compression should reduce size by **30–95%** depending on method.

Energy consumption must be recorded for every experiment.

---

# Important

Do not only patch the code.

Refactor the pipeline so:

• training is reproducible
• compression works across architectures
• energy metrics are reliable
• results are saved automatically.
