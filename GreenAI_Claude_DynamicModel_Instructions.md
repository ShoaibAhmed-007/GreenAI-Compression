🌟 Dynamic Model Compression — Preloaded Model Approach
Overview

Previously, the compression pipeline was attempted as fully dynamic, allowing any uploaded model to be compressed. This caused multiple errors due to differences in layer structures, quantization compatibility, and pruning requirements.

To simplify and stabilize the system, we now switch to a preloaded model approach:

The frontend allows the user to select from 15 curated pretrained PyTorch models.

Once selected, the backend applies the compress.py pipeline (Pruning, Quantization, Hybrid, Knowledge Distillation).

The backend returns metrics (accuracy, model size, latency, emissions).

This ensures all compression techniques work reliably without runtime errors.

⚡ Backend Flow (FastAPI)

Model Selection: User selects one of the 15 models from a dropdown.

Compression Method: User selects method:

Pruning

Quantization

Hybrid

Knowledge Distillation

Compression Execution:

from compress import run_compression
metrics = run_compression(model_path, method)

Evaluation & Metrics:

Accuracy

Model size

Latency

FLOPs

Energy/CO2 emissions

Return JSON to Frontend:

{
    "model_name": "ResNet18",
    "compression_method": "Quantization",
    "accuracy": 85.67,
    "size_MB": 11.31,
    "latency_ms": 2.2,
    "emissions_kg": 0.0021
}
🖥 Frontend Flow (Next.js)

Dropdown for Model Selection

Example: ResNet18, ResNet34, VGG16, MobileNetV2, EfficientNetB0 ... (total 15 models)

Compression Method Selection

Dropdown: Pruning / Quantization / Hybrid / KD

Submit Button

Sends POST request to FastAPI backend:

POST /compress
Body: { "model_name": "ResNet18", "method": "Pruning" }

Display Results

Cards for metrics: Accuracy, Size, Latency, CO2 emissions

Optional charts for comparison between baseline & compressed model

✅ Preloaded Models List (Example)
Model Name	Params	Typical Dataset
ResNet18	11.2M	CIFAR-10 / ImageNet
ResNet34	21.8M	CIFAR-10 / ImageNet
ResNet50	25.6M	ImageNet
VGG16	138M	CIFAR-10 / ImageNet
VGG19	143M	CIFAR-10 / ImageNet
MobileNetV2	3.4M	CIFAR-10 / ImageNet
EfficientNetB0	5.3M	ImageNet
EfficientNetB1	7.8M	ImageNet
DenseNet121	8.0M	CIFAR-10 / ImageNet
DenseNet169	14.3M	CIFAR-10 / ImageNet
SqueezeNet	1.2M	CIFAR-10 / ImageNet
ShuffleNetV2	1.0M	CIFAR-10 / ImageNet
AlexNet	61M	CIFAR-10 / ImageNet
InceptionV3	23.8M	ImageNet
GoogLeNet	6.8M	ImageNet

Note: These .pt files are stored in /backend/models/ and loaded dynamically by backend when a user selects a model.

⚙ Implementation Notes

Safe Compression:
Each model is compatible with compress.py pipeline (pruning, quantization, hybrid, KD).

Backend Validation:
Backend checks:

if model_name not in allowed_models:
    raise ValueError("Model not supported")

Energy Tracking:
Every compression run tracks CO2 emissions using codecarbon.

Future Expansion:
You can add more models to /backend/models/ and update the frontend dropdown.

Avoid Fully Dynamic Uploads:

Reduces errors from unsupported layers, dynamic shapes, or custom architectures.

Keeps pipeline stable and predictable.

🏁 Workflow Summary

User selects model from dropdown

User selects compression method

Backend loads model from /backend/models/

Runs compress.py pipeline

Evaluates metrics (accuracy, size, latency, emissions)

Returns results to frontend for display