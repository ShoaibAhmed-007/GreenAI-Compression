# Green AI – Model Compression

A full-stack application for exploring **energy-efficient deep learning** through model compression. It applies multiple compression techniques — pruning, quantization, and knowledge distillation — to a ResNet-18 model trained on CIFAR-10, then measures and visualises accuracy, model size, inference latency, energy consumption, and CO₂ emissions.

## Overview

Deep learning models are increasingly powerful but come with significant energy and carbon costs. This project demonstrates that aggressive compression can slash model size by up to **99 %** while retaining competitive accuracy and dramatically reducing energy usage.

### Compression Strategies

| # | Strategy | Description |
|---|----------|-------------|
| 1 | **Unstructured Pruning** | 70 % weight pruning + fine-tuning, saved in sparse format |
| 2 | **Post-Training Quantization** | Static INT8 quantization (weights & activations) |
| 3 | **Knowledge Distillation** | Compact student model trained from ResNet-18 teacher (~95 % smaller) |
| 4 | **Hybrid** | Compact student + dynamic quantization |
| 5 | **Ultra-Compact** | Pruned student + quantization + sparse save |

### Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 14, React 18, Tailwind CSS, Recharts |
| **Backend** | FastAPI, Uvicorn |
| **ML** | PyTorch, torchvision, CodeCarbon |
| **Dataset** | CIFAR-10 (auto-downloaded on first run) |

---

## Project Structure

```
GreenAI-Compression/
├── backend/
│   ├── main.py          # FastAPI server (REST API)
│   ├── train.py         # Baseline ResNet-18 training
│   ├── compress.py      # 5 compression strategies
│   ├── evaluate.py      # Accuracy, latency, FLOPs evaluation
│   └── energy.py        # Energy & CO₂ tracking (CodeCarbon)
├── frontend/
│   └── src/
│       ├── app/         # Next.js pages & layout
│       ├── components/  # Dashboard UI components
│       └── lib/         # API client
├── models/              # Saved model weights (.pth) – git-ignored
├── results/             # JSON metrics & reports
└── papers/              # Reference literature
```

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm**
- (Optional) NVIDIA GPU with CUDA for faster training

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/GreenAI-Compression.git
cd GreenAI-Compression
```

### 2. Backend Setup

```bash
# Create and activate a virtual environment
python -m venv greenai_env

# Windows
greenai_env\Scripts\activate

# macOS / Linux
source greenai_env/bin/activate

# Install Python dependencies
pip install torch torchvision fastapi uvicorn pydantic codecarbon pynvml

# (Recommended) CUDA allocator setting to reduce memory fragmentation
# PowerShell (Windows)
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Bash (macOS/Linux)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (Optional) Train the baseline model — downloads CIFAR-10 automatically
cd backend
python train.py

# (Optional) Run compression strategies
python compress.py

# (Optional) Evaluate all models
python evaluate.py

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000**.  
Interactive docs at **http://localhost:8000/docs**.

### 3. Frontend Setup

Open a **new terminal**:

```bash
cd frontend

# Install Node dependencies
npm install

# Start the development server
npm run dev
```

The dashboard will be available at **http://localhost:3000**.

> **Note:** Make sure the backend server is running on port 8000 before opening the frontend — it fetches all data from the API.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/results` | All compression results summary |
| `GET` | `/api/results/{strategy}` | Metrics for a specific strategy |
| `GET` | `/api/energy` | Energy & emissions report |
| `GET` | `/api/evaluation` | Full evaluation report |
| `GET` | `/api/models` | List available models with sizes |
| `POST` | `/api/compress` | Trigger compression on the baseline model |
| `GET` | `/api/compare` | Side-by-side comparison of all strategies |

---

## License

This project is developed as a Final Year Project (FYP) for academic purposes.
