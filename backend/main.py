# main.py
"""
Green AI FYP — Phase 8: FastAPI Backend
========================================
REST API endpoints for the GreenAI compression dashboard:

  GET  /api/health            — Health check
  GET  /api/results           — All compression results summary
  GET  /api/results/{strategy} — Metrics for a specific strategy
  GET  /api/energy            — Energy & emissions report
  GET  /api/evaluation        — Full evaluation report
  GET  /api/models            — List available models with sizes
  POST /api/compress          — Trigger compression on the baseline model
  POST /api/compress/dynamic  — Upload any model + apply dynamic compression
  GET  /api/compare           — Side-by-side comparison of all strategies

Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import json
import re
import subprocess
import sys
import time
import shutil
import traceback
import threading
import torch

app = FastAPI(
    title="GreenAI — Model Compression API",
    description="API for compressing deep learning models and tracking energy savings",
    version="1.0.0",
)

# CORS — allow frontend (Next.js dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "uploads")
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "pretrained_baselines")
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure directories
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRETRAINED_DIR, exist_ok=True)

# Track background task status
task_status = {
    "compress": {"running": False, "last_run": None, "error": None},
    "evaluate": {"running": False, "last_run": None, "error": None},
    "energy": {"running": False, "last_run": None, "error": None},
}


# ============================================================
# Helpers
# ============================================================
def load_json(filename):
    """Load a JSON file from the results directory."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _normalize_model_key(value: str) -> str:
    """Normalize model key names to a stable lowercase underscore format."""
    key = (value or "").strip().lower().replace("-", "_")
    key = re.sub(r"\s+", "_", key)
    key = re.sub(r"[^a-z0-9_]", "", key)
    return key


def _format_params_label(total_params: Optional[int]) -> str:
    """Format parameter counts as compact human-readable labels (e.g., 11.2M)."""
    if total_params is None:
        return "N/A"
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.1f}B"
    if total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.1f}M"
    if total_params >= 1_000:
        return f"{total_params / 1_000:.1f}K"
    return str(total_params)


def list_models():
    """List all model files with sizes."""
    models = []
    if os.path.exists(MODELS_DIR):
        for f in sorted(os.listdir(MODELS_DIR)):
            fpath = os.path.join(MODELS_DIR, f)
            if os.path.isfile(fpath):
                models.append({
                    "filename": f,
                    "size_MB": round(os.path.getsize(fpath) / 1e6, 2),
                })
    return models


def _save_to_compression_history(model_key: str, result: dict):
    """Append a compression result to the history file, keyed by model."""
    history_path = os.path.join(RESULTS_DIR, "compression_history.json")
    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = {}

    if model_key not in history:
        history[model_key] = []

    # Deduplicate by strategy (keep latest)
    strategy = result.get("strategy", result.get("compression_method", "unknown"))
    history[model_key] = [
        r for r in history[model_key]
        if (r.get("strategy") or r.get("compression_method")) != strategy
    ]
    history[model_key].append(result)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# ============================================================
# Baseline Models — pre-trained baselines for all 15 models
# ============================================================

def load_baseline_results():
    """Load baseline-like model metrics by aggregating per-model training result files."""
    suffixes = ("_training_result.json", "_training_results.json")
    baselines = {}

    if not os.path.exists(RESULTS_DIR):
        return baselines

    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.endswith(suffixes):
            continue

        path = os.path.join(RESULTS_DIR, filename)
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        model_name_raw = payload.get("model_name") or filename
        model_key = _normalize_model_key(model_name_raw)
        if not model_key:
            model_key = _normalize_model_key(
                filename.replace("_training_result.json", "").replace("_training_results.json", "")
            )

        total_params = payload.get("total_params", payload.get("parameters"))
        size_mb = payload.get("size_MB", payload.get("original_size_MB"))

        baselines[model_key] = {
            "model_key": model_key,
            "model_name": model_name_raw,
            "params_label": _format_params_label(total_params),
            "total_params": total_params,
            "input_size": payload.get("input_size"),
            "dataset": payload.get("dataset", "CIFAR10"),
            "accuracy": payload.get("accuracy"),
            "size_MB": round(size_mb, 2) if isinstance(size_mb, (int, float)) else size_mb,
            "latency_ms": payload.get("latency_ms"),
            "status": "ready",
        }

    return baselines


@app.get("/api/baselines")
async def get_baselines():
    """
    Get baseline metrics for all 15 pretrained models.
    Returns pre-computed accuracy, size, latency, params for each model.
    If not yet prepared, returns available model list with 'not_ready' status.
    """
    sys.path.insert(0, BACKEND_DIR)
    from compress import PRELOADED_MODELS

    baselines = load_baseline_results()

    # Merge with PRELOADED_MODELS metadata and mark missing entries as not_ready
    for key, cfg in PRELOADED_MODELS.items():
        if key not in baselines:
            baselines[key] = {
                "model_key": key,
                "model_name": cfg["name"],
                "params_label": cfg["params"],
                "input_size": cfg["input_size"],
                "dataset": cfg["dataset"],
                "status": "not_ready",
            }
            continue

        # Keep canonical display metadata from configured preloaded model catalog.
        baselines[key]["model_name"] = cfg["name"]
        baselines[key]["params_label"] = baselines[key].get("params_label") or cfg["params"]
        baselines[key]["input_size"] = baselines[key].get("input_size") or cfg["input_size"]
        baselines[key]["dataset"] = baselines[key].get("dataset") or cfg["dataset"]

    ready = sum(1 for v in baselines.values() if v.get("status") == "ready")

    return {"models": baselines, "ready_count": ready, "total_count": len(baselines)}


@app.get("/api/baselines/{model_key}")
async def get_baseline_detail(model_key: str):
    """Get baseline metrics for a specific model."""
    baselines = load_baseline_results()
    key = _normalize_model_key(model_key)
    if key not in baselines:
        raise HTTPException(status_code=404, detail=f"Baseline not found for '{model_key}'")
    return baselines[key]


# Track prepare task
prepare_task = {
    "running": False,
    "progress": "",
    "completed": 0,
    "total": 0,
    "current_model": "",
    "error": None,
}


@app.post("/api/prepare")
async def trigger_prepare(models: list[str] = None):
    """
    Start pre-downloading and evaluating one or more models.
    If models is None, prepares all 15 models.
    Runs in background — poll /api/prepare/status.
    """
    if prepare_task["running"]:
        raise HTTPException(status_code=409, detail="Preparation already running")

    sys.path.insert(0, BACKEND_DIR)
    from compress import PRELOADED_MODELS

    if models is None:
        model_keys = list(PRELOADED_MODELS.keys())
    else:
        model_keys = [m.lower().strip() for m in models]
        invalid = [m for m in model_keys if m not in PRELOADED_MODELS]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model(s): {invalid}. Available: {list(PRELOADED_MODELS.keys())}"
            )

    prepare_task["running"] = True
    prepare_task["progress"] = "Starting..."
    prepare_task["completed"] = 0
    prepare_task["total"] = len(model_keys)
    prepare_task["current_model"] = ""
    prepare_task["error"] = None

    def _run():
        try:
            from prepare_models import prepare_single_model
            import torch

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load existing
            results = {}
            results_path = os.path.join(RESULTS_DIR, "baseline_all_models.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    results = json.load(f)

            for i, key in enumerate(model_keys):
                if key in results and results[key].get("status") == "ready":
                    prepare_task["completed"] = i + 1
                    prepare_task["progress"] = f"{key} already ready, skipping"
                    prepare_task["current_model"] = key
                    continue

                prepare_task["current_model"] = key
                prepare_task["progress"] = f"Preparing {key} ({i+1}/{len(model_keys)})"

                try:
                    result = prepare_single_model(key, device=device)
                    results[key] = result
                except Exception as e:
                    results[key] = {
                        "model_key": key,
                        "model_name": PRELOADED_MODELS[key]["name"],
                        "status": "error",
                        "error": str(e),
                    }

                prepare_task["completed"] = i + 1

                # Save incrementally
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

            prepare_task["progress"] = "Complete"
        except Exception as e:
            prepare_task["error"] = str(e)
            traceback.print_exc()
        finally:
            prepare_task["running"] = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return JSONResponse(
        status_code=202,
        content={
            "message": f"Preparing {len(model_keys)} model(s)",
            "models": model_keys,
            "status_endpoint": "/api/prepare/status",
        }
    )


@app.get("/api/prepare/status")
async def prepare_status():
    """Check preparation progress."""
    return prepare_task


# ============================================================
# Compression History — saved results per model
# ============================================================

@app.get("/api/compression-history")
async def get_compression_history():
    """Get all saved compression results, organized by model."""
    path = os.path.join(RESULTS_DIR, "compression_history.json")
    if not os.path.exists(path):
        return {"history": {}}
    with open(path, "r") as f:
        return {"history": json.load(f)}


@app.get("/api/results")
async def get_results():
    """Get the full compression results summary."""
    data = load_json("compression_summary.json")
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No compression results found. Run compress.py first."
        )
    return data


@app.get("/api/results/{strategy}")
async def get_strategy_results(strategy: str):
    """
    Get results for a specific compression strategy.
    Valid keys: baseline, pruning_compressed, quantization_static,
    kd_compact_student, hybrid_student_quant, ultra_compact
    """
    data = load_json("compression_summary.json")
    if data is None:
        raise HTTPException(status_code=404, detail="No results found")

    if strategy not in data:
        available = list(data.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy}' not found. Available: {available}"
        )
    return {strategy: data[strategy]}


@app.get("/api/energy")
async def get_energy():
    """Get energy and emissions tracking report."""
    data = load_json("energy_report.json")
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No energy report found. Run energy.py first."
        )
    return data


@app.get("/api/evaluation")
async def get_evaluation():
    """Get full evaluation report (accuracy, FLOPs, latency, per-class)."""
    data = load_json("evaluation_report.json")
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation report found. Run evaluate.py first."
        )
    return data


@app.get("/api/models")
async def get_models():
    """List all saved model files and their sizes."""
    return {"models": list_models()}


@app.get("/api/compare")
async def compare_strategies():
    """
    Side-by-side comparison table of all compression strategies.
    Returns a normalized format for easy frontend rendering.
    """
    summary = load_json("compression_summary.json")
    energy = load_json("energy_report.json")
    evaluation = load_json("evaluation_report.json")

    if summary is None:
        raise HTTPException(status_code=404, detail="No results found")

    comparison = []
    for key, metrics in summary.items():
        entry = {
            "strategy": key,
            "accuracy": metrics.get("accuracy",
                        metrics.get("accuracy_top1", None)),
            "size_MB": metrics.get("size_MB",
                       metrics.get("size_MB_compressed",
                       metrics.get("size_MB_sparse", None))),
            "size_reduction_percent": metrics.get("size_reduction_percent",
                                     metrics.get("size_reduction_compressed_percent", 0)),
            "latency_ms": metrics.get("latency_ms", None),
            "params": metrics.get("params",
                      metrics.get("student_params",
                      metrics.get("total_params", None))),
        }

        # Add energy data if available
        if energy and "inference" in energy:
            inf_data = energy["inference"].get(key, {})
            entry["inference_energy_kWh"] = inf_data.get("energy_kWh", None)
            entry["co2_kg"] = inf_data.get("co2_kg", None)

        # Add evaluation data if available
        if evaluation and key in evaluation:
            eval_data = evaluation[key]
            entry["accuracy_top5"] = eval_data.get("accuracy_top5", None)
            entry["flops_M"] = eval_data.get("flops_M", None)
            entry["sparsity_percent"] = eval_data.get("sparsity_percent", None)

        comparison.append(entry)

    return {"comparison": comparison}


# ============================================================
# Background task endpoints
# ============================================================

class CompressRequest(BaseModel):
    """Request body for compression endpoint."""
    strategies: Optional[list] = None  # If None, run all
    description: Optional[str] = "Run compression pipeline"


def run_script(script_name, task_key):
    """Run a Python script as subprocess and track status."""
    task_status[task_key]["running"] = True
    task_status[task_key]["error"] = None
    try:
        python_exe = sys.executable
        result = subprocess.run(
            [python_exe, script_name],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        if result.returncode != 0:
            task_status[task_key]["error"] = result.stderr[-500:]
        task_status[task_key]["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except subprocess.TimeoutExpired:
        task_status[task_key]["error"] = "Script timed out (1 hour limit)"
    except Exception as e:
        task_status[task_key]["error"] = str(e)
    finally:
        task_status[task_key]["running"] = False


@app.post("/api/compress")
async def trigger_compression(background_tasks: BackgroundTasks):
    """
    Trigger the full compression pipeline in the background.
    Returns immediately; check /api/task-status for progress.
    """
    if task_status["compress"]["running"]:
        raise HTTPException(
            status_code=409,
            detail="Compression is already running"
        )
    background_tasks.add_task(run_script, "compress.py", "compress")
    return {
        "message": "Compression pipeline started in background",
        "status_endpoint": "/api/task-status"
    }


@app.post("/api/evaluate")
async def trigger_evaluation(background_tasks: BackgroundTasks):
    """Trigger model evaluation in the background."""
    if task_status["evaluate"]["running"]:
        raise HTTPException(status_code=409, detail="Evaluation already running")
    background_tasks.add_task(run_script, "evaluate.py", "evaluate")
    return {"message": "Evaluation started", "status_endpoint": "/api/task-status"}


@app.post("/api/energy/track")
async def trigger_energy_tracking(background_tasks: BackgroundTasks):
    """Trigger energy tracking in the background."""
    if task_status["energy"]["running"]:
        raise HTTPException(status_code=409, detail="Energy tracking already running")
    background_tasks.add_task(run_script, "energy.py", "energy")
    return {"message": "Energy tracking started", "status_endpoint": "/api/task-status"}


@app.get("/api/task-status")
async def get_task_status():
    """Check the status of background tasks."""
    return task_status


# ============================================================
# Preloaded Model Compression — 15 curated models
# ============================================================

class PreloadedCompressRequest(BaseModel):
    """Request body for preloaded model compression."""
    model_name: str
    method: str = "pruning"
    dataset: str = "CIFAR10"
    fine_tune_epochs: int = 5


# Track preloaded compression status
preloaded_task = {
    "running": False,
    "step": "",          # current step key: loading_model, loading_data, compressing, energy_tracking, evaluating, complete
    "detail": "",        # human-readable detail text
    "progress": "",      # same as detail (kept for backwards compatibility)
    "result": None,
    "error": None,
}

# Steps in order — frontend uses this list to render the progress bar
COMPRESSION_STEPS = [
    {"key": "loading_model", "label": "Loading Model"},
    {"key": "loading_data", "label": "Preparing Dataset"},
    {"key": "compressing", "label": "Compressing"},
    {"key": "energy_tracking", "label": "Energy Tracking"},
    {"key": "evaluating", "label": "Final Evaluation"},
    {"key": "complete", "label": "Complete"},
]


@app.get("/api/preloaded-models")
async def get_preloaded_models():
    """List all 15 available pretrained models."""
    sys.path.insert(0, BACKEND_DIR)
    from compress import PRELOADED_MODELS
    models = []
    for key, cfg in PRELOADED_MODELS.items():
        models.append({
            "key": key,
            "name": cfg["name"],
            "params": cfg["params"],
            "input_size": cfg["input_size"],
            "dataset": cfg["dataset"],
        })
    return {"models": models}


@app.post("/api/compress/preloaded")
async def compress_preloaded(req: PreloadedCompressRequest):
    """
    Start compressing a preloaded pretrained model in the background.
    Returns 202 immediately. Poll /api/compress/preloaded/status for progress.

    Body JSON:
        {
            "model_name": "resnet18",
            "method": "pruning",
            "dataset": "CIFAR10",
            "fine_tune_epochs": 5
        }
    """
    if preloaded_task["running"]:
        raise HTTPException(
            status_code=409,
            detail="A compression task is already running."
        )

    # Validate model name
    sys.path.insert(0, BACKEND_DIR)
    from compress import PRELOADED_MODELS
    model_key = req.model_name.lower().strip()
    if model_key not in PRELOADED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model_name}'. "
                   f"Available: {list(PRELOADED_MODELS.keys())}"
        )

    # Validate method
    valid_methods = ['pruning', 'quantization', 'hybrid', 'kd']
    method = req.method.lower().strip()
    if method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method '{req.method}'. Choose from: {valid_methods}"
        )

    # Validate dataset
    valid_datasets = ['CIFAR10', 'CIFAR100']
    dataset = req.dataset.upper()
    if dataset not in valid_datasets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset '{req.dataset}'. Choose from: {valid_datasets}"
        )

    # Reset task state
    preloaded_task["running"] = True
    preloaded_task["step"] = "loading_model"
    preloaded_task["detail"] = f"Loading {req.model_name}..."
    preloaded_task["progress"] = f"Loading {req.model_name}..."
    preloaded_task["error"] = None
    preloaded_task["result"] = None

    def _run_in_thread():
        try:
            from compress import run_compression

            def progress_cb(step, detail=''):
                preloaded_task["step"] = step
                preloaded_task["detail"] = detail
                preloaded_task["progress"] = detail

            result = run_compression(
                model_name=model_key,
                method=method,
                dataset=dataset,
                fine_tune_epochs=req.fine_tune_epochs,
                progress_cb=progress_cb,
            )

            # Remove internal path info
            result.pop("saved_path", None)

            preloaded_task["result"] = result
            preloaded_task["step"] = "complete"
            preloaded_task["detail"] = "Done!"
            preloaded_task["progress"] = "Complete"

            # Save result to last-result file
            result_path = os.path.join(RESULTS_DIR, "preloaded_compression_result.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            # Append to compression history (keyed by model)
            _save_to_compression_history(model_key, result)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            preloaded_task["error"] = error_msg
            preloaded_task["step"] = "error"
            preloaded_task["detail"] = error_msg
            traceback.print_exc()
        finally:
            preloaded_task["running"] = False

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()

    return JSONResponse(
        status_code=202,
        content={
            "message": "Compression started",
            "status_endpoint": "/api/compress/preloaded/status",
        }
    )


@app.get("/api/compress/preloaded/status")
async def preloaded_compress_status():
    """Check the status of the preloaded compression task, including step-by-step progress."""
    return {
        **preloaded_task,
        "steps": COMPRESSION_STEPS,
    }


@app.get("/api/compress/preloaded/result")
async def preloaded_compress_result():
    """Get the last preloaded compression result."""
    result = load_json("preloaded_compression_result.json")
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No preloaded compression result found yet."
        )
    return result


# ============================================================
# Dynamic Model Compression — upload any model
# ============================================================

# Track dynamic compression status
dynamic_task = {
    "running": False,
    "progress": "",
    "result": None,
    "error": None,
}


@app.post("/api/compress/dynamic")
async def dynamic_compress(
    model_file: UploadFile = File(...),
    strategy: str = Form("pruning"),
    dataset: str = Form("CIFAR10"),
    fine_tune_epochs: int = Form(5),
    architecture: str = Form("auto"),
    num_classes: int = Form(10),
):
    """
    Upload any PyTorch model (.pt/.pth) and apply dynamic compression.

    The architecture is auto-detected from the uploaded file.
    You can optionally specify it if auto-detection fails.

    - model_file: The PyTorch model file (state_dict or full model)
    - strategy: pruning | quantization | hybrid | kd
    - dataset: CIFAR10 | CIFAR100
    - fine_tune_epochs: Number of fine-tuning epochs (default: 5)
    - architecture: auto (default) | resnet18 | resnet34 | resnet50 | mobilenet_v2 | vgg16 | compact_student
    - num_classes: Number of output classes (default: 10)

    Returns JSON with compression metrics.
    """
    if dynamic_task["running"]:
        raise HTTPException(
            status_code=409,
            detail="A dynamic compression task is already running."
        )

    # Validate file extension
    filename = model_file.filename or "uploaded_model.pth"
    if not filename.endswith(('.pt', '.pth')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a .pt or .pth file."
        )

    # Validate strategy
    valid_strategies = ['pruning', 'quantization', 'hybrid', 'kd']
    if strategy.lower() not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy '{strategy}'. Choose from: {valid_strategies}"
        )

    # Validate dataset
    valid_datasets = ['CIFAR10', 'CIFAR100']
    if dataset.upper() not in valid_datasets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset '{dataset}'. Choose from: {valid_datasets}"
        )

    # Save uploaded file
    upload_path = os.path.join(UPLOADS_DIR, filename)
    try:
        with open(upload_path, "wb") as f:
            content = await model_file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Run compression synchronously (can take minutes for KD)
    dynamic_task["running"] = True
    dynamic_task["progress"] = f"Compressing with {strategy}..."
    dynamic_task["error"] = None
    dynamic_task["result"] = None

    try:
        # Import the dynamic compress function
        sys.path.insert(0, BACKEND_DIR)
        from compress import compress_dynamic

        result = compress_dynamic(
            model_path=upload_path,
            strategy=strategy.lower(),
            dataset=dataset.upper(),
            fine_tune_epochs=fine_tune_epochs,
            architecture=architecture.lower().strip(),
            num_classes=num_classes,
        )

        # Remove internal path info from result before returning
        result.pop("saved_path", None)

        dynamic_task["result"] = result
        dynamic_task["progress"] = "Complete"

        # Save result to results dir
        result_path = os.path.join(RESULTS_DIR, "dynamic_compression_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        dynamic_task["error"] = error_msg
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        dynamic_task["running"] = False


@app.get("/api/compress/dynamic/status")
async def dynamic_compress_status():
    """Check the status of the dynamic compression task."""
    return dynamic_task


@app.get("/api/compress/dynamic/result")
async def dynamic_compress_result():
    """Get the last dynamic compression result."""
    result = load_json("dynamic_compression_result.json")
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No dynamic compression result found. Upload a model first."
        )
    return result


@app.get("/api/uploads")
async def list_uploads():
    """List all uploaded model files."""
    uploads = []
    if os.path.exists(UPLOADS_DIR):
        for f in sorted(os.listdir(UPLOADS_DIR)):
            fpath = os.path.join(UPLOADS_DIR, f)
            if os.path.isfile(fpath) and f.endswith(('.pt', '.pth', '.pth.gz')):
                uploads.append({
                    "filename": f,
                    "size_MB": round(os.path.getsize(fpath) / 1e6, 2),
                })
    return {"uploads": uploads}


# ============================================================
# Metrics summary endpoint (for frontend dashboard)
# ============================================================

@app.get("/api/dashboard")
async def dashboard():
    """
    Aggregated dashboard data — combines results, energy, evaluation
    into one response for the frontend to render.
    """
    summary = load_json("compression_summary.json") or {}
    energy = load_json("energy_report.json") or {}
    evaluation = load_json("evaluation_report.json") or {}
    models = list_models()

    # Build strategy cards
    strategies = []
    baseline_data = summary.get("baseline", {})
    baseline_size = baseline_data.get("size_MB", 44.81)
    baseline_acc = baseline_data.get("accuracy", 0)

    strategy_names = {
        "baseline": "Baseline (ResNet18)",
        "pruning_compressed": "Pruning 70% + Gzip",
        "quantization_static": "Static Quantization INT8",
        "quantization_dynamic": "Dynamic Quantization",
        "kd_compact_student": "KD → Compact Student",
        "hybrid_student_quant": "Student + Quantization",
        "ultra_compact": "Ultra-Compact",
    }

    for key, display_name in strategy_names.items():
        if key not in summary:
            continue
        m = summary[key]
        size = m.get("size_MB", m.get("size_MB_compressed",
               m.get("size_MB_sparse", baseline_size)))
        strategies.append({
            "key": key,
            "name": display_name,
            "accuracy": m.get("accuracy", 0),
            "size_MB": size,
            "size_reduction": round(100 * (baseline_size - size) / baseline_size, 2)
                             if baseline_size > 0 else 0,
            "latency_ms": m.get("latency_ms", 0),
            "params": m.get("params", m.get("student_params",
                     m.get("total_params", 0))),
        })

    return {
        "strategies": strategies,
        "energy": energy.get("savings", {}),
        "models": models,
        "gpu_available": torch.cuda.is_available(),
        "task_status": task_status,
    }


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("Starting GreenAI API server...")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True,
                reload_excludes=["*.pth", "*.json", "*.csv"])
