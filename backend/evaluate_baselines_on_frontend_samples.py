"""Evaluate baseline models on the frontend comparison sample images.

This script reuses backend main.py helpers so preprocessing and sample selection
match the frontend model-comparison workflow.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from compress import PRELOADED_MODELS
from main import (
    BACKEND_DIR,
    CIFAR10_CLASSES,
    _build_comparison_sample_images,
    _load_baseline_model_for_comparison,
    _predict_model_on_asset_images_batch,
    _resolve_assets_image_by_path,
    load_baseline_results,
)


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_dump(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _build_sample_bundle(limit: int) -> List[Dict[str, Any]]:
    samples = _build_comparison_sample_images(limit=limit)
    bundle: List[Dict[str, Any]] = []

    for sample in samples:
        source_path = str(sample.get("source_path") or "").strip()
        if not source_path:
            raise ValueError(f"Sample id={sample.get('id')} has no source_path.")

        image, inferred_label, resolved_path = _resolve_assets_image_by_path(source_path)
        label = str(sample.get("label") or "").strip().lower()
        if label not in CIFAR10_CLASSES:
            label = inferred_label if inferred_label in CIFAR10_CLASSES else ""

        bundle.append(
            {
                "sample_id": int(sample.get("id", -1)),
                "true_label": label or None,
                "class_index": sample.get("class_index"),
                "is_focus_class": bool(sample.get("is_focus_class", False)),
                "source_path": resolved_path,
                "image": image,
            }
        )

    return bundle


def _evaluate_single_model(
    model_key: str,
    model_name: str,
    sample_bundle: List[Dict[str, Any]],
    baseline_catalog: Dict[str, Dict[str, Any]],
    device: torch.device,
    enable_tta: bool,
    sample_limit: int,
) -> Dict[str, Any]:
    model = _load_baseline_model_for_comparison(model_key, device=device)
    images = [entry["image"] for entry in sample_bundle]

    predictions = _predict_model_on_asset_images_batch(
        model,
        images,
        device=device,
        enable_tta=enable_tta,
    )

    per_sample: List[Dict[str, Any]] = []
    known_count = 0
    correct_count = 0
    confidence_values: List[float] = []

    for sample, prediction in zip(sample_bundle, predictions):
        pred_class = str(prediction.get("predicted_class", ""))
        confidence = _safe_float(prediction.get("confidence"))
        if confidence is not None:
            confidence_values.append(confidence)

        true_label = sample.get("true_label")
        correct = None
        if isinstance(true_label, str) and true_label in CIFAR10_CLASSES:
            known_count += 1
            correct = pred_class.lower() == true_label.lower()
            if correct:
                correct_count += 1

        per_sample.append(
            {
                "sample_id": sample["sample_id"],
                "source_path": sample["source_path"],
                "true_label": true_label,
                "class_index": sample.get("class_index"),
                "is_focus_class": sample.get("is_focus_class", False),
                "predicted_class": pred_class,
                "predicted_index": prediction.get("predicted_index"),
                "confidence_percent": confidence,
                "correct": correct,
                "prediction": prediction,
            }
        )

    accuracy = round((100.0 * correct_count / known_count), 2) if known_count > 0 else None
    mean_confidence = (
        round(sum(confidence_values) / len(confidence_values), 2)
        if confidence_values
        else None
    )

    baseline_meta = baseline_catalog.get(model_key, {})

    return {
        "generated_at": _utc_now_iso(),
        "status": "ok",
        "evaluation_name": "frontend_sample_images_baseline_eval",
        "sample_source": "/api/model-comparison/sample-images",
        "sample_limit_requested": sample_limit,
        "model_key": model_key,
        "model_name": model_name,
        "device": str(device),
        "tta_enabled": bool(enable_tta),
        "samples_evaluated": len(per_sample),
        "known_label_count": known_count,
        "correct_count": correct_count,
        "accuracy_percent": accuracy,
        "mean_confidence_percent": mean_confidence,
        "baseline_catalog_metrics": baseline_meta,
        "results": per_sample,
    }


def run(
    sample_limit: int,
    enable_tta: bool,
    output_dir: str,
    prefix: str,
    force_cpu: bool,
) -> Dict[str, Any]:
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_catalog = load_baseline_results()
    sample_bundle = _build_sample_bundle(limit=sample_limit)

    summary_models: List[Dict[str, Any]] = []
    sample_ids = [entry["sample_id"] for entry in sample_bundle]
    sample_paths = [entry["source_path"] for entry in sample_bundle]

    model_items = list(PRELOADED_MODELS.items())

    for model_key, cfg in model_items:
        model_name = str(cfg.get("name") or model_key)
        details_filename = f"{prefix}_{model_key}_details.json"
        details_path = os.path.join(output_dir, details_filename)

        print(f"[BaselineEval] {model_key}: running on {len(sample_bundle)} samples...")
        try:
            detail_payload = _evaluate_single_model(
                model_key=model_key,
                model_name=model_name,
                sample_bundle=sample_bundle,
                baseline_catalog=baseline_catalog,
                device=device,
                enable_tta=enable_tta,
                sample_limit=sample_limit,
            )
            _json_dump(details_path, detail_payload)

            summary_models.append(
                {
                    "model_key": model_key,
                    "model_name": model_name,
                    "status": "ok",
                    "details_file": os.path.basename(details_path),
                    "samples_evaluated": detail_payload.get("samples_evaluated"),
                    "accuracy_percent": detail_payload.get("accuracy_percent"),
                    "mean_confidence_percent": detail_payload.get("mean_confidence_percent"),
                }
            )
        except Exception as exc:
            error_payload = {
                "generated_at": _utc_now_iso(),
                "status": "error",
                "evaluation_name": "frontend_sample_images_baseline_eval",
                "sample_source": "/api/model-comparison/sample-images",
                "sample_limit_requested": sample_limit,
                "model_key": model_key,
                "model_name": model_name,
                "device": str(device),
                "tta_enabled": bool(enable_tta),
                "error": str(exc),
            }
            _json_dump(details_path, error_payload)

            summary_models.append(
                {
                    "model_key": model_key,
                    "model_name": model_name,
                    "status": "error",
                    "details_file": os.path.basename(details_path),
                    "error": str(exc),
                }
            )
        finally:
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    successful = sum(1 for item in summary_models if item.get("status") == "ok")
    failed = len(summary_models) - successful

    summary_payload = {
        "generated_at": _utc_now_iso(),
        "evaluation_name": "frontend_sample_images_baseline_eval",
        "backend_dir": BACKEND_DIR,
        "device": str(device),
        "tta_enabled": bool(enable_tta),
        "sample_source": "/api/model-comparison/sample-images",
        "sample_limit_requested": sample_limit,
        "sample_ids": sample_ids,
        "sample_source_paths": sample_paths,
        "total_models": len(summary_models),
        "successful_models": successful,
        "failed_models": failed,
        "models": summary_models,
    }

    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    _json_dump(summary_path, summary_payload)

    return {
        "summary_path": summary_path,
        "output_dir": output_dir,
        "total_models": len(summary_models),
        "successful_models": successful,
        "failed_models": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all baseline models on frontend sample images and save detailed JSON files.",
    )
    parser.add_argument("--sample-limit", type=int, default=10, help="Number of frontend sample images to evaluate.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.abspath(os.path.join(BACKEND_DIR, "..", "results")),
        help="Directory for output detail files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="baseline_frontend_sample10",
        help="Filename prefix for generated JSON outputs.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable test-time augmentation (default keeps TTA enabled to match frontend default).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outcome = run(
        sample_limit=max(1, int(args.sample_limit)),
        enable_tta=not bool(args.no_tta),
        output_dir=os.path.abspath(args.output_dir),
        prefix=str(args.prefix).strip() or "baseline_frontend_sample10",
        force_cpu=bool(args.cpu),
    )
    print("[BaselineEval] Completed")
    print(json.dumps(outcome, indent=2))
