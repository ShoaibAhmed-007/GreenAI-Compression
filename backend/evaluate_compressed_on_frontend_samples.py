"""Evaluate all compressed model options on frontend sample images.

Uses the same backend comparison path as frontend model-comparison batch API
to validate compressed-model behavior after preprocessing alignment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from main import (
    BACKEND_DIR,
    ModelComparisonBatchRequest,
    _build_comparison_sample_images,
    _build_model_comparison_options,
    _find_compression_history_entry,
    compare_models_on_batch,
)


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _first_numeric(payload: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key not in payload:
            continue
        value = _to_float(payload.get(key))
        if value is not None:
            return value
    return None


def _json_dump(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_strategy_result_metrics(model_key: str, strategy: str) -> Dict[str, Any]:
    result_path = os.path.abspath(
        os.path.join(BACKEND_DIR, "..", "results", f"{model_key}_{strategy}_compression_result.json")
    )
    if not os.path.exists(result_path):
        return {}

    try:
        with open(result_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    return {
        "source_file": result_path,
        "expected_baseline_accuracy_percent": _first_numeric(
            payload,
            ["baseline_accuracy"],
        ),
        "expected_compressed_accuracy_percent": _first_numeric(
            payload,
            ["compressed_accuracy", "accuracy", "accuracy_top1"],
        ),
    }


async def _evaluate_option(
    option: Dict[str, Any],
    baseline_ready: Dict[str, Dict[str, Any]],
    sample_ids: List[int],
    batch_size: int,
    enable_tta: bool,
) -> Dict[str, Any]:
    model_key = str(option.get("model_key") or "").strip()
    compressed_key = str(option.get("key") or "").strip()
    strategy = str(option.get("strategy") or "").strip()

    if model_key not in baseline_ready:
        raise RuntimeError(f"No ready baseline available for compressed option '{compressed_key}'.")

    req = ModelComparisonBatchRequest(
        sample_ids=sample_ids,
        baseline_model_key=model_key,
        compressed_model_key=compressed_key,
        batch_size=int(batch_size),
        enable_tta=bool(enable_tta),
    )

    batch_result = await compare_models_on_batch(req)

    effective_model_key = str(batch_result.get("compressed_model_key") or model_key).strip() if isinstance(batch_result, dict) else model_key
    effective_strategy = str(batch_result.get("compressed_strategy") or strategy).strip() if isinstance(batch_result, dict) else strategy
    fallback_applied = bool(batch_result.get("fallback_applied", False)) if isinstance(batch_result, dict) else False

    expected_model_key = effective_model_key if fallback_applied else model_key
    expected_strategy = effective_strategy if fallback_applied else strategy

    result_metrics = _read_strategy_result_metrics(expected_model_key, expected_strategy)
    expected_baseline_acc = _to_float(result_metrics.get("expected_baseline_accuracy_percent"))
    expected_compressed_acc = _to_float(result_metrics.get("expected_compressed_accuracy_percent"))
    expected_accuracy_source = "result_file"
    expected_accuracy_source_file = result_metrics.get("source_file")

    if expected_baseline_acc is None or expected_compressed_acc is None:
        history_entry = _find_compression_history_entry(expected_model_key, expected_strategy) or {}
        if expected_baseline_acc is None:
            expected_baseline_acc = _first_numeric(history_entry, ["baseline_accuracy"])
        if expected_compressed_acc is None:
            expected_compressed_acc = _first_numeric(history_entry, ["compressed_accuracy", "accuracy", "accuracy_top1"])

        if expected_accuracy_source_file:
            expected_accuracy_source = "result_file+history_fallback"
        else:
            expected_accuracy_source = "history"

    summary = batch_result.get("summary", {}) if isinstance(batch_result, dict) else {}
    observed_baseline_acc = _to_float(summary.get("baseline_accuracy_percent"))
    observed_compressed_acc = _to_float(summary.get("compressed_accuracy_percent"))

    baseline_delta = None
    if observed_baseline_acc is not None and expected_baseline_acc is not None:
        baseline_delta = round(observed_baseline_acc - expected_baseline_acc, 2)

    compressed_delta = None
    if observed_compressed_acc is not None and expected_compressed_acc is not None:
        compressed_delta = round(observed_compressed_acc - expected_compressed_acc, 2)

    diagnosis = {
        "expected_baseline_accuracy_percent": expected_baseline_acc,
        "expected_compressed_accuracy_percent": expected_compressed_acc,
        "expected_accuracy_source": expected_accuracy_source,
        "expected_accuracy_source_file": expected_accuracy_source_file,
        "observed_baseline_accuracy_percent": observed_baseline_acc,
        "observed_compressed_accuracy_percent": observed_compressed_acc,
        "baseline_accuracy_delta_percent": baseline_delta,
        "compressed_accuracy_delta_percent": compressed_delta,
    }

    return {
        "generated_at": _utc_now_iso(),
        "status": "ok",
        "evaluation_name": "frontend_sample_images_compressed_eval",
        "model_key": model_key,
        "compressed_model_key": compressed_key,
        "strategy": strategy,
        "effective_model_key": effective_model_key,
        "effective_strategy": effective_strategy,
        "fallback_applied": fallback_applied,
        "expected_model_key": expected_model_key,
        "expected_strategy": expected_strategy,
        "strategy_label": option.get("strategy_label"),
        "label": option.get("label"),
        "sample_ids": sample_ids,
        "sample_count": len(sample_ids),
        "batch_size": int(batch_size),
        "tta_enabled": bool(enable_tta),
        "diagnosis": diagnosis,
        "batch_result": batch_result,
    }


async def run(
    sample_limit: int,
    batch_size: int,
    enable_tta: bool,
    output_dir: str,
    prefix: str,
) -> Dict[str, Any]:
    options = _build_model_comparison_options()
    baseline_models = options.get("baseline_models", [])
    compressed_models = options.get("compressed_models", [])

    baseline_ready = {
        str(item.get("key")): item
        for item in baseline_models
        if str(item.get("status", "ready")).lower() == "ready"
    }

    selected_samples = _build_comparison_sample_images(limit=int(sample_limit))
    sample_ids = [int(item.get("id")) for item in selected_samples]
    sample_paths = [str(item.get("source_path")) for item in selected_samples]

    report_items: List[Dict[str, Any]] = []

    for option in compressed_models:
        model_key = str(option.get("model_key") or "").strip()
        compressed_key = str(option.get("key") or "").strip()
        safe_key = compressed_key.replace("::", "__")
        details_name = f"{prefix}_{safe_key}_details.json"
        details_path = os.path.join(output_dir, details_name)

        print(f"[CompressedEval] {compressed_key}: evaluating {len(sample_ids)} samples...")

        try:
            payload = await _evaluate_option(
                option=option,
                baseline_ready=baseline_ready,
                sample_ids=sample_ids,
                batch_size=batch_size,
                enable_tta=enable_tta,
            )
            _json_dump(details_path, payload)

            diagnosis = payload.get("diagnosis", {})
            report_items.append(
                {
                    "compressed_model_key": compressed_key,
                    "model_key": model_key,
                    "strategy": option.get("strategy"),
                    "effective_model_key": payload.get("effective_model_key"),
                    "effective_strategy": payload.get("effective_strategy"),
                    "fallback_applied": bool(payload.get("fallback_applied", False)),
                    "expected_model_key": payload.get("expected_model_key"),
                    "expected_strategy": payload.get("expected_strategy"),
                    "status": "ok",
                    "details_file": os.path.basename(details_path),
                    "observed_baseline_accuracy_percent": diagnosis.get("observed_baseline_accuracy_percent"),
                    "observed_compressed_accuracy_percent": diagnosis.get("observed_compressed_accuracy_percent"),
                    "expected_compressed_accuracy_percent": diagnosis.get("expected_compressed_accuracy_percent"),
                    "compressed_accuracy_delta_percent": diagnosis.get("compressed_accuracy_delta_percent"),
                }
            )
        except HTTPException as http_exc:
            error_payload = {
                "generated_at": _utc_now_iso(),
                "status": "error",
                "evaluation_name": "frontend_sample_images_compressed_eval",
                "compressed_model_key": compressed_key,
                "model_key": model_key,
                "strategy": option.get("strategy"),
                "error": str(http_exc.detail),
            }
            _json_dump(details_path, error_payload)
            report_items.append(
                {
                    "compressed_model_key": compressed_key,
                    "model_key": model_key,
                    "strategy": option.get("strategy"),
                    "status": "error",
                    "details_file": os.path.basename(details_path),
                    "error": str(http_exc.detail),
                }
            )
        except Exception as exc:
            error_payload = {
                "generated_at": _utc_now_iso(),
                "status": "error",
                "evaluation_name": "frontend_sample_images_compressed_eval",
                "compressed_model_key": compressed_key,
                "model_key": model_key,
                "strategy": option.get("strategy"),
                "error": str(exc),
            }
            _json_dump(details_path, error_payload)
            report_items.append(
                {
                    "compressed_model_key": compressed_key,
                    "model_key": model_key,
                    "strategy": option.get("strategy"),
                    "status": "error",
                    "details_file": os.path.basename(details_path),
                    "error": str(exc),
                }
            )

    total = len(report_items)
    successful = sum(1 for item in report_items if item.get("status") == "ok")
    failed = total - successful

    summary_payload = {
        "generated_at": _utc_now_iso(),
        "evaluation_name": "frontend_sample_images_compressed_eval",
        "backend_dir": BACKEND_DIR,
        "sample_source": "/api/model-comparison/sample-images",
        "sample_limit_requested": int(sample_limit),
        "sample_ids": sample_ids,
        "sample_source_paths": sample_paths,
        "tta_enabled": bool(enable_tta),
        "batch_size": int(batch_size),
        "total_compressed_options": total,
        "successful_options": successful,
        "failed_options": failed,
        "options": report_items,
    }

    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    _json_dump(summary_path, summary_payload)

    return {
        "summary_path": summary_path,
        "output_dir": output_dir,
        "total_compressed_options": total,
        "successful_options": successful,
        "failed_options": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all compressed model options on frontend sample images and compare to recorded accuracy.",
    )
    parser.add_argument("--sample-limit", type=int, default=10, help="Number of frontend sample images to evaluate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for compare-batch requests.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.abspath(os.path.join(BACKEND_DIR, "..", "results")),
        help="Directory for output detail files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="compressed_frontend_sample10",
        help="Filename prefix for generated outputs.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable test-time augmentation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outcome = asyncio.run(
        run(
            sample_limit=max(1, int(args.sample_limit)),
            batch_size=max(1, min(int(args.batch_size), 64)),
            enable_tta=not bool(args.no_tta),
            output_dir=os.path.abspath(args.output_dir),
            prefix=str(args.prefix).strip() or "compressed_frontend_sample10",
        )
    )
    print("[CompressedEval] Completed")
    print(json.dumps(outcome, indent=2))
