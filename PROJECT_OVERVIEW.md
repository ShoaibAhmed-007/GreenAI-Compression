# PROJECT OVERVIEW

## Project Idea
- Build a Green AI pipeline that compares baseline CNN models against compressed versions.
- Track accuracy, model size, latency, energy use, and CO2 emissions across the full workflow.
- Provide a production-style API + dashboard for preparation, compression, history, and image-level comparison.

Code anchors:
- [backend/main.py](backend/main.py#L1)
- [backend/compress.py](backend/compress.py#L2379)
- [frontend/src/app/page.tsx](frontend/src/app/page.tsx#L19)

## Core Objectives
- Train/prepare strong baseline models for all supported architectures.
- Apply multiple compression methods while preserving acceptable performance.
- Compare compressed emissions against the correct model-specific baseline emissions.
- Keep results reproducible and queryable through normalized backend history endpoints.

Code anchors:
- [backend/train.py](backend/train.py#L234)
- [backend/prepare_models.py](backend/prepare_models.py#L40)
- [backend/main.py](backend/main.py#L1354)
- [backend/main.py](backend/main.py#L2726)

## Current Supported Baseline Models (11)
- resnet18
- resnet34
- mobilenet_v2
- efficientnet_b0
- efficientnet_b1
- densenet121
- densenet169
- squeezenet
- shufflenet_v2
- inception_v3
- googlenet

Code anchors:
- [backend/train.py](backend/train.py#L45)
- [backend/train.py](backend/train.py#L59)
- [backend/compress.py](backend/compress.py#L2247)

## End-to-End Workflow (Input -> Output)
1. Baseline training or preparation
- Baselines can be produced through full training with train.py or fast preparation through prepare_models.py.
- Training writes per-model results and merges into training_results_all without wiping previous entries.

2. Compression request
- Frontend triggers curated compression through /api/compress/preloaded.
- Uploaded custom models are handled through /api/compress/dynamic.
- Legacy /api/compress endpoint is intentionally deprecated (410) to avoid old ambiguous flow.

3. Compression execution
- Backend loads model and data loaders, then applies selected method (pruning, quantization, hybrid, kd).
- Metrics include accuracy, size, latency, and emissions/energy fields.

4. Emissions normalization and baseline integrity
- Backend normalizes every compression result before saving/serving.
- Baseline training emissions are preserved as immutable reference fields.
- Projected compressed CO2 from size ratio is used as fallback/diagnostic only when measured compressed CO2 is absent.

5. Persistence
- Compression history is stored in results/compression_history.json keyed by model.
- APIs serve normalized, model-key-consistent payloads for dashboard and compare views.

6. Visualization and image-level comparison
- Dashboard consumes baselines + normalized compression history.
- Model comparison endpoint runs baseline vs compressed inference on local Assets images and reports quality/performance/emissions deltas.

Code anchors:
- [backend/train.py](backend/train.py#L475)
- [backend/main.py](backend/main.py#L2863)
- [backend/main.py](backend/main.py#L2953)
- [backend/main.py](backend/main.py#L3099)
- [backend/main.py](backend/main.py#L1324)
- [backend/main.py](backend/main.py#L1354)
- [backend/main.py](backend/main.py#L2638)
- [backend/main.py](backend/main.py#L1938)

## Compression Methods in Active Flow
- Pruning: iterative unstructured pruning plus fine-tuning and best-artifact saving.
- Quantization: QAT/static paths with environment-aware fallback behavior.
- Hybrid: prune + retrain + quantization pipeline.
- KD: teacher to compact student distillation.

Code anchors:
- [backend/compress.py](backend/compress.py#L1186)
- [backend/compress.py](backend/compress.py#L1363)
- [backend/compress.py](backend/compress.py#L1580)
- [backend/compress.py](backend/compress.py#L1894)

## API Surface (Operational)
- Baselines and preparation
- GET /api/baselines
- POST /api/prepare
- GET /api/prepare/status

- Compression
- POST /api/compress/preloaded
- GET /api/compress/preloaded/status
- POST /api/compress/dynamic
- GET /api/compress/dynamic/status
- POST /api/compress (deprecated)

- Comparison and dashboard
- GET /api/compression-history
- GET /api/compare
- GET /api/dashboard
- GET /api/model-comparison/options
- POST /api/model-comparison/compare

Code anchors:
- [backend/main.py](backend/main.py#L1653)
- [backend/main.py](backend/main.py#L2536)
- [backend/main.py](backend/main.py#L2628)
- [backend/main.py](backend/main.py#L2953)
- [backend/main.py](backend/main.py#L3065)
- [backend/main.py](backend/main.py#L3099)
- [backend/main.py](backend/main.py#L3205)
- [backend/main.py](backend/main.py#L2863)
- [backend/main.py](backend/main.py#L2638)
- [backend/main.py](backend/main.py#L2726)
- [backend/main.py](backend/main.py#L3242)
- [backend/main.py](backend/main.py#L1721)
- [backend/main.py](backend/main.py#L1938)

## Data and Artifact Flow
- Inputs
- CIFAR datasets via torchvision loaders.
- Optional local Assets images for inference comparison.
- Optional uploaded model artifacts for dynamic compression.

- Model artifacts
- Baseline checkpoints under models and models/pretrained_baselines.
- Uploaded models under models/uploads.
- Compressed artifacts under models/compressed and related upload subpaths.

- Results artifacts
- results/*_training_result.json per baseline model.
- results/training_results_all.json aggregated baseline summary.
- results/compression_history.json normalized history source of truth.
- results/preloaded_compression_result.json and results/dynamic_compression_result.json latest-run snapshots.

Code anchors:
- [backend/train.py](backend/train.py#L374)
- [backend/train.py](backend/train.py#L475)
- [backend/main.py](backend/main.py#L57)
- [backend/main.py](backend/main.py#L2638)
- [frontend/src/lib/api.ts](frontend/src/lib/api.ts#L352)

## Frontend Integration Summary
- Home dashboard orchestrates baselines, history, charts, and compression dialog flows.
- API layer normalizes dynamic results and maps them into chart/table strategy structures.
- Energy section prefers immutable baseline training CO2 fields when available.

Code anchors:
- [frontend/src/app/page.tsx](frontend/src/app/page.tsx#L19)
- [frontend/src/app/page.tsx](frontend/src/app/page.tsx#L296)
- [frontend/src/lib/api.ts](frontend/src/lib/api.ts#L100)
- [frontend/src/lib/api.ts](frontend/src/lib/api.ts#L856)
- [frontend/src/components/EnergySection.tsx](frontend/src/components/EnergySection.tsx#L59)

## Important Logic Rules (Current)
- Baseline training emissions and compressed emissions are semantically separate fields.
- Result normalization occurs before persistence and when reading history.
- Projected compressed emissions are diagnostic/fallback values, not hard overrides of measured compressed totals.
- Compare and dashboard flows are history-first and multi-model aware.

Code anchors:
- [backend/main.py](backend/main.py#L1324)
- [backend/main.py](backend/main.py#L1354)
- [backend/main.py](backend/main.py#L1416)
- [backend/main.py](backend/main.py#L2638)
- [backend/main.py](backend/main.py#L3242)

## Known Limitations
- Some legacy result files still exist for backward compatibility and fallback behavior.
- Global in-memory task states serialize work per task type (prepare/preloaded/dynamic).
- Dynamic compression can be long-running depending on model size and method.
- Existing old history entries may still be mixed on disk until re-saved/migrated, though API normalization now mitigates this.

Code anchors:
- [backend/main.py](backend/main.py#L2726)
- [backend/main.py](backend/main.py#L2897)
- [backend/main.py](backend/main.py#L3205)
- [backend/main.py](backend/main.py#L2638)
