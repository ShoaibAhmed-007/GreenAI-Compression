# PROJECT OVERVIEW

## Project Idea
- Build a Green AI workflow that compares baseline vs compressed CNN models.
- Measure model quality, size, latency, energy, and CO2 impact.
- Provide a web dashboard for compression runs, history, and image-level model comparison.

Code anchors:
- [backend/main.py](backend/main.py#L1006)
- [backend/compress.py](backend/compress.py#L2379)
- [frontend/src/app/page.tsx](frontend/src/app/page.tsx#L1)

## Objectives
- Reduce model size while keeping acceptable accuracy.
- Track carbon/energy impact during training and inference.
- Support both curated pretrained models and uploaded custom PyTorch models.
- Give developers a fast feedback loop through API + dashboard views.

Code anchors:
- [backend/compress.py](backend/compress.py#L1186)
- [backend/compress.py](backend/compress.py#L1363)
- [backend/compress.py](backend/compress.py#L1580)
- [backend/compress.py](backend/compress.py#L1894)

## Complete Workflow (Input -> Output)
1. Baseline preparation
- User triggers model preparation from dashboard.
- Backend loads pretrained torchvision models, adapts heads to CIFAR classes, runs short head-only tuning, and saves baseline weights/metrics.

2. Compression request
- User selects model + method + dataset + epochs in UI.
- Frontend calls preloaded compression API.
- Backend runs compression in a background thread and exposes progress steps.

3. Compression execution
- Backend loads baseline model and CIFAR loaders.
- Strategy applied: pruning, quantization, hybrid, or KD.
- Strategy output includes accuracy, size, latency, emissions, energy, and metadata.

4. Emissions and fair comparison
- CodeCarbon tracking is applied for training/inference workloads.
- Fair benchmark metrics are computed baseline vs compressed.
- Compressed CO2 is projected from baseline CO2 by size ratio when needed.

5. Persistence
- Backend writes result JSON files into results directory and appends compression history.
- Frontend polls status and stores normalized results in localStorage.

6. Visualization and comparison
- Dashboard renders cards, charts, tables, and energy blocks.
- Model-comparison page runs baseline vs compressed inference on selected Assets images (single or batch).

Code anchors:
- [backend/main.py](backend/main.py#L1666)
- [backend/main.py](backend/main.py#L2014)
- [backend/main.py](backend/main.py#L2125)
- [backend/main.py](backend/main.py#L2159)
- [frontend/src/components/ModelDashboard.tsx](frontend/src/components/ModelDashboard.tsx#L166)
- [frontend/src/lib/api.ts](frontend/src/lib/api.ts#L481)

## Working Methodology
- Baseline models
- Curated catalog in PRELOADED_MODELS with per-model input size and metadata.
- Baseline weights saved under models/pretrained_baselines and baseline metrics stored in results JSON.

- Compression methods used in API flow
- Pruning: unstructured L1 pruning + fine-tuning + smallest-artifact save.
- Quantization: QAT on CPU when possible, with dynamic quantization fallback.
- Hybrid: prune + fine-tune + quantization.
- KD: teacher to CompactStudent distillation with accuracy guards.

- Emissions tracking
- Uses CodeCarbon tracker wrappers.
- Tracks training and inference emissions/energy.
- Produces fair baseline/compressed benchmark fields and sanity warnings.

- Image comparison methodology
- Images loaded from local Assets.
- Supports both class-folder structure and flat files with CIFAR class tokens in filenames.
- Uses CIFAR normalization and model-specific input adaptation for inference.

Code anchors:
- [backend/compress.py](backend/compress.py#L2247)
- [backend/compress.py](backend/compress.py#L615)
- [backend/main.py](backend/main.py#L267)
- [backend/main.py](backend/main.py#L1190)

## System Flow (Logical Flow)
- UI -> API -> ML Pipeline -> Results -> UI

- Detailed flow
1. Dashboard requests data from /api/dashboard, /api/baselines, /api/compression-history.
2. User action starts /api/prepare or /api/compress/preloaded or /api/compress/dynamic.
3. Backend executes model/data/compression/emissions/evaluation logic.
4. Backend writes/updates JSON artifacts in results and model files in models.
5. Frontend polls status endpoints and merges results into localStorage-backed history.
6. Charts/tables/model comparison views render normalized metrics.

Code anchors:
- [backend/main.py](backend/main.py#L2300)
- [frontend/src/app/page.tsx](frontend/src/app/page.tsx#L34)
- [frontend/src/lib/api.ts](frontend/src/lib/api.ts#L572)

## Key Features
- FastAPI backend with health, prepare, compress, evaluate, energy, and dashboard APIs.
- Preloaded model compression with step-by-step status tracking.
- Dynamic compression for uploaded .pt/.pth models with architecture auto-detection.
- CIFAR-based model comparison page with single and batch inference on Assets images.
- CO2 and energy display with reduction calculations and sanity-warning surfacing.
- Persistent frontend compression history (localStorage) with migration from legacy key.
- Legacy script endpoint support for full pipeline tasks.

Code anchors:
- [backend/main.py](backend/main.py#L1006)
- [backend/main.py](backend/main.py#L2014)
- [backend/main.py](backend/main.py#L2159)
- [backend/main.py](backend/main.py#L1494)
- [frontend/src/components/EnergySection.tsx](frontend/src/components/EnergySection.tsx#L1)
- [frontend/src/lib/api.ts](frontend/src/lib/api.ts#L514)

## Tech Stack
- Frontend
- Next.js 14, React 18, TypeScript, Tailwind CSS, Recharts.

- Backend
- FastAPI, Uvicorn, Pydantic, Python threading/subprocess orchestration.

- ML and data
- PyTorch, torchvision, CIFAR-10/CIFAR-100 loaders, PIL.

- Carbon/energy
- CodeCarbon (online/offline tracker modes).

- Storage/artifacts
- JSON files in results, model checkpoints in models, browser localStorage for UI history.

Code anchors:
- [frontend/package.json](frontend/package.json#L1)
- [backend/main.py](backend/main.py#L1)
- [backend/compress.py](backend/compress.py#L1)

## Data Flow
- Training/compression inputs
- CIFAR dataset loaded via torchvision loaders.
- Uploaded model files stored in models/uploads.
- Optional image samples loaded from Assets for model comparison.

- Intermediate artifacts
- Baseline checkpoints: models/pretrained_baselines.
- Compressed artifacts: models/compressed or models/uploads/compressed.

- Output metrics
- Compression outputs: results/compression_history.json, results/preloaded_compression_result.json, results/dynamic_compression_result.json.
- Evaluation outputs: results/evaluation_report.json.
- Energy outputs: results/energy_report.json and emissions CSV files.

- Frontend consumption
- API responses normalized in frontend lib layer.
- Saved results persisted to localStorage and reused for charts/tables after refresh.

Code anchors:
- [backend/main.py](backend/main.py#L57)
- [backend/compress.py](backend/compress.py#L2115)
- [frontend/src/lib/api.ts](frontend/src/lib/api.ts#L87)

## Important Logic / Assumptions
- Quantized models are executed on CPU paths in multiple flows.
- Input image size is model-dependent; loaders auto-adjust batch size by input size.
- Compression accuracy guards are used to prevent excessive drop during iterative steps.
- CO2 reduction can be projected from model-size ratio for consistency checks.
- Assets image labels assume CIFAR-10 class names, aliases, or filename tokens.
- Dashboard favors latest deduped result per model-strategy key in local storage.

Code anchors:
- [backend/compress.py](backend/compress.py#L1406)
- [backend/compress.py](backend/compress.py#L1116)
- [backend/compress.py](backend/compress.py#L802)
- [backend/compress.py](backend/compress.py#L578)
- [backend/main.py](backend/main.py#L243)
- [frontend/src/lib/api.ts](frontend/src/lib/api.ts#L529)

## Known Issues / Limitations
- Model-count mismatch in comments/docs
- Several backend comments/docs say 15 preloaded models, but active PRELOADED_MODELS catalog currently contains 11.

- Dynamic compression endpoint is synchronous
- /api/compress/dynamic handles long-running work in-request (not background), which can cause long waits/timeouts for heavy runs.

- Single-task concurrency per pipeline type
- Global task state blocks concurrent prepare/preloaded/dynamic runs of the same type.

- Results-file dependency
- Many APIs return 404 until corresponding scripts or compression runs generate expected JSON files.

- CO2 numbers may be projected, not always directly measured
- Comparison logic can derive compressed CO2 from baseline size ratio; this improves consistency but is an assumption.

- Quantization fallback behavior
- QAT/INT8 conversion may fallback to dynamic quantization in some environments, so output strategy internals vary by runtime backend support.

- Assets image source is local-folder based
- Model-comparison uses local Assets files; there is no dedicated API endpoint for direct image upload.

- Legacy/stale UI paths exist
- Some older frontend components include fallback model lists beyond active backend catalog and are not the primary dashboard flow.

Code anchors:
- [backend/main.py](backend/main.py#L1018)
- [backend/compress.py](backend/compress.py#L2247)
- [backend/main.py](backend/main.py#L2159)
- [backend/main.py](backend/main.py#L1674)
- [backend/main.py](backend/main.py#L1810)
- [backend/main.py](backend/main.py#L1398)
- [backend/compress.py](backend/compress.py#L1471)
- [backend/main.py](backend/main.py#L1178)
- [frontend/src/components/ModelUpload.tsx](frontend/src/components/ModelUpload.tsx#L63)
