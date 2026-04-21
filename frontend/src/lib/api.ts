const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function readApiError(res: Response): Promise<string> {
  const fallback = `API error: ${res.status} ${res.statusText}`;
  try {
    const payload = await res.json();
    if (typeof payload?.detail === 'string' && payload.detail.trim() !== '') {
      return payload.detail;
    }
    if (typeof payload?.message === 'string' && payload.message.trim() !== '') {
      return payload.message;
    }
  } catch {
    // Ignore parse errors and keep fallback message.
  }
  return fallback;
}

export async function fetchAPI(endpoint: string) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    cache: 'no-store',
  });
  if (!res.ok) {
    throw new Error(await readApiError(res));
  }
  return res.json();
}

export async function postAPI(endpoint: string, body?: any) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    throw new Error(await readApiError(res));
  }
  return res.json();
}

export async function uploadModel(
  file: File,
  strategy: string,
  dataset: string = 'CIFAR10',
  fineTuneEpochs: number = 5
): Promise<DynamicResult> {
  const formData = new FormData();
  formData.append('model_file', file);
  formData.append('strategy', strategy);
  formData.append('dataset', dataset);
  formData.append('fine_tune_epochs', fineTuneEpochs.toString());

  const res = await fetch(`${API_BASE}/api/compress/dynamic`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  const data = await res.json();
  return normalizeDynamicResult(data as DynamicResult);
}

export async function compressPreloaded(
  modelName: string,
  method: string,
  dataset: string = 'CIFAR10',
  fineTuneEpochs: number = 5
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/compress/preloaded`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_name: modelName,
      method,
      dataset,
      fine_tune_epochs: fineTuneEpochs,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Compression failed: ${res.status}`);
  }
}

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : undefined;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function readPath(source: unknown, path: string): unknown {
  const segments = path.split('.');
  let current: unknown = source;

  for (const segment of segments) {
    if (!current || typeof current !== 'object') {
      return undefined;
    }

    const record = current as Record<string, unknown>;
    if (!(segment in record)) {
      return undefined;
    }

    current = record[segment];
  }

  return current;
}

function firstFiniteNumber(source: Record<string, unknown>, keys: string[]): number | undefined {
  for (const key of keys) {
    const rawValue = key.includes('.') ? readPath(source, key) : source[key];
    const numeric = toFiniteNumber(rawValue);
    if (numeric != null) {
      return numeric;
    }
  }
  return undefined;
}

export function normalizeDynamicResult(result: DynamicResult): DynamicResult {
  const raw = result as unknown as Record<string, unknown>;

  const baselineAccuracy = firstFiniteNumber(raw, [
    'baseline_accuracy',
    'baseline_accuracy_percent',
    'comparison.full_dataset_metrics.baseline_accuracy_percent',
    'observed_baseline_accuracy_percent',
  ]);
  const compressedAccuracy = firstFiniteNumber(raw, [
    'compressed_accuracy',
    'accuracy',
    'accuracy_top1',
    'top1_accuracy',
    'comparison.full_dataset_metrics.compressed_accuracy_percent',
    'observed_compressed_accuracy_percent',
    'expected_compressed_accuracy_percent',
  ]);

  const baselineSize = firstFiniteNumber(raw, [
    'baseline_size_MB',
    'original_size_MB',
    'size_MB_standard',
    'comparison.full_dataset_metrics.baseline_size_MB',
  ]);
  const compressedSize = firstFiniteNumber(raw, [
    'size_MB',
    'compressed_size_MB',
    'size_MB_quant',
    'size_MB_sparse',
    'size_MB_compressed',
    'comparison.full_dataset_metrics.compressed_size_MB',
  ]);

  const computedReduction =
    baselineSize != null && baselineSize > 0 && compressedSize != null
      ? ((baselineSize - compressedSize) / baselineSize) * 100
      : undefined;

  const compressionRatio = firstFiniteNumber(raw, ['compression_ratio', 'size_compression_ratio']);
  const ratioReduction =
    compressionRatio != null && compressionRatio > 0
      ? (1 - 1 / compressionRatio) * 100
      : undefined;

  const reduction = firstFiniteNumber(raw, [
    'size_reduction_percent',
    'size_reduction',
    'size_reduction_sparse_percent',
    'size_reduction_quant_percent',
    'comparison.full_dataset_metrics.size_reduction_percent',
  ]) ?? computedReduction ?? ratioReduction;

  const normalizedBaselineSize = baselineSize ?? compressedSize;
  const normalizedCompressedSize = compressedSize ?? baselineSize;

  const immutableBaselineTrainingCo2 = firstFiniteNumber(raw, [
    'baseline_training_co2_kg',
    'baseline_training_emissions_kg',
    'baseline_co2_kg',
  ]);
  const baselineTotalCo2 = immutableBaselineTrainingCo2 ?? firstFiniteNumber(raw, [
    'baseline_total_emissions_kg',
    'baseline_total_co2_kg',
    'comparison.full_dataset_metrics.baseline_co2_kg',
  ]);
  const fallbackCompressedCo2 = firstFiniteNumber(raw, [
    'compressed_total_emissions_kg',
    'compressed_co2_kg',
    'co2_kg',
    'emissions_kg',
    'inference_co2_kg',
    'inference_emissions_kg',
    'training_co2_kg',
    'training_emissions_kg',
    'comparison.full_dataset_metrics.compressed_co2_kg',
  ]);

  const projectedCompressedCo2 =
    baselineTotalCo2 != null &&
    baselineTotalCo2 > 0 &&
    normalizedBaselineSize != null &&
    normalizedBaselineSize > 0 &&
    normalizedCompressedSize != null
      ? baselineTotalCo2 * (normalizedCompressedSize / normalizedBaselineSize)
      : undefined;

  const normalizedCompressedCo2 = fallbackCompressedCo2 ?? projectedCompressedCo2;
  const normalizedReduction =
    baselineTotalCo2 != null &&
    baselineTotalCo2 > 0 &&
    normalizedCompressedCo2 != null
      ? ((baselineTotalCo2 - normalizedCompressedCo2) / baselineTotalCo2) * 100
      : toFiniteNumber(result.emissions_reduction_percent);

  const baselineTotalEnergy = firstFiniteNumber(raw, [
    'baseline_total_energy_kwh',
    'baseline_training_energy_kwh',
    'comparison.full_dataset_metrics.baseline_energy_kwh',
  ]);
  const compressedTotalEnergy = firstFiniteNumber(raw, [
    'compressed_total_energy_kwh',
    'energy_kwh',
    'inference_energy_kwh',
    'training_energy_kwh',
    'comparison.full_dataset_metrics.compressed_energy_kwh',
  ]);
  const trainingCo2 = firstFiniteNumber(raw, ['training_co2_kg', 'training_emissions_kg']);
  const inferenceCo2 = firstFiniteNumber(raw, ['inference_co2_kg', 'inference_emissions_kg']);
  const trainingEnergy = firstFiniteNumber(raw, ['training_energy_kwh']);
  const inferenceEnergy = firstFiniteNumber(raw, ['inference_energy_kwh']);

  const normalizedEnergyReduction =
    baselineTotalEnergy != null &&
    baselineTotalEnergy > 0 &&
    compressedTotalEnergy != null
      ? ((baselineTotalEnergy - compressedTotalEnergy) / baselineTotalEnergy) * 100
      : toFiniteNumber(result.energy_reduction_percent);

  const baselineLatency = firstFiniteNumber(raw, [
    'baseline_latency_ms',
    'comparison.full_dataset_metrics.baseline_latency_ms',
  ]);
  const compressedLatency = firstFiniteNumber(raw, [
    'compressed_latency_ms',
    'latency_ms',
    'inference_latency_ms',
    'comparison.full_dataset_metrics.compressed_latency_ms',
  ]);
  const normalizedLatencySpeedup =
    firstFiniteNumber(raw, ['latency_speedup_percent']) ??
    (baselineLatency != null && baselineLatency > 0 && compressedLatency != null
      ? ((baselineLatency - compressedLatency) / baselineLatency) * 100
      : undefined);

  const normalizedBaselineAccuracy =
    baselineAccuracy ?? toFiniteNumber(result.baseline_accuracy) ?? Number.NaN;
  const normalizedCompressedAccuracy =
    compressedAccuracy ?? toFiniteNumber(result.compressed_accuracy) ?? Number.NaN;
  const normalizedBaselineSizeValue =
    normalizedBaselineSize ?? toFiniteNumber(result.baseline_size_MB) ?? Number.NaN;
  const normalizedCompressedSizeValue =
    normalizedCompressedSize ?? toFiniteNumber(result.size_MB) ?? Number.NaN;

  return {
    ...result,
    baseline_accuracy: normalizedBaselineAccuracy,
    compressed_accuracy: normalizedCompressedAccuracy,
    baseline_size_MB: normalizedBaselineSizeValue,
    original_size_MB: normalizedBaselineSize ?? result.original_size_MB,
    size_MB: normalizedCompressedSizeValue,
    compressed_size_MB: normalizedCompressedSize ?? result.compressed_size_MB,
    size_reduction_percent:
      reduction != null ? Number(reduction.toFixed(2)) : Number.NaN,
    baseline_latency_ms: baselineLatency ?? result.baseline_latency_ms,
    compressed_latency_ms: compressedLatency ?? result.compressed_latency_ms,
    latency_ms: compressedLatency ?? result.latency_ms,
    latency_speedup_percent:
      normalizedLatencySpeedup != null
        ? Number(normalizedLatencySpeedup.toFixed(2))
        : result.latency_speedup_percent,
    baseline_training_co2_kg:
      immutableBaselineTrainingCo2 ?? result.baseline_training_co2_kg,
    baseline_total_emissions_kg: baselineTotalCo2 ?? result.baseline_total_emissions_kg,
    training_co2_kg: trainingCo2 ?? result.training_co2_kg,
    inference_co2_kg: inferenceCo2 ?? result.inference_co2_kg,
    compressed_total_emissions_kg:
      normalizedCompressedCo2 ?? result.compressed_total_emissions_kg,
    co2_kg: normalizedCompressedCo2 ?? inferenceCo2 ?? trainingCo2 ?? result.co2_kg,
    emissions_kg:
      normalizedCompressedCo2 ??
      inferenceCo2 ??
      trainingCo2 ??
      toFiniteNumber(result.emissions_kg) ??
      Number.NaN,
    baseline_total_energy_kwh: baselineTotalEnergy ?? result.baseline_total_energy_kwh,
    compressed_total_energy_kwh: compressedTotalEnergy ?? result.compressed_total_energy_kwh,
    training_energy_kwh: trainingEnergy ?? result.training_energy_kwh,
    inference_energy_kwh: inferenceEnergy ?? result.inference_energy_kwh,
    energy_kwh:
      compressedTotalEnergy ??
      inferenceEnergy ??
      trainingEnergy ??
      result.energy_kwh,
    emissions_reduction_percent:
      normalizedReduction != null
        ? Number(normalizedReduction.toFixed(2))
        : result.emissions_reduction_percent,
    energy_reduction_percent:
      normalizedEnergyReduction != null
        ? Number(normalizedEnergyReduction.toFixed(2))
        : result.energy_reduction_percent,
  };
}

// ============================================================
// Types
// ============================================================

export interface CompressionStep {
  key: string;
  label: string;
}

export interface AccuracyCheckpoint {
  stage: string;
  step: number;
  accuracy: number;
  baseline_accuracy: number;
  accuracy_drop: number;
  within_threshold: boolean;
  allowed_drop: number;
}

export interface CompressionStatus {
  running: boolean;
  step: string;
  detail: string;
  progress: string;
  result: DynamicResult | null;
  error: string | null;
  steps: CompressionStep[];
}

export async function getCompressionStatus(): Promise<CompressionStatus> {
  const res = await fetch(`${API_BASE}/api/compress/preloaded/status`, {
    cache: 'no-store',
  });
  if (!res.ok) {
    throw new Error(`Status check failed: ${res.status}`);
  }
  const data = await res.json();
  if (data?.result) {
    data.result = normalizeDynamicResult(data.result as DynamicResult);
  }
  return data;
}

export interface DynamicResult {
  strategy: string;
  resolved_strategy?: string;
  resolved_technique?: string;
  user_intent_layer?: 'Smart' | 'Preset' | 'Manual' | string;
  model_name?: string;
  model_key?: string;
  saved_at?: string;
  source_result_file?: string;
  compression_method?: string;
  dataset?: string;
  input_size?: number;
  baseline_accuracy: number;
  compressed_accuracy: number;
  accuracy_drop_threshold?: number;
  accuracy_checkpoints?: AccuracyCheckpoint[];
  size_MB: number;
  baseline_size_MB: number;
  original_size_MB?: number;
  compressed_size_MB?: number;
  size_reduction_percent: number;
  latency_ms?: number;
  emissions_kg: number;
  co2_kg?: number;
  energy_kwh?: number;
  training_emissions_kg?: number;
  training_co2_kg?: number;
  training_energy_kwh?: number;
  inference_emissions_kg?: number;
  inference_co2_kg?: number;
  inference_energy_kwh?: number;
  flops?: number;
  flops_M?: number;
  sparsity_percent?: number;
  total_params?: number;
  nonzero_params?: number;
  pruning_amount?: number;
  quantization_type?: string;
  pipeline?: string;
  student_params?: number;
  teacher_params?: number;
  param_reduction_percent?: number;
  kd_epochs?: number;
  fine_tune_epochs?: number;
  baseline_latency_ms?: number;
  compressed_latency_ms?: number;
  latency_speedup_percent?: number;
  baseline_total_emissions_kg?: number;
  baseline_training_co2_kg?: number;
  compressed_total_emissions_kg?: number;
  baseline_total_energy_kwh?: number;
  baseline_training_energy_kwh?: number;
  compressed_total_energy_kwh?: number;
  emissions_reduction_percent?: number;
  energy_reduction_percent?: number;
  sanity_warnings?: string[];
  co2_method?: string;
  inference_images_processed?: number;
  inference_iterations?: number;
  benchmark_window_seconds?: number;
  benchmark_warmup_seconds?: number;
  energy_per_1k_images?: number | null;
  hardware_saturation_level?: number;
  bottleneck_analysis?: string;
  runtime_warning?: string;
  runtime_warnings?: string[];
  smart_router_enabled?: boolean;
  smart_router_group?: string;
  smart_router_strategy?: string;
  smart_target_batch_size?: number;
  smart_effective_batch_size?: number;
  smart_precision_preference?: string;
  benchmark_training_epochs?: number;
  benchmark_training_max_batches?: number;
  benchmark_inference_max_batches?: number | null;
}

export interface Strategy {
  key: string;
  name: string;
  accuracy: number;
  size_MB: number;
  size_reduction: number;
  latency_ms?: number;
  baseline_latency_ms?: number;
  compressed_latency_ms?: number;
  latency_speedup_percent?: number;
  params: number;
  inference_energy_kWh?: number;
  training_energy_kwh?: number;
  inference_energy_kwh?: number;
  training_co2_kg?: number;
  inference_co2_kg?: number;
  baseline_co2_kg?: number;
  compressed_co2_kg?: number;
  co2_kg?: number;
  accuracy_top5?: number;
  flops_M?: number;
  sparsity_percent?: number;
}

export interface DashboardData {
  strategies: Strategy[];
  energy: Record<string, any>;
  models: { filename: string; size_MB: number }[];
  gpu_available: boolean;
  task_status: Record<string, { running: boolean; last_run: string | null; error: string | null }>;
}

// ============================================================
// Baseline model types
// ============================================================

export interface BaselineModel {
  model_key: string;
  model_name: string;
  params_label: string;
  total_params?: number;
  input_size: number;
  dataset: string;
  accuracy?: number;
  size_MB?: number;
  latency_ms?: number;
  training_co2_kg?: number | null;
  training_energy_kwh?: number | null;
  result_updated_at?: string;
  status: 'ready' | 'not_ready' | 'error';
  error?: string;
}

export interface BaselinesResponse {
  models: Record<string, BaselineModel>;
  ready_count: number;
  total_count: number;
}

export interface PrepareStatus {
  running: boolean;
  progress: string;
  completed: number;
  total: number;
  current_model: string;
  error: string | null;
}

export interface CompressionHistoryResponse {
  history: Record<string, DynamicResult[]>;
}

// ============================================================
// Baseline API calls
// ============================================================

export async function getBaselines(): Promise<BaselinesResponse> {
  return fetchAPI('/api/baselines');
}

export async function getPrepareStatus(): Promise<PrepareStatus> {
  return fetchAPI('/api/prepare/status');
}

export async function triggerPrepare(models?: string[]): Promise<void> {
  const res = await fetch(`${API_BASE}/api/prepare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: models ? JSON.stringify(models) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Prepare failed');
  }
}

export async function getCompressionHistory(): Promise<CompressionHistoryResponse> {
  return fetchAPI('/api/compression-history');
}

// ============================================================
// Model comparison API calls
// ============================================================

export interface ModelComparisonSample {
  id: number;
  label: string;
  class_index?: number;
  is_focus_class?: boolean;
  source_path?: string;
  image_data_url: string;
}

export interface ModelComparisonBaselineOption {
  key: string;
  name: string;
  input_size: number;
  dataset: string;
  size_MB?: number | null;
  co2_kg?: number | null;
  status?: string;
}

export interface ModelComparisonCompressedOption {
  key: string;
  model_key: string;
  model_name: string;
  strategy: string;
  strategy_label: string;
  label: string;
  size_MB?: number | null;
  co2_kg?: number | null;
  artifact_name?: string;
  saved_at?: string | null;
  source_result_file?: string | null;
  baseline_accuracy?: number | null;
  compressed_accuracy?: number | null;
  baseline_size_MB?: number | null;
  compressed_size_MB?: number | null;
  size_reduction_percent?: number | null;
  latency_ms?: number | null;
  latency_speedup_percent?: number | null;
  baseline_total_emissions_kg?: number | null;
  compressed_total_emissions_kg?: number | null;
  training_co2_kg?: number | null;
  inference_co2_kg?: number | null;
  baseline_total_energy_kwh?: number | null;
  compressed_total_energy_kwh?: number | null;
  training_energy_kwh?: number | null;
  inference_energy_kwh?: number | null;
  energy_kwh?: number | null;
  emissions_reduction_percent?: number | null;
  energy_reduction_percent?: number | null;
  co2_method?: string | null;
  bottleneck_analysis?: string | null;
  observed_baseline_accuracy_percent?: number | null;
  observed_compressed_accuracy_percent?: number | null;
  observed_baseline_latency_ms_per_image?: number | null;
  observed_compressed_latency_ms_per_image?: number | null;
  expected_compressed_accuracy_percent?: number | null;
  compressed_accuracy_delta_percent?: number | null;
  baseline_latency_ms?: number | null;
  compressed_latency_ms?: number | null;
  prediction_agreement_percent?: number | null;
  sample_count?: number | null;
  details_file?: string | null;
  summary_generated_at?: string | null;
  fallback_applied?: boolean;
  effective_model_key?: string;
  effective_strategy?: string;
}

export interface ModelComparisonOptionsResponse {
  baseline_models: ModelComparisonBaselineOption[];
  compressed_models: ModelComparisonCompressedOption[];
}

export interface ModelComparisonPrediction {
  predicted_class: string;
  predicted_index: number;
  confidence: number;
  input_size: number;
  temperature?: number;
  top_k?: Array<{
    class_name: string;
    class_index: number;
    probability: number;
  }>;
  quality_warnings?: string[];
  blur_edge_score?: number;
  normalization_mean?: [number, number, number];
  normalization_std?: [number, number, number];
  tta_variants?: number;
}

export interface ModelComparisonCaseDiagnostics {
  sample_id: number;
  true_label: string;
  baseline_predicted_class: string;
  compressed_predicted_class: string;
  baseline_confidence: number;
  compressed_confidence: number;
  confidence_drop_percent: number;
  significant_confidence_drop: boolean;
  focus_class: boolean;
  focus_class_misclassification: boolean;
  baseline_correct: boolean;
  compressed_correct: boolean;
  baseline_top3?: Array<{
    class_name: string;
    class_index: number;
    probability: number;
  }>;
  compressed_top3?: Array<{
    class_name: string;
    class_index: number;
    probability: number;
  }>;
}

export interface ModelComparisonResult {
  sample: {
    id: number;
    true_label: string;
    is_focus_class?: boolean;
    source_path?: string;
    image_data_url: string;
  };
  baseline: {
    model_key: string;
    model_name: string;
    prediction: ModelComparisonPrediction;
    accuracy?: number | null;
    latency_ms?: number | null;
    size_MB?: number | null;
    co2_kg?: number | null;
    energy_kwh?: number | null;
  };
  compressed: {
    model_key: string;
    strategy: string;
    model_name: string;
    strategy_label: string;
    prediction: ModelComparisonPrediction;
    accuracy?: number | null;
    latency_ms?: number | null;
    size_MB?: number | null;
    co2_kg?: number | null;
    energy_kwh?: number | null;
    artifact?: string;
  };
  comparison: {
    confidence_delta_percent: number;
    size_reduction_percent?: number | null;
    co2_reduction_percent?: number | null;
    energy_reduction_percent?: number | null;
    accuracy_delta_percent?: number | null;
    latency_reduction_percent?: number | null;
    prediction_match?: boolean;
    baseline_correct?: boolean;
    compressed_correct?: boolean;
    confidence_drop_alert?: boolean;
    confidence_drop_threshold_percent?: number;
    full_dataset_metrics?: {
      baseline_accuracy_percent?: number | null;
      compressed_accuracy_percent?: number | null;
      accuracy_delta_percent?: number | null;
      baseline_latency_ms?: number | null;
      compressed_latency_ms?: number | null;
      latency_reduction_percent?: number | null;
      baseline_size_MB?: number | null;
      compressed_size_MB?: number | null;
      size_reduction_percent?: number | null;
      baseline_co2_kg?: number | null;
      compressed_co2_kg?: number | null;
      co2_reduction_percent?: number | null;
      baseline_energy_kwh?: number | null;
      compressed_energy_kwh?: number | null;
      energy_reduction_percent?: number | null;
    };
    summary: string;
  };
  diagnostics?: {
    case?: ModelComparisonCaseDiagnostics;
    quality_warnings?: string[];
    baseline_top3?: Array<{
      class_name: string;
      class_index: number;
      probability: number;
    }>;
    compressed_top3?: Array<{
      class_name: string;
      class_index: number;
      probability: number;
    }>;
  };
  preprocessing?: {
    resize?: string;
    baseline_input_size?: number;
    compressed_input_size?: number;
    normalize_mean?: number[];
    normalize_std?: number[];
    dataset_loader?: string;
    tta_enabled?: boolean;
    tta_variants?: number;
  };
  device?: string;
}

export interface ModelComparisonBatchItem {
  sample_id: number;
  true_label: string;
  is_focus_class?: boolean;
  confidence_delta_percent?: number;
  quality_warnings?: string[];
  image_data_url: string;
  baseline_prediction: ModelComparisonPrediction;
  compressed_prediction: ModelComparisonPrediction;
  diagnostics?: ModelComparisonCaseDiagnostics;
}

export interface ModelComparisonBatchDiagnostics {
  focus_misclassifications?: Array<{
    sample_id: number;
    true_label: string;
    baseline_predicted_class: string;
    compressed_predicted_class: string;
    baseline_confidence: number;
    compressed_confidence: number;
  }>;
  significant_confidence_drop_cases?: Array<{
    sample_id: number;
    true_label: string;
    baseline_confidence: number;
    compressed_confidence: number;
    confidence_drop_percent: number;
  }>;
}

export interface ModelComparisonBatchResult {
  count: number;
  batch_size: number;
  baseline_model_key: string;
  compressed_model_key: string;
  compressed_strategy: string;
  compressed_artifact?: string;
  results: ModelComparisonBatchItem[];
  summary: {
    baseline_accuracy_percent: number;
    compressed_accuracy_percent: number;
    prediction_agreement_percent: number;
    baseline_latency_ms_per_image: number;
    compressed_latency_ms_per_image: number;
    per_class_accuracy?: Record<
      string,
      {
        count: number;
        baseline_accuracy_percent: number;
        compressed_accuracy_percent: number;
      }
    >;
    focus_class_accuracy?: Record<
      string,
      {
        count: number;
        baseline_accuracy_percent: number;
        compressed_accuracy_percent: number;
      }
    >;
    focus_misclassification_count?: number;
    significant_confidence_drop_count?: number;
  };
  diagnostics?: ModelComparisonBatchDiagnostics;
  preprocessing?: {
    resize?: string;
    baseline_input_size?: number;
    compressed_input_size?: number;
    normalize_mean?: number[];
    normalize_std?: number[];
    dataset_loader?: string;
    tta_enabled?: boolean;
    assets_dir?: string;
  };
  device?: string;
}

export async function getModelComparisonOptions(): Promise<ModelComparisonOptionsResponse> {
  return fetchAPI('/api/model-comparison/options');
}

export async function getModelComparisonSamples(limit: number = 10): Promise<ModelComparisonSample[]> {
  const payload = await fetchAPI(`/api/model-comparison/sample-images?limit=${limit}`);
  if (Array.isArray(payload?.samples)) {
    return payload.samples as ModelComparisonSample[];
  }
  return [];
}

export async function compareModelsOnImage(payload: {
  sample_id: number;
  baseline_model_key: string;
  compressed_model_key: string;
  enable_tta?: boolean;
}): Promise<ModelComparisonResult> {
  return postAPI('/api/model-comparison/compare', payload);
}

export async function compareModelsOnBatch(payload: {
  sample_ids: number[];
  baseline_model_key: string;
  compressed_model_key: string;
  batch_size?: number;
  enable_tta?: boolean;
}): Promise<ModelComparisonBatchResult> {
  return postAPI('/api/model-comparison/compare-batch', payload);
}

export interface CompareImageClassPrediction {
  class_name: string;
  class_index: number;
  probability: number;
}

export interface CompareImageModelResult {
  model_key: string;
  strategy?: string;
  class: string;
  confidence: number;
  top3: CompareImageClassPrediction[];
  prediction: ModelComparisonPrediction;
}

export interface CompareImageResponse {
  input: {
    source: 'sample' | 'upload';
    sample_image_path?: string | null;
    upload_filename?: string | null;
    true_label?: string | null;
    image_data_url: string;
  };
  baseline: CompareImageModelResult;
  compressed: CompareImageModelResult;
  comparison: {
    prediction_match: boolean;
    prediction_mismatch_warning?: string | null;
    confidence_delta_percent: number;
    confidence_drop_alert: boolean;
    confidence_drop_threshold_percent: number;
    baseline_correct?: boolean | null;
    compressed_correct?: boolean | null;
  };
  diagnostics?: {
    case?: Record<string, any>;
    quality_warnings?: string[];
  };
  preprocessing?: {
    resize?: string;
    baseline_input_size?: number;
    compressed_input_size?: number;
    normalize_mean?: number[];
    normalize_std?: number[];
    tta_enabled?: boolean;
    tta_variants?: number;
  };
  device?: string;
}

export async function compareImage(payload: {
  baseline_model_key: string;
  compressed_model_key: string;
  sample_image_path?: string;
  image_file?: File;
  enable_tta?: boolean;
}): Promise<CompareImageResponse> {
  const formData = new FormData();
  formData.append('baseline_model_key', payload.baseline_model_key);
  formData.append('compressed_model_key', payload.compressed_model_key);
  formData.append('enable_tta', String(Boolean(payload.enable_tta)));

  if (payload.sample_image_path) {
    formData.append('sample_image_path', payload.sample_image_path);
  }
  if (payload.image_file) {
    formData.append('image_file', payload.image_file);
  }

  const res = await fetch(`${API_BASE}/compare-image`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error(await readApiError(res));
  }

  return res.json();
}

// ============================================================
// localStorage persistence for compression results
// ============================================================
const STORAGE_KEY = 'compressionHistory';
const LEGACY_STORAGE_KEY = 'greenai_compression_results';

function normalizeStorageToken(value: unknown): string {
  const token = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/-/g, '_')
    .replace(/[^a-z0-9_]/g, '');
  return token || 'unknown';
}

export function getResultStorageKey(result: DynamicResult): string {
  const modelToken = normalizeStorageToken(result.model_key || result.model_name || 'unknown_model');
  const strategyToken = normalizeStorageToken(result.strategy || result.compression_method || 'unknown_strategy');
  return `${modelToken}_${strategyToken}`;
}

function normalizeStoredResult(rawResult: DynamicResult): DynamicResult {
  const normalized = normalizeDynamicResult(rawResult);
  const rawWithAliases = rawResult as DynamicResult & {
    timestamp?: string;
    created_at?: string;
  };
  const savedAt =
    (typeof normalized.saved_at === 'string' && normalized.saved_at.trim() !== ''
      ? normalized.saved_at
      : undefined) ||
    (typeof rawWithAliases.timestamp === 'string' && rawWithAliases.timestamp.trim() !== ''
      ? rawWithAliases.timestamp
      : undefined) ||
    (typeof rawWithAliases.created_at === 'string' && rawWithAliases.created_at.trim() !== ''
      ? rawWithAliases.created_at
      : undefined);

  return {
    ...normalized,
    saved_at: savedAt,
  };
}

function dedupeResults(results: DynamicResult[]): DynamicResult[] {
  const byKey = new Map<string, DynamicResult>();

  for (const result of results) {
    const key = getResultStorageKey(result);
    const existing = byKey.get(key);

    if (!existing) {
      byKey.set(key, result);
      continue;
    }

    const existingTs = Date.parse(existing.saved_at || '');
    const incomingTs = Date.parse(result.saved_at || '');

    if (Number.isFinite(incomingTs) && !Number.isFinite(existingTs)) {
      byKey.set(key, result);
      continue;
    }
    if (Number.isFinite(incomingTs) && Number.isFinite(existingTs) && incomingTs >= existingTs) {
      byKey.set(key, result);
      continue;
    }
    if (!Number.isFinite(incomingTs) && !Number.isFinite(existingTs)) {
      byKey.set(key, result);
    }
  }

  const deduped = Array.from(byKey.values());
  deduped.sort((a, b) => {
    const aTs = Date.parse(a.saved_at || '');
    const bTs = Date.parse(b.saved_at || '');
    if (!Number.isFinite(aTs) && !Number.isFinite(bTs)) return 0;
    if (!Number.isFinite(aTs)) return -1;
    if (!Number.isFinite(bTs)) return 1;
    return aTs - bTs;
  });
  return deduped;
}

function persistResults(results: DynamicResult[]): DynamicResult[] {
  const deduped = dedupeResults(results);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(deduped));
  } catch {
    // Ignore quota errors to keep app functional.
  }
  return deduped;
}

export function loadSavedResults(): DynamicResult[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY) ?? localStorage.getItem(LEGACY_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    const normalized = parsed.map((r) => normalizeStoredResult(r as DynamicResult));
    const deduped = dedupeResults(normalized);
    // Migrate old key transparently once data is loaded.
    if (!localStorage.getItem(STORAGE_KEY)) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(deduped));
    }
    return deduped;
  } catch {
    return [];
  }
}

export function saveResult(result: DynamicResult): DynamicResult[] {
  const normalized = normalizeStoredResult({
    ...result,
    saved_at: new Date().toISOString(),
  });
  const existing = loadSavedResults();
  const key = getResultStorageKey(normalized);
  const filtered = existing.filter((r) => getResultStorageKey(r) !== key);
  const updated = [...filtered, normalized];
  return persistResults(updated);
}

export function deleteSavedResultByKey(resultKey: string): DynamicResult[] {
  if (typeof window === 'undefined') return [];
  const existing = loadSavedResults();
  const filtered = existing.filter((result) => getResultStorageKey(result) !== resultKey);
  return persistResults(filtered);
}

export function clearSavedResults(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(LEGACY_STORAGE_KEY);
}

function formatStrategyToken(token: string): string {
  const normalized = token.trim().toLowerCase();
  const labels: Record<string, string> = {
    smart: 'Smart Router',
    maximize_speed: 'Maximize Speed',
    minimize_size: 'Minimize Size',
    preserve_accuracy: 'Preserve Accuracy',
    pruning: 'Pruning',
    quantization: 'Quantization',
    hybrid: 'Hybrid',
    kd: 'Knowledge Distillation',
  };
  if (labels[normalized]) {
    return labels[normalized];
  }
  return normalized
    .replace(/apply_/g, '')
    .replace(/_/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

/** Convert a DynamicResult into a Strategy so it can be shown in charts/tables */
export function dynamicResultToStrategy(r: DynamicResult): Strategy {
  const normalized = normalizeDynamicResult(r);
  const modelLabel = normalized.model_name || normalized.model_key || 'Custom';
  const requestedLabel = formatStrategyToken(normalized.strategy || normalized.compression_method || '');
  const smartResolvedLabel = normalized.smart_router_strategy
    ? formatStrategyToken(normalized.smart_router_strategy)
    : '';
  const resolvedTechniqueLabel = normalized.resolved_technique
    ? formatStrategyToken(normalized.resolved_technique)
    : '';

  let methodLabel = requestedLabel;
  if (normalized.strategy === 'smart' && smartResolvedLabel) {
    methodLabel = `${requestedLabel} (${smartResolvedLabel})`;
  } else if (normalized.user_intent_layer === 'Preset' && resolvedTechniqueLabel) {
    methodLabel = `${requestedLabel} (${resolvedTechniqueLabel})`;
  }

  if (!methodLabel) {
    methodLabel = formatStrategyToken(normalized.compression_method || normalized.strategy || 'compression');
  }
  const baselineTotalCo2 = normalized.baseline_training_co2_kg ?? normalized.baseline_total_emissions_kg;
  const compressedTotalCo2 = normalized.compressed_total_emissions_kg;
  const trainingCo2 = normalized.training_co2_kg ?? normalized.training_emissions_kg;
  const totalCo2 = compressedTotalCo2;
  const inferenceCo2 = normalized.inference_co2_kg ?? normalized.inference_emissions_kg ?? normalized.emissions_kg;
  const trainingEnergy = normalized.training_energy_kwh;
  const totalEnergy = normalized.compressed_total_energy_kwh;
  const inferenceEnergy = normalized.inference_energy_kwh ?? normalized.energy_kwh;
  return {
    key: `dyn_${(normalized.model_key || normalized.model_name || 'x').replace(/\s/g, '_')}_${normalized.strategy}`,
    name: `${modelLabel} · ${methodLabel}`,
    accuracy: normalized.compressed_accuracy,
    size_MB: normalized.size_MB,
    size_reduction: normalized.size_reduction_percent,
    // latency_ms: r.latency_ms,
    params: normalized.total_params || normalized.nonzero_params || 0,
    baseline_co2_kg: baselineTotalCo2 ?? undefined,
    compressed_co2_kg: compressedTotalCo2 ?? inferenceCo2,
    co2_kg: totalCo2 ?? inferenceCo2,
    training_co2_kg: trainingCo2,
    inference_co2_kg: inferenceCo2,
    training_energy_kwh: trainingEnergy ?? totalEnergy,
    inference_energy_kwh: inferenceEnergy ?? totalEnergy,
    inference_energy_kWh: inferenceEnergy ?? totalEnergy,
    flops_M: normalized.flops_M,
    sparsity_percent: normalized.sparsity_percent,
  };
}
