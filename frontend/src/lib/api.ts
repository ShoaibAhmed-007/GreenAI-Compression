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

export function normalizeDynamicResult(result: DynamicResult): DynamicResult {
  const baselineSize =
    toFiniteNumber(result.baseline_size_MB) ?? toFiniteNumber(result.original_size_MB);
  const compressedSize =
    toFiniteNumber(result.size_MB) ?? toFiniteNumber(result.compressed_size_MB);

  const computedReduction =
    baselineSize != null && baselineSize > 0 && compressedSize != null
      ? ((baselineSize - compressedSize) / baselineSize) * 100
      : undefined;

  const reduction =
    toFiniteNumber(result.size_reduction_percent) ?? computedReduction ?? 0;

  const safeBaselineSize = baselineSize ?? compressedSize ?? 0;
  const safeCompressedSize = compressedSize ?? safeBaselineSize;

  const immutableBaselineTrainingCo2 = toFiniteNumber(result.baseline_training_co2_kg);
  const baselineTotalCo2 = immutableBaselineTrainingCo2 ?? toFiniteNumber(result.baseline_total_emissions_kg);
  const fallbackCompressedCo2 =
    toFiniteNumber(result.compressed_total_emissions_kg) ??
    toFiniteNumber(result.co2_kg) ??
    toFiniteNumber(result.emissions_kg);

  const projectedCompressedCo2 =
    baselineTotalCo2 != null && baselineTotalCo2 > 0 && safeBaselineSize > 0
      ? baselineTotalCo2 * (safeCompressedSize / safeBaselineSize)
      : undefined;

  const normalizedCompressedCo2 = projectedCompressedCo2 ?? fallbackCompressedCo2;
  const normalizedReduction =
    baselineTotalCo2 != null &&
    baselineTotalCo2 > 0 &&
    normalizedCompressedCo2 != null
      ? ((baselineTotalCo2 - normalizedCompressedCo2) / baselineTotalCo2) * 100
      : toFiniteNumber(result.emissions_reduction_percent);

  return {
    ...result,
    baseline_size_MB: safeBaselineSize,
    size_MB: safeCompressedSize,
    size_reduction_percent: Number(reduction.toFixed(2)),
    baseline_training_co2_kg:
      immutableBaselineTrainingCo2 ?? result.baseline_training_co2_kg,
    baseline_total_emissions_kg: baselineTotalCo2 ?? result.baseline_total_emissions_kg,
    compressed_total_emissions_kg:
      normalizedCompressedCo2 ?? result.compressed_total_emissions_kg,
    co2_kg: normalizedCompressedCo2 ?? result.co2_kg,
    emissions_kg: normalizedCompressedCo2 ?? result.emissions_kg,
    emissions_reduction_percent:
      normalizedReduction != null
        ? Number(normalizedReduction.toFixed(2))
        : result.emissions_reduction_percent,
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
  model_name?: string;
  model_key?: string;
  saved_at?: string;
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

/** Convert a DynamicResult into a Strategy so it can be shown in charts/tables */
export function dynamicResultToStrategy(r: DynamicResult): Strategy {
  const normalized = normalizeDynamicResult(r);
  const modelLabel = normalized.model_name || normalized.model_key || 'Custom';
  const methodLabel = normalized.compression_method || normalized.strategy || '';
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
    name: `${modelLabel} · ${methodLabel.charAt(0).toUpperCase() + methodLabel.slice(1)}`,
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
