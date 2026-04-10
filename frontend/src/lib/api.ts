const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function fetchAPI(endpoint: string) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    cache: 'no-store',
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
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
    throw new Error(`API error: ${res.status} ${res.statusText}`);
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

  return {
    ...result,
    baseline_size_MB: safeBaselineSize,
    size_MB: safeCompressedSize,
    size_reduction_percent: Number(reduction.toFixed(2)),
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
  // latency_ms: number;
  emissions_kg: number;
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
}

export interface Strategy {
  key: string;
  name: string;
  accuracy: number;
  size_MB: number;
  size_reduction: number;
  // latency_ms: number;
  params: number;
  inference_energy_kWh?: number;
  training_energy_kwh?: number;
  inference_energy_kwh?: number;
  training_co2_kg?: number;
  inference_co2_kg?: number;
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
}

export interface ModelComparisonResult {
  sample: {
    id: number;
    true_label: string;
    image_data_url: string;
  };
  baseline: {
    model_key: string;
    model_name: string;
    prediction: ModelComparisonPrediction;
    size_MB?: number | null;
    co2_kg?: number | null;
  };
  compressed: {
    model_key: string;
    strategy: string;
    model_name: string;
    strategy_label: string;
    prediction: ModelComparisonPrediction;
    size_MB?: number | null;
    co2_kg?: number | null;
    artifact?: string;
  };
  comparison: {
    confidence_delta_percent: number;
    size_reduction_percent?: number | null;
    co2_reduction_percent?: number | null;
    summary: string;
  };
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
}): Promise<ModelComparisonResult> {
  return postAPI('/api/model-comparison/compare', payload);
}

// ============================================================
// localStorage persistence for compression results
// ============================================================
const STORAGE_KEY = 'greenai_compression_results';

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
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    const normalized = parsed.map((r) => normalizeStoredResult(r as DynamicResult));
    return dedupeResults(normalized);
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
}

/** Convert a DynamicResult into a Strategy so it can be shown in charts/tables */
export function dynamicResultToStrategy(r: DynamicResult): Strategy {
  const normalized = normalizeDynamicResult(r);
  const modelLabel = normalized.model_name || normalized.model_key || 'Custom';
  const methodLabel = normalized.compression_method || normalized.strategy || '';
  const trainingCo2 = normalized.training_co2_kg ?? normalized.training_emissions_kg;
  const inferenceCo2 = normalized.inference_co2_kg ?? normalized.inference_emissions_kg ?? normalized.emissions_kg;
  const trainingEnergy = normalized.training_energy_kwh;
  const inferenceEnergy = normalized.inference_energy_kwh ?? normalized.energy_kwh;
  return {
    key: `dyn_${(normalized.model_key || normalized.model_name || 'x').replace(/\s/g, '_')}_${normalized.strategy}`,
    name: `${modelLabel} · ${methodLabel.charAt(0).toUpperCase() + methodLabel.slice(1)}`,
    accuracy: normalized.compressed_accuracy,
    size_MB: normalized.size_MB,
    size_reduction: normalized.size_reduction_percent,
    // latency_ms: r.latency_ms,
    params: normalized.total_params || normalized.nonzero_params || 0,
    co2_kg: inferenceCo2,
    training_co2_kg: trainingCo2,
    inference_co2_kg: inferenceCo2,
    training_energy_kwh: trainingEnergy,
    inference_energy_kwh: inferenceEnergy,
    inference_energy_kWh: inferenceEnergy,
    flops_M: normalized.flops_M,
    sparsity_percent: normalized.sparsity_percent,
  };
}
