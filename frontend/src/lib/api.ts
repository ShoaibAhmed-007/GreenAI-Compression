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
  return res.json();
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

// ============================================================
// Types
// ============================================================

export interface CompressionStep {
  key: string;
  label: string;
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
  return res.json();
}

export interface DynamicResult {
  strategy: string;
  model_name?: string;
  model_key?: string;
  compression_method?: string;
  dataset?: string;
  input_size?: number;
  baseline_accuracy: number;
  compressed_accuracy: number;
  size_MB: number;
  baseline_size_MB: number;
  size_reduction_percent: number;
  latency_ms: number;
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
  latency_ms: number;
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
// localStorage persistence for compression results
// ============================================================
const STORAGE_KEY = 'greenai_compression_results';

export function loadSavedResults(): DynamicResult[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveResult(result: DynamicResult): DynamicResult[] {
  const existing = loadSavedResults();
  const key = `${result.model_key || result.model_name || 'unknown'}_${result.strategy}`;
  const filtered = existing.filter(r => {
    const rKey = `${r.model_key || r.model_name || 'unknown'}_${r.strategy}`;
    return rKey !== key;
  });
  const updated = [...filtered, result];
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  } catch { /* quota exceeded */ }
  return updated;
}

export function clearSavedResults(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(STORAGE_KEY);
}

/** Convert a DynamicResult into a Strategy so it can be shown in charts/tables */
export function dynamicResultToStrategy(r: DynamicResult): Strategy {
  const modelLabel = r.model_name || r.model_key || 'Custom';
  const methodLabel = r.compression_method || r.strategy || '';
  const trainingCo2 = r.training_co2_kg ?? r.training_emissions_kg;
  const inferenceCo2 = r.inference_co2_kg ?? r.inference_emissions_kg ?? r.emissions_kg;
  const trainingEnergy = r.training_energy_kwh;
  const inferenceEnergy = r.inference_energy_kwh ?? r.energy_kwh;
  return {
    key: `dyn_${(r.model_key || r.model_name || 'x').replace(/\s/g, '_')}_${r.strategy}`,
    name: `${modelLabel} · ${methodLabel.charAt(0).toUpperCase() + methodLabel.slice(1)}`,
    accuracy: r.compressed_accuracy,
    size_MB: r.size_MB,
    size_reduction: r.size_reduction_percent,
    latency_ms: r.latency_ms,
    params: r.total_params || r.nonzero_params || 0,
    co2_kg: inferenceCo2,
    training_co2_kg: trainingCo2,
    inference_co2_kg: inferenceCo2,
    training_energy_kwh: trainingEnergy,
    inference_energy_kwh: inferenceEnergy,
    inference_energy_kWh: inferenceEnergy,
    flops_M: r.flops_M,
    sparsity_percent: r.sparsity_percent,
  };
}
