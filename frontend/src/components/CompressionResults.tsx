'use client';

import { DynamicResult } from '@/lib/api';

interface CompressionResultsProps {
  result: DynamicResult | null;
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function readPath(source: Record<string, unknown>, path: string): unknown {
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

function firstFiniteValue(result: DynamicResult, keys: string[]): number | null {
  const raw = result as unknown as Record<string, unknown>;

  for (const key of keys) {
    const value = key.includes('.') ? readPath(raw, key) : raw[key];
    const numeric = toFiniteNumber(value);
    if (numeric != null) {
      return numeric;
    }
  }

  return null;
}

function firstTextValue(result: DynamicResult, keys: string[]): string | null {
  const raw = result as unknown as Record<string, unknown>;

  for (const key of keys) {
    const value = key.includes('.') ? readPath(raw, key) : raw[key];
    if (typeof value === 'string' && value.trim() !== '') {
      return value;
    }
  }

  return null;
}

function formatPercent(value: number | null): string {
  if (value == null) return 'Not Available';
  return `${value.toFixed(2)}%`;
}

function formatSize(value: number | null): string {
  if (value == null) return 'Not Available';
  return `${value.toFixed(2)} MB`;
}

function formatMs(value: number | null): string {
  if (value == null) return 'Not Available';
  return `${value.toFixed(2)} ms`;
}

function formatCo2(value: number | null): string {
  if (value == null) return 'Not Available';
  if (value > 0 && value < 0.000001) return '<0.000001 kg';
  return `${value.toFixed(6)} kg`;
}

function formatEnergyPer1k(value: number | null): string {
  if (value == null) return 'Not Available';
  return `${value.toFixed(8)} kWh / 1k images`;
}

function formatRatio(value: number | null): string {
  if (value == null) return 'Not Available';
  return `${(value * 100).toFixed(1)}%`;
}

function formatStrategyToken(value: string): string {
  const normalized = value.trim().toLowerCase();
  const labels: Record<string, string> = {
    smart: 'Auto-Green (Smart)',
    maximize_speed: 'Preset: Maximize Speed',
    minimize_size: 'Preset: Minimize Size',
    preserve_accuracy: 'Preset: Preserve Accuracy',
    pruning: 'Manual: Pruning',
    quantization: 'Manual: Quantization',
    hybrid: 'Manual: Hybrid',
    kd: 'Manual: Knowledge Distillation',
  };
  if (labels[normalized]) return labels[normalized];
  return normalized
    .replace(/apply_/g, '')
    .replace(/_/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function sanitizeFilenameToken(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/-/g, '_')
    .replace(/[^a-z0-9_]/g, '') || 'unknown';
}

function buildExportFilename(result: DynamicResult): string {
  const modelToken = sanitizeFilenameToken(result.model_key || result.model_name || 'model');
  const strategyToken = sanitizeFilenameToken(result.strategy || result.compression_method || 'compression');

  const savedAtMs = Date.parse(result.saved_at || '');
  const timestamp = Number.isFinite(savedAtMs)
    ? new Date(savedAtMs).toISOString().replace(/[:.]/g, '-')
    : null;

  return timestamp
    ? `${modelToken}_${strategyToken}_compression_result_${timestamp}.json`
    : `${modelToken}_${strategyToken}_compression_result.json`;
}

function strategyLabel(result: DynamicResult | null): string {
  if (!result) return 'Not Available';
  const raw = String(
    result.strategy ||
      result.compression_method ||
      firstTextValue(result, ['effective_strategy', 'resolved_strategy', 'resolved_technique']) ||
      ''
  ).trim();
  if (raw === '') return 'Not Available';
  return formatStrategyToken(raw);
}

function resolveAccuracy(result: DynamicResult | null): number | null {
  if (!result) return null;
  return firstFiniteValue(result, [
    'compressed_accuracy',
    'accuracy',
    'accuracy_top1',
    'top1_accuracy',
    'comparison.full_dataset_metrics.compressed_accuracy_percent',
    'observed_compressed_accuracy_percent',
    'expected_compressed_accuracy_percent',
    'batch_result.summary.compressed_accuracy_percent',
  ]);
}

function resolveLatency(result: DynamicResult | null): number | null {
  if (!result) return null;
  return firstFiniteValue(result, [
    'compressed_latency_ms',
    'latency_ms',
    'inference_latency_ms',
    'comparison.full_dataset_metrics.compressed_latency_ms',
    'batch_result.summary.compressed_latency_ms',
  ]);
}

function resolveSize(result: DynamicResult | null): number | null {
  if (!result) return null;
  return firstFiniteValue(result, [
    'compressed_size_MB',
    'size_MB',
    'size_MB_quant',
    'size_MB_sparse',
    'size_MB_compressed',
    'comparison.full_dataset_metrics.compressed_size_MB',
  ]);
}

function resolveCo2(result: DynamicResult | null): number | null {
  if (!result) return null;
  return firstFiniteValue(result, [
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
}

function resolveEnergyPer1k(result: DynamicResult | null): number | null {
  if (!result) return null;

  const direct = firstFiniteValue(result, [
    'energy_per_1k_images',
    'comparison.full_dataset_metrics.energy_per_1k_images',
  ]);
  if (direct != null) {
    return direct;
  }

  const totalEnergy = firstFiniteValue(result, [
    'inference_energy_kwh',
    'energy_kwh',
    'compressed_total_energy_kwh',
  ]);
  const processedImages = firstFiniteValue(result, [
    'inference_images_processed',
    'sample_count',
    'batch_result.aggregate.sample_count',
  ]);

  if (totalEnergy != null && processedImages != null && processedImages > 0) {
    return (totalEnergy / processedImages) * 1000;
  }

  return null;
}

function resolveSaturation(result: DynamicResult | null): number | null {
  if (!result) return null;

  const value = firstFiniteValue(result, [
    'hardware_saturation_level',
    'utilization_ratio',
    'gpu_utilization_ratio',
  ]);

  if (value == null) return null;
  if (value > 1 && value <= 100) {
    return value / 100;
  }
  return value;
}

export function CompressionResults({ result }: CompressionResultsProps) {
  if (!result) {
    return (
      <div className="bg-surface-container-low rounded-xl p-4 text-sm text-on-surface-variant">
        No compressed result available for the selected technique.
      </div>
    );
  }

  const accuracy = resolveAccuracy(result);
  const latency = resolveLatency(result);
  const size = resolveSize(result);
  const co2 = resolveCo2(result);
  const energyPer1k = resolveEnergyPer1k(result);
  const saturation = resolveSaturation(result);
  const co2Method =
    firstTextValue(result, ['co2_method', 'emissions_method', 'benchmark.method']) || 'Not Available';
  const bottleneck =
    firstTextValue(result, ['bottleneck_analysis', 'bottleneck', 'diagnosis.summary']) || 'Not Available';
  const smartRoute = result.strategy === 'smart' ? (result.smart_router_strategy || result.resolved_strategy) : null;
  const resolvedTechnique = result.resolved_technique ? formatStrategyToken(result.resolved_technique) : null;
  const userIntentLayer = result.user_intent_layer || null;

  const handleExportResult = () => {
    if (typeof window === 'undefined') return;

    try {
      const content = JSON.stringify(result, null, 2);
      const blob = new Blob([content], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.download = buildExportFilename(result);
      document.body.appendChild(link);
      link.click();
      link.remove();

      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export compression result JSON:', error);
    }
  };

  return (
    <div className="space-y-3">
      <div className="bg-surface-container-low rounded-xl p-4">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Applied Compression Technique</p>
            <p className="text-base font-semibold text-on-surface mt-1">{strategyLabel(result)}</p>
          </div>
          <button
            type="button"
            onClick={handleExportResult}
            className="btn-secondary w-full sm:w-auto"
            title="Download this compression result as JSON"
            aria-label="Download this compression result as JSON"
          >
            <span className="material-symbols-outlined text-base">download</span>
            Export JSON
          </button>
        </div>
        {smartRoute && (
          <p className="text-xs text-on-surface-variant mt-1">
            Router selected: <span className="font-semibold text-primary">{formatStrategyToken(String(smartRoute))}</span>
          </p>
        )}
        {userIntentLayer && (
          <p className="text-xs text-on-surface-variant mt-1">
            Intent layer: <span className="font-semibold text-secondary">{userIntentLayer}</span>
          </p>
        )}
        {resolvedTechnique && (
          <p className="text-xs text-on-surface-variant mt-1">
            Resolved technique: <span className="font-semibold text-tertiary">{resolvedTechnique}</span>
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="bg-surface-container-low rounded-xl p-4">
          <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Compressed Model Size</p>
          <p className="text-2xl font-technical font-bold text-primary mt-1">{formatSize(size)}</p>
        </div>

        <div className="bg-surface-container-low rounded-xl p-4">
          <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Compressed Accuracy</p>
          <p className="text-2xl font-technical font-bold text-secondary mt-1">{formatPercent(accuracy)}</p>
        </div>

        <div className="bg-surface-container-low rounded-xl p-4">
          <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Compressed Latency</p>
          <p className="text-2xl font-technical font-bold text-tertiary mt-1">{formatMs(latency)}</p>
        </div>

        <div className="bg-surface-container-low rounded-xl p-4">
          <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Compressed CO2 Emissions</p>
          <p className="text-2xl font-technical font-bold text-on-surface mt-1">{formatCo2(co2)}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="bg-surface-container-low rounded-xl p-4">
          <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Energy Intensity</p>
          <p className="text-sm font-technical font-semibold text-primary mt-1">{formatEnergyPer1k(energyPer1k)}</p>
        </div>

        <div className="bg-surface-container-low rounded-xl p-4">
          <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Hardware Saturation</p>
          <p className="text-sm font-technical font-semibold text-secondary mt-1">{formatRatio(saturation)}</p>
        </div>
      </div>

      <div className="bg-surface-container-low rounded-xl p-4 space-y-2">
        <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Benchmark Notes</p>
        <p className="text-sm text-on-surface"><span className="font-semibold">CO2 method:</span> {co2Method}</p>
        <p className="text-sm text-on-surface"><span className="font-semibold">Bottleneck analysis:</span> {bottleneck}</p>
      </div>

      <p className="text-xs text-on-surface-variant/50">
        Compression Results section intentionally shows compressed metrics only.
      </p>
    </div>
  );
}
