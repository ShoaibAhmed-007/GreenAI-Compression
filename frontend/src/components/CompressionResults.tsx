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

function strategyLabel(result: DynamicResult | null): string {
  if (!result) return 'Not Available';
  const raw = String(result.strategy || result.compression_method || '').trim();
  if (raw === '') return 'Not Available';
  return formatStrategyToken(raw);
}

function resolveAccuracy(result: DynamicResult | null): number | null {
  if (!result) return null;
  return toFiniteNumber(result.compressed_accuracy ?? null);
}

function resolveLatency(result: DynamicResult | null): number | null {
  if (!result) return null;
  return toFiniteNumber(result.compressed_latency_ms ?? result.latency_ms ?? null);
}

function resolveSize(result: DynamicResult | null): number | null {
  if (!result) return null;
  return toFiniteNumber(result.compressed_size_MB ?? result.size_MB ?? null);
}

function resolveCo2(result: DynamicResult | null): number | null {
  if (!result) return null;
  return toFiniteNumber(
    result.compressed_total_emissions_kg ??
      result.training_co2_kg ??
      result.training_emissions_kg ??
      result.inference_co2_kg ??
      result.inference_emissions_kg ??
      result.co2_kg ??
      result.emissions_kg ??
      null
  );
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
  const energyPer1k = toFiniteNumber(result.energy_per_1k_images ?? null);
  const saturation = toFiniteNumber(result.hardware_saturation_level ?? null);
  const co2Method = result.co2_method || 'Not Available';
  const bottleneck = result.bottleneck_analysis || 'Not Available';
  const smartRoute = result.strategy === 'smart' ? (result.smart_router_strategy || result.resolved_strategy) : null;
  const resolvedTechnique = result.resolved_technique ? formatStrategyToken(result.resolved_technique) : null;
  const userIntentLayer = result.user_intent_layer || null;

  return (
    <div className="space-y-3">
      <div className="bg-surface-container-low rounded-xl p-4">
        <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Applied Compression Technique</p>
        <p className="text-base font-semibold text-on-surface mt-1">{strategyLabel(result)}</p>
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
