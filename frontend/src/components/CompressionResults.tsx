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

function strategyLabel(result: DynamicResult | null): string {
  if (!result) return 'Not Available';
  const raw = String(result.strategy || result.compression_method || '').trim();
  if (raw === '') return 'Not Available';
  return raw.charAt(0).toUpperCase() + raw.slice(1);
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
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm text-gray-600">
        No compressed result available for the selected technique.
      </div>
    );
  }

  const accuracy = resolveAccuracy(result);
  const latency = resolveLatency(result);
  const size = resolveSize(result);
  const co2 = resolveCo2(result);

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-gray-200 bg-white p-3">
        <p className="text-xs uppercase tracking-wide text-gray-500">Applied Compression Technique</p>
        <p className="text-base font-semibold text-gray-900 mt-1">{strategyLabel(result)}</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="rounded-lg border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">Compressed Model Size</p>
          <p className="text-lg font-bold text-gray-900 mt-1">{formatSize(size)}</p>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">Compressed Accuracy</p>
          <p className="text-lg font-bold text-green-700 mt-1">{formatPercent(accuracy)}</p>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">Compressed Latency</p>
          <p className="text-lg font-bold text-blue-700 mt-1">{formatMs(latency)}</p>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-3">
          <p className="text-xs uppercase tracking-wide text-gray-500">Compressed CO2 Emissions</p>
          <p className="text-lg font-bold text-emerald-700 mt-1">{formatCo2(co2)}</p>
        </div>
      </div>

      <p className="text-xs text-gray-500">
        Compression Results section intentionally shows compressed metrics only.
      </p>
    </div>
  );
}
