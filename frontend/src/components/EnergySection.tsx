'use client';

import { BaselineModel, DynamicResult, getResultStorageKey } from '@/lib/api';

interface EnergySectionProps {
  energy: Record<string, any>;
  savedResults?: DynamicResult[];
  baselines?: Record<string, BaselineModel>;
  onDeleteResult?: (resultKey: string) => void;
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

function normalizeModelKey(value?: string): string {
  return (value || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/-/g, '_');
}

function formatCo2(value: number | null): string {
  if (value == null) return 'Not Available';
  if (value > 0 && value < 0.000001) return '<0.000001 kg';
  return `${value.toFixed(6)} kg`;
}

function formatEnergy(value: number | null): string {
  return value == null ? 'Not Available' : `${value.toFixed(8)} kWh`;
}

export function EnergySection({
  energy,
  savedResults = [],
  baselines = {},
  onDeleteResult,
}: EnergySectionProps) {
  const hasSavings = energy && Object.keys(energy).length > 0;
  const dedupedResults = Object.values(
    savedResults.reduce((acc, result) => {
      const key = getResultStorageKey(result);
      acc[key] = result;
      return acc;
    }, {} as Record<string, DynamicResult>)
  );

  const comparisonRows = dedupedResults
    .filter((result) => (result.strategy || '').toLowerCase() !== 'baseline')
    .map((result) => {
      const modelKey = normalizeModelKey(result.model_key || result.model_name);
      const baseline = baselines[modelKey];

      const baselineCo2 = toFiniteNumber(result.baseline_total_emissions_kg ?? baseline?.training_co2_kg);
      const compressedTrainCo2 = toFiniteNumber(result.training_co2_kg ?? result.training_emissions_kg);
      const compressedInferCo2 = toFiniteNumber(
        result.inference_co2_kg ?? result.inference_emissions_kg ?? result.emissions_kg
      );
      const compressedCo2 = toFiniteNumber(result.compressed_total_emissions_kg) ?? compressedTrainCo2 ?? compressedInferCo2;

      const reductionPercent =
        baselineCo2 != null && baselineCo2 > 0 && compressedCo2 != null
          ? ((baselineCo2 - compressedCo2) / baselineCo2) * 100
          : null;

      const suspiciousReduction =
        reductionPercent != null && reductionPercent > 80 && (result.size_reduction_percent ?? 0) < 30;

      return {
        key: getResultStorageKey(result),
        label: `${result.model_name || result.model_key || 'Model'} · ${result.compression_method || result.strategy}`,
        baselineCo2,
        compressedCo2,
        compressedTrainCo2,
        compressedInferCo2,
        reductionPercent,
        suspiciousReduction,
      };
    });

  const hasComparisons = comparisonRows.length > 0;
  const showSection = hasSavings || hasComparisons;

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
          <span className="text-green-600 text-sm">⚡</span>
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Energy & Carbon Emissions
          </h3>
          <p className="text-xs text-gray-500">Phase 7 — CodeCarbon tracking</p>
        </div>
      </div>

      {!showSection ? (
        <div className="bg-gray-50 rounded-lg p-6 text-center">
          <p className="text-sm text-gray-500 mb-2">
            No energy data available yet.
          </p>
          <p className="text-xs text-gray-400">
            Run energy tracking from the Actions panel or compress a model.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {/* Baseline vs compressed CO2 */}
          {hasComparisons && (
            <>
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                Baseline vs Compressed CO2
              </p>
              {comparisonRows.map((row) => {
                const reductionText =
                  row.reductionPercent == null
                    ? 'Not Available'
                    : `${row.reductionPercent.toFixed(2)}%`;

                const reductionTone =
                  row.reductionPercent == null
                    ? 'text-gray-500'
                    : row.reductionPercent > 0
                      ? 'text-green-600'
                      : row.reductionPercent < 0
                        ? 'text-red-600'
                        : 'text-gray-600';

                return (
                  <div key={row.key} className="bg-gray-50 rounded-lg p-4 border border-gray-100">
                    <div className="flex items-start justify-between gap-3 mb-3">
                      <p className="text-sm font-medium text-gray-700">{row.label}</p>
                      <div className="flex items-start gap-2">
                        <div className="text-right">
                          <p className={`text-lg font-bold ${reductionTone}`}>{reductionText}</p>
                          <p className="text-xs text-gray-500">CO2 reduction</p>
                        </div>
                        {onDeleteResult && (
                          <button
                            type="button"
                            onClick={() => onDeleteResult(row.key)}
                            className="inline-flex items-center justify-center w-7 h-7 rounded-md border border-red-200 text-red-600 hover:bg-red-50 transition-colors"
                            title="Delete this saved result"
                            aria-label="Delete this saved result"
                          >
                            ×
                          </button>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      <div className="rounded-lg border border-gray-200 bg-white p-3">
                        <p className="text-[11px] uppercase tracking-wide text-gray-500">Baseline CO2</p>
                        <p className="text-sm font-semibold text-gray-800 mt-1">{formatCo2(row.baselineCo2)}</p>
                      </div>

                      <div className="rounded-lg border border-gray-200 bg-white p-3">
                        <p className="text-[11px] uppercase tracking-wide text-gray-500">Compressed CO2</p>
                        <p className="text-sm font-semibold text-gray-800 mt-1">{formatCo2(row.compressedCo2)}</p>
                      </div>

                      <div className="rounded-lg border border-gray-200 bg-white p-3">
                        <p className="text-[11px] uppercase tracking-wide text-gray-500">Compressed Train / Infer</p>
                        <p className="text-sm font-semibold text-gray-800 mt-1">
                          {`${formatCo2(row.compressedTrainCo2)} / ${formatCo2(row.compressedInferCo2)}`}
                        </p>
                      </div>
                    </div>

                    <div>
                      <p className="text-[11px] text-gray-500 mt-2">
                        Fair comparison prefers total benchmark CO2 when available; otherwise it falls back to train/infer fields.
                      </p>
                      {row.suspiciousReduction && (
                        <p className="text-[11px] text-amber-700 mt-1">
                          Warning: very high CO2 reduction with small size reduction. Re-check tracking workload parity.
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </>
          )}

          {/* Original energy tracking data */}
          {hasSavings && (
            <>
              {hasComparisons && (
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mt-4">
                  Baseline Energy Tracking
                </p>
              )}
              {Object.entries(energy).map(([key, data]: [string, any]) => {
                if (key === 'training_compact_vs_baseline') {
                  return (
                    <div key={key} className="bg-green-50 rounded-lg p-4 border border-green-100">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-green-800">
                            Training Energy Savings
                          </p>
                          <p className="text-xs text-green-600 mt-0.5">
                            Compact Student vs ResNet18 Baseline
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-2xl font-bold text-green-700">
                            {data.energy_saving_percent}%
                          </p>
                          <p className="text-xs text-green-600">less energy</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4 mt-3 pt-3 border-t border-green-200">
                        <div>
                          <p className="text-xs text-green-600">Baseline</p>
                          <p className="text-sm font-mono text-green-800">
                            {formatEnergy(toFiniteNumber(data.baseline_energy_kWh))}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-green-600">Student</p>
                          <p className="text-sm font-mono text-green-800">
                            {formatEnergy(toFiniteNumber(data.student_energy_kWh))}
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                }

                return (
                  <div key={key} className="bg-gray-50 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-700">
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {`Energy: ${formatEnergy(toFiniteNumber(data.inference_energy_kWh))}`}
                      </p>
                    </div>
                    <div className="text-right">
                      <span className={`text-lg font-bold ${
                        data.energy_saving_percent > 0 ? 'text-green-600' : 'text-gray-600'
                      }`}>
                        {data.energy_saving_percent !== undefined
                          ? `${data.energy_saving_percent}%`
                          : '—'}
                      </span>
                      {data.energy_saving_percent > 0 && (
                        <p className="text-xs text-green-600">saved</p>
                      )}
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      )}
    </div>
  );
}
