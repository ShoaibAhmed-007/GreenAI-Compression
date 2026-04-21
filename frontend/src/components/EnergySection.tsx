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

function straightLabel(value: string): string {
  return value
    .replace(/\s*·\s*/g, ' - ')
    .replace(/\s+/g, ' ')
    .trim();
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
      const rawResult = result as unknown as Record<string, unknown>;
      const modelKey = normalizeModelKey(result.model_key || result.model_name);
      const baseline = baselines[modelKey];

      const baselineCo2 = toFiniteNumber(
        result.baseline_total_emissions_kg ??
          result.baseline_training_co2_kg ??
          baseline?.training_co2_kg ??
          rawResult.baseline_co2_kg ??
          rawResult.baseline_emissions_kg
      );
      const compressedTrainCo2 = toFiniteNumber(result.training_co2_kg ?? result.training_emissions_kg);
      const compressedInferCo2 = toFiniteNumber(
        result.inference_co2_kg ?? result.inference_emissions_kg ?? result.emissions_kg
      );
      const compressedCo2 = toFiniteNumber(
        result.compressed_total_emissions_kg ??
          rawResult.compressed_training_co2_kg ??
          rawResult.compressed_co2_kg ??
          rawResult.compressed_emissions_kg
      ) ?? compressedInferCo2 ?? compressedTrainCo2;

      const reductionPercent =
        baselineCo2 != null && baselineCo2 > 0 && compressedCo2 != null
          ? ((baselineCo2 - compressedCo2) / baselineCo2) * 100
          : null;

      const suspiciousReduction =
        reductionPercent != null && reductionPercent > 80 && (result.size_reduction_percent ?? 0) < 30;

      return {
        key: getResultStorageKey(result),
        label: `${straightLabel(result.model_name || result.model_key || 'Model')} - ${straightLabel(result.compression_method || result.strategy || 'Compression')}`,
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
    <div className="bg-surface-container p-6 rounded-2xl space-y-6">
      <div className="flex items-center gap-3">
        <span className="material-symbols-outlined text-secondary">energy_savings_leaf</span>
        <div>
          <h3 className="text-lg font-headline font-semibold text-on-surface">
            Sustainability Impact
          </h3>
          <p className="text-xs text-on-surface-variant/60">Phase 7 — CodeCarbon tracking</p>
        </div>
      </div>

      {!showSection ? (
        <div className="bg-surface-container-low rounded-xl p-6 text-center">
          <p className="text-sm text-on-surface-variant mb-2">
            No energy data available yet.
          </p>
          <p className="text-xs text-on-surface-variant/50">
            Run energy tracking from the Actions panel or compress a model.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Baseline vs compressed CO2 */}
          {hasComparisons && (
            <>
              <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">
                Baseline vs Compressed CO2
              </p>
              {comparisonRows.map((row) => {
                const reductionText =
                  row.reductionPercent == null
                    ? 'Not Available'
                    : `${row.reductionPercent.toFixed(2)}%`;

                const reductionTone =
                  row.reductionPercent == null
                    ? 'text-on-surface-variant'
                    : row.reductionPercent > 0
                      ? 'text-primary'
                      : row.reductionPercent < 0
                        ? 'text-error'
                        : 'text-on-surface-variant';

                return (
                  <div key={row.key} className="bg-surface-container-low rounded-xl p-4 space-y-3">
                    <div className="flex items-start justify-between gap-3">
                      <p className="text-sm font-medium text-on-surface">{row.label}</p>
                      <div className="flex items-start gap-2">
                        <div className="text-right">
                          <p className={`text-lg font-bold font-technical ${reductionTone}`}>{reductionText}</p>
                          <p className="text-xs text-on-surface-variant/50">CO2 reduction</p>
                        </div>
                        {onDeleteResult && (
                          <button
                            type="button"
                            onClick={() => onDeleteResult(row.key)}
                            className="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-error-container/10 text-error hover:bg-error-container/20 transition-colors"
                            title="Delete this saved result"
                            aria-label="Delete this saved result"
                          >
                            <span className="material-symbols-outlined text-sm">close</span>
                          </button>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div className="bg-surface-container-lowest rounded-xl p-3" style={{ borderLeft: '4px solid #ffb4ab' }}>
                        <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Baseline CO2</p>
                        <p className="text-sm font-technical text-on-surface mt-1">{formatCo2(row.baselineCo2)}</p>
                      </div>

                      <div className="bg-surface-container-lowest rounded-xl p-3" style={{ borderLeft: '4px solid #5bdda8' }}>
                        <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Compressed CO2</p>
                        <p className="text-sm font-technical text-primary mt-1">{formatCo2(row.compressedCo2)}</p>
                      </div>

                      <div className="bg-surface-container-lowest rounded-xl p-3">
                        <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Train / Infer</p>
                        <p className="text-sm font-technical text-on-surface mt-1">
                          {`${formatCo2(row.compressedTrainCo2)} / ${formatCo2(row.compressedInferCo2)}`}
                        </p>
                      </div>
                    </div>

                    <div>
                      <p className="text-[10px] text-on-surface-variant/50">
                        Fair comparison prefers total benchmark CO2 when available; otherwise it falls back to train/infer fields.
                      </p>
                      {row.suspiciousReduction && (
                        <div className="flex items-start gap-2 mt-2 bg-error-container/10 p-2.5 rounded-lg">
                          <span className="material-symbols-outlined text-error text-sm">warning</span>
                          <p className="text-[10px] text-on-error-container">
                            Warning: very high CO2 reduction with small size reduction. Re-check tracking workload parity.
                          </p>
                        </div>
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
                <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mt-4">
                  Baseline Energy Tracking
                </p>
              )}
              {Object.entries(energy).map(([key, data]: [string, any]) => {
                if (key === 'training_compact_vs_baseline') {
                  return (
                    <div key={key} className="bg-primary/10 rounded-xl p-5">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-primary">
                            Training Energy Savings
                          </p>
                          <p className="text-xs text-primary/60 mt-0.5">
                            Compact Student vs ResNet18 Baseline
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-4xl font-headline font-bold text-primary">
                            {data.energy_saving_percent}%
                          </p>
                          <p className="text-xs text-primary/60">less energy</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4 mt-4 pt-4" style={{ borderTop: '1px solid rgba(91, 221, 168, 0.15)' }}>
                        <div>
                          <p className="text-xs text-primary/60 font-technical">Baseline</p>
                          <p className="text-sm font-technical text-primary">
                            {formatEnergy(toFiniteNumber(data.baseline_energy_kWh))}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-primary/60 font-technical">Student</p>
                          <p className="text-sm font-technical text-primary">
                            {formatEnergy(toFiniteNumber(data.student_energy_kWh))}
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                }

                return (
                  <div key={key} className="bg-surface-container-low rounded-xl p-4 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-on-surface">
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </p>
                      <p className="text-xs text-on-surface-variant/60 mt-0.5 font-technical">
                        {`Energy: ${formatEnergy(toFiniteNumber(data.inference_energy_kWh))}`}
                      </p>
                    </div>
                    <div className="text-right">
                      <span className={`text-lg font-bold font-technical ${
                        data.energy_saving_percent > 0 ? 'text-primary' : 'text-on-surface-variant'
                      }`}>
                        {data.energy_saving_percent !== undefined
                          ? `${data.energy_saving_percent}%`
                          : '—'}
                      </span>
                      {data.energy_saving_percent > 0 && (
                        <p className="text-xs text-primary/60">saved</p>
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
