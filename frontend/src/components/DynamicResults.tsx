'use client';

import { useState } from 'react';
import { DynamicResult } from '@/lib/api';

interface DynamicResultsProps {
  results: DynamicResult[];
  onClear: () => void;
}

const STRATEGY_LABELS: Record<string, string> = {
  pruning: 'Pruning',
  quantization: 'Quantization',
  hybrid: 'Hybrid',
  kd: 'Knowledge Distillation',
};

function formatCo2(value: number): string {
  if (!Number.isFinite(value)) return 'Not Available';
  if (value > 0 && value < 0.000001) return '<0.000001 kg';
  return `${value.toFixed(6)} kg`;
}

export function DynamicResults({ results, onClear }: DynamicResultsProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(
    results.length > 0 ? results.length - 1 : null
  );

  if (results.length === 0) return null;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Compression Results
          <span className="ml-2 text-sm font-normal text-gray-400">
            ({results.length} saved)
          </span>
        </h3>
        <button
          onClick={onClear}
          className="text-xs text-red-500 hover:text-red-700 px-2 py-1 rounded hover:bg-red-50 transition-colors"
        >
          Clear All
        </button>
      </div>

      <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
        {[...results].reverse().map((result, reverseIdx) => {
          const idx = results.length - 1 - reverseIdx;
          const isExpanded = expandedIdx === idx;
          const sizeReduction = result.size_reduction_percent;
          const accDiff = result.compressed_accuracy - result.baseline_accuracy;
          const modelLabel = result.model_name || result.model_key || 'Model';
          const methodLabel = STRATEGY_LABELS[result.strategy] || result.strategy;
          const trainingCo2 = result.training_co2_kg ?? result.training_emissions_kg;
          const inferenceCo2 = result.inference_co2_kg ?? result.inference_emissions_kg ?? result.emissions_kg;
          const baselineTotalCo2 = result.baseline_total_emissions_kg;
          const compressedTotalCo2 = result.compressed_total_emissions_kg ?? result.co2_kg ?? result.emissions_kg;
          const emissionsReduction =
            result.emissions_reduction_percent ??
            (baselineTotalCo2 && baselineTotalCo2 > 0 && compressedTotalCo2 != null
              ? ((baselineTotalCo2 - compressedTotalCo2) / baselineTotalCo2) * 100
              : null);
          const baselineLatency = result.baseline_latency_ms;
          const compressedLatency = result.compressed_latency_ms ?? result.latency_ms;
          const latencySpeedup =
            result.latency_speedup_percent ??
            (baselineLatency && baselineLatency > 0 && compressedLatency != null
              ? ((baselineLatency - compressedLatency) / baselineLatency) * 100
              : null);

          return (
            <div key={idx} className="border border-gray-200 rounded-lg overflow-hidden">
              {/* Header — always visible */}
              <button
                onClick={() => setExpandedIdx(isExpanded ? null : idx)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors text-left"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <span className={`inline-flex px-2 py-0.5 rounded text-xs font-medium ${
                    sizeReduction > 50 ? 'bg-green-100 text-green-700' :
                    sizeReduction > 20 ? 'bg-blue-100 text-blue-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {methodLabel}
                  </span>
                  <span className="text-sm font-medium text-gray-900 truncate">
                    {modelLabel}
                  </span>
                </div>
                <div className="flex items-center gap-4 flex-shrink-0">
                  <span className="text-sm font-mono text-green-600">
                    ↓{sizeReduction.toFixed(1)}%
                  </span>
                  <span className={`text-sm font-mono ${accDiff >= -1 ? 'text-blue-600' : 'text-red-500'}`}>
                    {result.compressed_accuracy}%
                  </span>
                  <svg
                    className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none" viewBox="0 0 24 24" stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>

              {/* Expanded details */}
              {isExpanded && (
                <div className="px-4 pb-4 border-t border-gray-100">
                  {/* Key Metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3 mb-4">
                    <MiniMetric label="Baseline Acc." value={`${result.baseline_accuracy}%`} />
                    <MiniMetric
                      label="Compressed Acc."
                      value={`${result.compressed_accuracy}%`}
                      accent={accDiff >= -1}
                    />
                    <MiniMetric
                      label="Size"
                      value={`${result.size_MB} MB`}
                      sub={`from ${result.baseline_size_MB} MB`}
                    />
                    <MiniMetric
                      label="Latency"
                      value={compressedLatency != null ? `${compressedLatency.toFixed(2)} ms` : 'Not Available'}
                      sub={
                        baselineLatency != null
                          ? `baseline ${baselineLatency.toFixed(2)} ms`
                          : undefined
                      }
                    />
                  </div>

                  {/* Size comparison bar */}
                  <div className="mb-4">
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                      <span>Baseline: {result.baseline_size_MB} MB</span>
                      <span>Compressed: {result.size_MB} MB</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div
                        className="bg-green-500 h-2.5 rounded-full transition-all duration-500"
                        style={{
                          width: `${Math.max(5, (result.size_MB / result.baseline_size_MB) * 100)}%`,
                        }}
                      />
                    </div>
                  </div>

                  {/* Detail grid */}
                  <div className="bg-gray-50 rounded-lg p-3">
                    <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs">
                      {result.total_params != null && (
                        <DetailRow label="Total Params" value={result.total_params.toLocaleString()} />
                      )}
                      {result.nonzero_params != null && (
                        <DetailRow label="Non-zero" value={result.nonzero_params.toLocaleString()} />
                      )}
                      {result.sparsity_percent != null && result.sparsity_percent > 0 && (
                        <DetailRow label="Sparsity" value={`${result.sparsity_percent}%`} />
                      )}
                      {result.pruning_amount != null && (
                        <DetailRow label="Pruning" value={`${(result.pruning_amount * 100).toFixed(0)}%`} />
                      )}
                      {result.quantization_type && (
                        <DetailRow label="Quantization" value={result.quantization_type} />
                      )}
                      {result.flops_M != null && result.flops_M > 0 && (
                        <DetailRow label="FLOPs" value={`${result.flops_M} M`} />
                      )}
                      {trainingCo2 != null && (
                        <DetailRow label="Train CO₂" value={formatCo2(trainingCo2)} />
                      )}
                      {inferenceCo2 != null && (
                        <DetailRow label="Infer CO₂" value={formatCo2(inferenceCo2)} />
                      )}
                      {baselineTotalCo2 != null && (
                        <DetailRow label="Baseline Total CO₂" value={formatCo2(baselineTotalCo2)} />
                      )}
                      {compressedTotalCo2 != null && (
                        <DetailRow label="Compressed Total CO₂" value={formatCo2(compressedTotalCo2)} />
                      )}
                      {emissionsReduction != null && (
                        <DetailRow label="CO₂ Reduction" value={`${emissionsReduction.toFixed(2)}%`} />
                      )}
                      {latencySpeedup != null && (
                        <DetailRow label="Latency Speedup" value={`${latencySpeedup.toFixed(2)}%`} />
                      )}
                      {result.training_energy_kwh != null && (
                        <DetailRow label="Train Energy" value={`${result.training_energy_kwh.toFixed(8)} kWh`} />
                      )}
                      {(result.inference_energy_kwh ?? result.energy_kwh) != null && (
                        <DetailRow
                          label="Infer Energy"
                          value={`${(result.inference_energy_kwh ?? result.energy_kwh ?? 0).toFixed(8)} kWh`}
                        />
                      )}
                      {result.student_params != null && (
                        <DetailRow label="Student Params" value={result.student_params.toLocaleString()} />
                      )}
                      {result.teacher_params != null && (
                        <DetailRow label="Teacher Params" value={result.teacher_params.toLocaleString()} />
                      )}
                      {result.param_reduction_percent != null && (
                        <DetailRow label="Param ↓" value={`${result.param_reduction_percent}%`} />
                      )}
                      {result.dataset && (
                        <DetailRow label="Dataset" value={result.dataset} />
                      )}
                      {result.pipeline && (
                        <div className="col-span-2">
                          <DetailRow label="Pipeline" value={result.pipeline} />
                        </div>
                      )}
                      {Array.isArray(result.sanity_warnings) && result.sanity_warnings.length > 0 && (
                        <div className="col-span-2 mt-1 rounded border border-amber-200 bg-amber-50 p-2">
                          <p className="text-[11px] font-semibold text-amber-800">Sanity warnings</p>
                          {result.sanity_warnings.map((warning, warningIdx) => (
                            <p key={`warn-${warningIdx}`} className="text-[11px] text-amber-700 mt-0.5">
                              • {warning}
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MiniMetric({
  label,
  value,
  sub,
  accent = true,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div className="bg-gray-50 rounded-lg p-2 text-center">
      <p className="text-[10px] text-gray-500">{label}</p>
      <p className={`text-sm font-bold ${accent ? 'text-green-700' : 'text-red-600'}`}>
        {value}
      </p>
      {sub && <p className="text-[10px] text-gray-400">{sub}</p>}
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-900 font-medium text-right">{value}</span>
    </>
  );
}
