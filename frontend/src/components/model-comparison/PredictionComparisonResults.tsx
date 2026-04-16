'use client';

import { CompareImageResponse } from '@/lib/api';

interface PredictionComparisonResultsProps {
  result: CompareImageResponse;
}

function formatPercent(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Not Available';
  return `${value.toFixed(2)}%`;
}

function tone(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'text-on-surface-variant';
  if (value > 0) return 'text-primary';
  if (value < 0) return 'text-secondary';
  return 'text-on-surface-variant';
}

export default function PredictionComparisonResults({
  result,
}: PredictionComparisonResultsProps) {
  const hasMismatch = result.comparison?.prediction_mismatch_warning;

  return (
    <div className="bg-surface-container rounded-xl p-8 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-headline font-bold text-on-surface">Inference Results</h2>
        {hasMismatch && (
          <div className="flex items-center gap-2 bg-error-container/20 text-error px-3 py-1.5 rounded-lg ghost-border">
            <span className="material-symbols-outlined text-[18px]">warning</span>
            <span className="text-xs font-semibold font-technical">PREDICTION MISMATCH</span>
          </div>
        )}
      </div>

      {hasMismatch && (
        <div className="bg-error-container/10 rounded-xl p-3 ghost-border">
          <p className="text-xs text-on-error-container">{result.comparison!.prediction_mismatch_warning}</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Baseline Predictions */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-technical text-on-surface-variant uppercase tracking-widest">
              Baseline ({result.baseline.model_key})
            </span>
          </div>
          <div className="bg-surface-container-low rounded-xl p-6 space-y-4">
            {result.baseline.top3.map((item, idx) => (
              <div key={`base-top-${item.class_index}`} className="space-y-1">
                <div className="flex justify-between items-end">
                  <span className={`capitalize ${idx === 0 ? 'font-bold text-lg text-on-surface' : 'text-sm font-medium text-on-surface/70'}`}>
                    {item.class_name}
                  </span>
                  <span className={`font-technical ${idx === 0 ? 'text-primary' : 'text-on-surface/70 text-sm'}`}>
                    {formatPercent(item.probability)}
                  </span>
                </div>
                <div className={`w-full ${idx === 0 ? 'h-2' : 'h-1.5'} bg-surface-variant rounded-full overflow-hidden`}>
                  <div
                    className={`h-full rounded-full ${idx === 0 ? 'bg-primary' : 'bg-primary/40'}`}
                    style={{ width: `${item.probability || 0}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Compressed Predictions */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-technical text-on-surface-variant uppercase tracking-widest">
              Compressed ({result.compressed.model_key} - {result.compressed.strategy})
            </span>
          </div>
          <div className={`bg-surface-container-low rounded-xl p-6 space-y-4 ${
            hasMismatch ? 'ring-2 ring-error/10 ghost-border' : ''
          }`}>
            {result.compressed.top3.map((item, idx) => (
              <div key={`comp-top-${item.class_index}`} className="space-y-1">
                <div className="flex justify-between items-end">
                  <span className={`capitalize ${idx === 0 ? `font-bold text-lg ${hasMismatch ? 'text-error' : 'text-on-surface'}` : 'text-sm font-medium text-on-surface/70'}`}>
                    {item.class_name}
                  </span>
                  <span className={`font-technical ${idx === 0 ? (hasMismatch ? 'text-error' : 'text-secondary') : 'text-on-surface/70 text-sm'}`}>
                    {formatPercent(item.probability)}
                  </span>
                </div>
                <div className={`w-full ${idx === 0 ? 'h-2' : 'h-1.5'} bg-surface-variant rounded-full overflow-hidden`}>
                  <div
                    className={`h-full rounded-full ${idx === 0 ? (hasMismatch ? 'bg-error' : 'bg-secondary') : 'bg-secondary/40'}`}
                    style={{ width: `${item.probability || 0}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="bg-surface-container-low rounded-xl p-4">
          <p className="text-[10px] text-on-surface-variant uppercase tracking-wider font-technical">Prediction Match</p>
          <p className={`text-lg font-bold mt-1 ${result.comparison?.prediction_match ? 'text-primary' : 'text-error'}`}>
            {result.comparison?.prediction_match ? 'Matched' : 'Different'}
          </p>
        </div>
        <div className="bg-surface-container-low rounded-xl p-4">
          <p className="text-[10px] text-on-surface-variant uppercase tracking-wider font-technical">Confidence Delta</p>
          <p className={`text-lg font-bold font-technical mt-1 ${tone(result.comparison?.confidence_delta_percent)}`}>
            {formatPercent(result.comparison?.confidence_delta_percent)}
          </p>
        </div>
        <div className="bg-surface-container-low rounded-xl p-4">
          <p className="text-[10px] text-on-surface-variant uppercase tracking-wider font-technical">Input Type</p>
          <p className="text-lg font-bold mt-1 text-on-surface capitalize">{result.input.source}</p>
        </div>
      </div>

      {result.diagnostics?.quality_warnings && result.diagnostics.quality_warnings.length > 0 && (
        <div className="bg-error-container/10 rounded-xl p-4 ghost-border">
          <div className="flex items-center gap-2 mb-2">
            <span className="material-symbols-outlined text-error text-sm">warning</span>
            <p className="text-sm font-semibold text-on-error-container">Preprocessing warnings</p>
          </div>
          <ul className="text-xs text-on-error-container/80 list-disc list-inside space-y-0.5">
            {result.diagnostics.quality_warnings.map((warning, idx) => (
              <li key={`warning-${idx}`}>{warning}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
