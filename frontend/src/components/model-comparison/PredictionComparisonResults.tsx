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
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'text-gray-700';
  if (value > 0) return 'text-green-700';
  if (value < 0) return 'text-amber-700';
  return 'text-gray-700';
}

export default function PredictionComparisonResults({
  result,
}: PredictionComparisonResultsProps) {
  return (
    <div className="card space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">Inference Results</h3>

      {result.comparison?.prediction_mismatch_warning && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-3">
          <p className="text-sm font-semibold text-amber-900">Prediction mismatch detected</p>
          <p className="text-xs text-amber-800 mt-1">{result.comparison.prediction_mismatch_warning}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-lg border border-blue-200 p-4 bg-blue-50">
          <p className="text-xs uppercase tracking-wide text-blue-700">Baseline Model</p>
          <h4 className="text-base font-semibold text-blue-900 mt-1">{result.baseline.model_key}</h4>
          <div className="mt-3 space-y-1 text-sm text-blue-900">
            <p>
              Predicted class: <span className="font-semibold capitalize">{result.baseline.class}</span>
            </p>
            <p>
              Confidence: <span className="font-semibold">{formatPercent(result.baseline.confidence)}</span>
            </p>
          </div>
          <div className="mt-2">
            <p className="text-xs font-semibold text-blue-800">Top-3 predictions</p>
            <ul className="text-xs text-blue-900 mt-1 space-y-0.5">
              {result.baseline.top3.map((item) => (
                <li key={`base-top-${item.class_index}`} className="flex justify-between gap-2">
                  <span className="capitalize">{item.class_name}</span>
                  <span className="font-mono">{formatPercent(item.probability)}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="rounded-lg border border-green-200 p-4 bg-green-50">
          <p className="text-xs uppercase tracking-wide text-green-700">Compressed Model</p>
          <h4 className="text-base font-semibold text-green-900 mt-1">
            {result.compressed.model_key} - {result.compressed.strategy}
          </h4>
          <div className="mt-3 space-y-1 text-sm text-green-900">
            <p>
              Predicted class: <span className="font-semibold capitalize">{result.compressed.class}</span>
            </p>
            <p>
              Confidence: <span className="font-semibold">{formatPercent(result.compressed.confidence)}</span>
            </p>
          </div>
          <div className="mt-2">
            <p className="text-xs font-semibold text-green-800">Top-3 predictions</p>
            <ul className="text-xs text-green-900 mt-1 space-y-0.5">
              {result.compressed.top3.map((item) => (
                <li key={`comp-top-${item.class_index}`} className="flex justify-between gap-2">
                  <span className="capitalize">{item.class_name}</span>
                  <span className="font-mono">{formatPercent(item.probability)}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="rounded-lg border border-gray-200 p-3 bg-white">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Prediction Match</p>
          <p className="text-lg font-bold mt-1 text-gray-900">
            {result.comparison?.prediction_match ? 'Matched' : 'Different'}
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 p-3 bg-white">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Confidence Delta</p>
          <p className={`text-lg font-bold mt-1 ${tone(result.comparison?.confidence_delta_percent)}`}>
            {formatPercent(result.comparison?.confidence_delta_percent)}
          </p>
        </div>
        <div className="rounded-lg border border-gray-200 p-3 bg-white">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Input Type</p>
          <p className="text-lg font-bold mt-1 text-gray-900 capitalize">{result.input.source}</p>
        </div>
      </div>

      {result.diagnostics?.quality_warnings && result.diagnostics.quality_warnings.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
          <p className="text-sm font-semibold text-amber-900">Preprocessing warnings</p>
          <ul className="text-xs text-amber-800 mt-1 list-disc list-inside space-y-0.5">
            {result.diagnostics.quality_warnings.map((warning, idx) => (
              <li key={`warning-${idx}`}>{warning}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
