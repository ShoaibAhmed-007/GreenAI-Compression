'use client';

import { Strategy } from '@/lib/api';

interface ComparisonTableProps {
  strategies: Strategy[];
  onDeleteStrategy?: (strategyKey: string) => void;
}

function formatCo2(value: number | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'Not Available';
  if (value > 0 && value < 0.000001) return '<0.000001 kg';
  return `${value.toFixed(6)} kg`;
}

function formatPercent(value: number | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'Not Available';
  return `${value.toFixed(2)}%`;
}

function formatSize(value: number | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'Not Available';
  return `${value.toFixed(2)} MB`;
}

function straightLabel(value: string): string {
  return value
    .replace(/\s*·\s*/g, ' - ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function ComparisonTable({ strategies, onDeleteStrategy }: ComparisonTableProps) {
  if (!strategies.length) {
    return (
      <div className="card">
        <h3 className="text-lg font-headline font-semibold text-on-surface mb-4">Strategy Comparison</h3>
        <div className="h-60 flex items-center justify-center">
          <div className="text-center">
            <div className="w-12 h-12 bg-surface-container-high rounded-lg flex items-center justify-center mx-auto mb-3">
              <span className="material-symbols-outlined text-on-surface-variant/40">assignment</span>
            </div>
            <p className="text-sm text-on-surface-variant">No results yet</p>
            <p className="text-xs text-on-surface-variant/50 mt-1">
              All compression results will accumulate here
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden">
      <h3 className="text-lg font-headline font-semibold text-on-surface mb-4">Strategy Comparison</h3>
      <div className="overflow-x-auto -mx-6 -mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-surface-container-low">
              <th className="text-left py-3 px-4 font-bold text-[10px] text-on-surface-variant uppercase tracking-widest">Strategy</th>
              <th className="text-right py-3 px-4 font-bold text-[10px] text-on-surface-variant uppercase tracking-widest">Acc%</th>
              <th className="text-right py-3 px-4 font-bold text-[10px] text-on-surface-variant uppercase tracking-widest">Size</th>
              <th className="text-right py-3 px-4 font-bold text-[10px] text-on-surface-variant uppercase tracking-widest">↓ Size</th>
              {/* <th className="text-right py-3 px-4 font-bold text-[10px] text-on-surface-variant uppercase tracking-widest">Latency</th> */}
              <th className="text-right py-3 px-4 font-bold text-[10px] text-on-surface-variant uppercase tracking-widest">Train CO₂</th>
              <th className="text-right py-3 px-4 font-bold text-[10px] text-on-surface-variant uppercase tracking-widest">Infer CO₂</th>
              {onDeleteStrategy && (
                <th className="text-center py-3 px-4 font-bold text-[10px] text-on-surface-variant uppercase tracking-widest">Del</th>
              )}
            </tr>
          </thead>
          <tbody>
            {strategies.map((s, i) => {
              const isBaseline = s.key === 'baseline';
              const isBest = !isBaseline && s.size_reduction ===
                Math.max(...strategies.filter(x => x.key !== 'baseline').map(x => x.size_reduction));

              return (
                <tr
                  key={s.key}
                  className={`transition-colors ${
                    i % 2 === 0 ? 'bg-surface-container' : 'bg-surface-container-low'
                  } ${isBaseline ? '' : isBest ? 'ring-1 ring-inset ring-primary/20' : ''} hover:bg-surface-container-high`}
                >
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      {isBest && <span className="badge-green">Best</span>}
                      {isBaseline && <span className="badge-blue">Base</span>}
                      <span className={`font-medium ${isBaseline ? 'text-on-surface-variant' : 'text-on-surface'}`}>
                        {straightLabel(s.name)}
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right font-technical">
                    <span className={
                      s.accuracy >= (strategies[0]?.accuracy || 0)
                        ? 'text-primary' : 'text-secondary'
                    }>
                      {formatPercent(s.accuracy)}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right font-technical text-on-surface">
                    {formatSize(s.size_MB)}
                  </td>
                  <td className="py-3 px-4 text-right font-technical">
                    {isBaseline ? (
                      <span className="text-on-surface-variant/40">—</span>
                    ) : (
                      <span className={s.size_reduction > 0 ? 'text-primary' : 'text-error'}>
                        {s.size_reduction > 0 ? '↓' : '↑'} {Math.abs(s.size_reduction).toFixed(2)}%
                      </span>
                    )}
                  </td>
                  {/* <td className="py-3 px-4 text-right font-technical text-on-surface">
                    {s.latency_ms} ms
                  </td> */}
                  <td className="py-3 px-4 text-right font-technical text-on-surface-variant">
                    {formatCo2(s.training_co2_kg)}
                  </td>
                  <td className="py-3 px-4 text-right font-technical text-on-surface-variant">
                    {s.inference_co2_kg != null
                      ? formatCo2(s.inference_co2_kg)
                      : s.co2_kg != null
                        ? formatCo2(s.co2_kg)
                        : 'Not Available'}
                  </td>
                  {onDeleteStrategy && (
                    <td className="py-3 px-4 text-center">
                      <button
                        type="button"
                        onClick={() => onDeleteStrategy(s.key)}
                        className="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-error-container/10 text-error hover:bg-error-container/20 transition-colors"
                        title="Delete this saved result"
                        aria-label="Delete this saved result"
                      >
                        <span className="material-symbols-outlined text-sm">close</span>
                      </button>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
