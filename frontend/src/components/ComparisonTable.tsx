'use client';

import { Strategy } from '@/lib/api';

interface ComparisonTableProps {
  strategies: Strategy[];
}

export function ComparisonTable({ strategies }: ComparisonTableProps) {
  if (!strategies.length) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Strategy Comparison</h3>
        <div className="h-60 flex items-center justify-center">
          <div className="text-center">
            <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-3">
              <span className="text-gray-400 text-xl">📋</span>
            </div>
            <p className="text-sm text-gray-500">No results yet</p>
            <p className="text-xs text-gray-400 mt-1">
              All compression results will accumulate here
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Strategy Comparison</h3>
      <div className="overflow-x-auto -mx-6 -mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-y border-gray-100">
              <th className="text-left py-3 px-4 font-medium text-gray-600">Strategy</th>
              <th className="text-right py-3 px-4 font-medium text-gray-600">Acc%</th>
              <th className="text-right py-3 px-4 font-medium text-gray-600">Size</th>
              <th className="text-right py-3 px-4 font-medium text-gray-600">↓ Size</th>
              <th className="text-right py-3 px-4 font-medium text-gray-600">Latency</th>
              <th className="text-right py-3 px-4 font-medium text-gray-600">Train CO₂</th>
              <th className="text-right py-3 px-4 font-medium text-gray-600">Infer CO₂</th>
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
                  className={`border-b border-gray-50 ${
                    isBaseline ? 'bg-blue-50/50' : isBest ? 'bg-green-50/50' : ''
                  } hover:bg-gray-50/80 transition-colors`}
                >
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      {isBest && <span className="badge-green">Best</span>}
                      {isBaseline && <span className="badge-blue">Base</span>}
                      <span className={`font-medium ${isBaseline ? 'text-gray-600' : 'text-gray-900'}`}>
                        {s.name}
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right font-mono">
                    <span className={
                      s.accuracy >= (strategies[0]?.accuracy || 0)
                        ? 'text-green-600' : 'text-amber-600'
                    }>
                      {s.accuracy}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-gray-700">
                    {s.size_MB} MB
                  </td>
                  <td className="py-3 px-4 text-right font-mono">
                    {isBaseline ? (
                      <span className="text-gray-400">—</span>
                    ) : (
                      <span className={s.size_reduction > 0 ? 'text-green-600' : 'text-red-500'}>
                        {s.size_reduction > 0 ? '↓' : '↑'} {Math.abs(s.size_reduction)}%
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-gray-700">
                    {s.latency_ms} ms
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-gray-700">
                    {s.training_co2_kg != null ? `${s.training_co2_kg.toFixed(6)} kg` : '—'}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-gray-700">
                    {s.inference_co2_kg != null
                      ? `${s.inference_co2_kg.toFixed(6)} kg`
                      : s.co2_kg != null
                        ? `${s.co2_kg.toFixed(6)} kg`
                        : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
