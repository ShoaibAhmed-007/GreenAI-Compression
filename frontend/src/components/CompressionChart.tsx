'use client';

import { Strategy } from '@/lib/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, LabelList,
} from 'recharts';
import { useEffect, useState } from 'react';

interface ChartProps {
  strategies: Strategy[];
  modelName?: string;
}

type ChartView = 'size' | 'carbon' | 'radar';

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

function strategyBaselineCo2(strategy: Strategy, fallbackBaseline: number | null): number | null {
  return toFiniteNumber(
    strategy.baseline_co2_kg ?? strategy.training_co2_kg ?? fallbackBaseline
  );
}

function strategyCompressedCo2(strategy: Strategy): number | null {
  return toFiniteNumber(
    strategy.compressed_co2_kg ?? strategy.co2_kg ?? strategy.training_co2_kg ?? strategy.inference_co2_kg
  );
}

function reductionPercent(baseline: number | null, compressed: number | null): number | null {
  if (baseline == null || compressed == null || baseline <= 0) return null;
  return ((baseline - compressed) / baseline) * 100;
}

function formatCo2Value(value: unknown): string {
  const numeric = toFiniteNumber(value);
  if (numeric == null) return 'Not Available';
  if (numeric > 0 && numeric < 0.000001) return '<0.000001 kg';
  return `${numeric.toFixed(6)} kg`;
}

export function CompressionChart({ strategies, modelName }: ChartProps) {
  const [view, setView] = useState<ChartView>('size');

  useEffect(() => {
    if (!strategies.length) return;

    const baseline = strategies.find(s => s.key === 'baseline');
    const baselineCo2 = baseline
      ? toFiniteNumber(baseline.co2_kg ?? baseline.training_co2_kg ?? baseline.baseline_co2_kg)
      : null;
    
    const compressedOnly = strategies.filter(s => s.key !== 'baseline');
    const carbonRows = compressedOnly.length > 0 ? compressedOnly : strategies;

    const suspiciousRows = carbonRows.filter((s) => {
      const baselineValue = strategyBaselineCo2(s, baselineCo2);
      const compressedValue = s.key === 'baseline' ? baselineCo2 : strategyCompressedCo2(s);
      const reduction = reductionPercent(baselineValue, compressedValue);
      return reduction != null && reduction > 80;
    });

    if (suspiciousRows.length > 0) {
      console.warn(
        '[CompressionAnalysis] CO2 reduction above 80% detected. Verify size and emissions metadata.',
        suspiciousRows.map(s => s.name)
      );
    }
  }, [strategies]);

  const title = modelName
    ? `Compression Analysis — ${modelName}`
    : 'Compression Analysis';

  if (!strategies.length) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
        <div className="h-60 flex items-center justify-center">
          <div className="text-center">
            <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-3">
              <span className="text-gray-400 text-xl">📊</span>
            </div>
            <p className="text-sm text-gray-500">Run a model to see CO2 comparison</p>
            <p className="text-xs text-gray-400 mt-1">
              {modelName
                ? 'Run a compression method above to see results here'
                : 'Select a model and compress it to see analysis'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  const sizeData = strategies.map(s => ({
    name: s.name.replace(/[→·]/g, '').substring(0, 20),
    'Size (MB)': s.size_MB,
    'Reduction (%)': s.size_reduction,
  }));

  const baseline = strategies.find(s => s.key === 'baseline');
  const baselineCo2 = baseline
    ? toFiniteNumber(baseline.co2_kg ?? baseline.training_co2_kg ?? baseline.baseline_co2_kg)
    : null;
  const compressedOnly = strategies.filter(s => s.key !== 'baseline');
  const carbonRows = compressedOnly.length > 0 ? compressedOnly : strategies;

  const carbonData = carbonRows.map((s) => {
    const baselineValue = strategyBaselineCo2(s, baselineCo2);
    const compressedValue = s.key === 'baseline' ? baselineCo2 : strategyCompressedCo2(s);
    const reduction = reductionPercent(baselineValue, compressedValue);

    return {
      name: s.name.replace(/[→·]/g, '').substring(0, 24),
      'Baseline CO₂ (kg)': baselineValue,
      'Compressed CO₂ (kg)': compressedValue,
      'CO₂ Reduction (%)': reduction,
      'Reduction Label': reduction != null ? `${reduction.toFixed(1)}%` : '',
    };
  });

  const hasCarbonValues = carbonData.some((row) =>
    row['Baseline CO₂ (kg)'] != null || row['Compressed CO₂ (kg)'] != null
  );



  const radarData = strategies
    .filter(s => s.key !== 'baseline')
    .map(s => {
      const bAcc = baseline?.accuracy || 1;
      return {
        name: s.name.replace(/[→·]/g, '').substring(0, 18),
        Accuracy: Math.round((s.accuracy / bAcc) * 100),
        'Size Reduction': Math.round(s.size_reduction),
      };
    });

  const chartKey = [
    view,
    strategies
      .map((s) => `${s.key}:${s.size_MB}:${s.size_reduction}:${s.baseline_co2_kg ?? ''}:${s.compressed_co2_kg ?? ''}:${s.co2_kg ?? ''}`)
      .join('|'),
  ].join('::');

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          {title}
        </h3>
        <div className="flex gap-1">
          {[
            { key: 'size' as ChartView, label: 'Size' },
            { key: 'carbon' as ChartView, label: 'CO₂ Emissions' },
            { key: 'radar' as ChartView, label: 'Radar' },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setView(tab.key)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${view === tab.key
                ? 'bg-green-100 text-green-700'
                : 'text-gray-500 hover:bg-gray-100'
                }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="h-72">
        {view === 'carbon' && !hasCarbonValues ? (
          <div className="h-full flex items-center justify-center text-center bg-gray-50 rounded-lg">
            <div>
              <p className="text-sm text-gray-500">Run a model to see CO2 comparison</p>
              <p className="text-xs text-gray-400 mt-1">
                Baseline and compressed emissions will appear after valid results are loaded.
              </p>
            </div>
          </div>
        ) : (
          <ResponsiveContainer key={chartKey} width="100%" height="100%">
            {view === 'size' ? (
              <BarChart data={sizeData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="Size (MB)" fill="#22c55e" radius={[4, 4, 0, 0]} />
              </BarChart>
            ) : view === 'carbon' ? (
              <BarChart data={carbonData} margin={{ top: 5, right: 20, left: 10, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" />
                <YAxis
                  tick={{ fontSize: 11 }}
                  label={{ value: 'CO₂ emissions (kg)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip formatter={(value, name) => [formatCo2Value(value), name]} />
                <Legend />
                <Bar dataKey="Baseline CO₂ (kg)" fill="#ef4444" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Compressed CO₂ (kg)" fill="#16a34a" radius={[4, 4, 0, 0]}>
                  <LabelList dataKey="Reduction Label" position="top" fill="#166534" fontSize={10} />
                </Bar>
              </BarChart>
            ) : (
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                <PolarGrid />
                <PolarAngleAxis dataKey="name" tick={{ fontSize: 10 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
                <Radar name="Accuracy" dataKey="Accuracy" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} />
                <Radar name="Size Reduction" dataKey="Size Reduction" stroke="#22c55e" fill="#22c55e" fillOpacity={0.15} />
                <Legend />
                <Tooltip />
              </RadarChart>
            )}
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
