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
  const ownBaseline = toFiniteNumber(strategy.baseline_co2_kg);
  if (ownBaseline != null) {
    return ownBaseline;
  }
  return fallbackBaseline;
}

function strategyCompressedCo2(strategy: Strategy): number | null {
  return toFiniteNumber(
    strategy.compressed_co2_kg ?? strategy.co2_kg ?? strategy.inference_co2_kg ?? strategy.training_co2_kg
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

function straightLabel(raw: string, maxLen: number): string {
  const cleaned = raw
    .replace(/\s*·\s*/g, ' - ')
    .replace(/\s+/g, ' ')
    .trim();

  if (cleaned.length <= maxLen) return cleaned;
  return `${cleaned.slice(0, Math.max(1, maxLen - 1))}…`;
}

export function CompressionChart({ strategies, modelName }: ChartProps) {
  const [view, setView] = useState<ChartView>('size');

  useEffect(() => {
    if (!strategies.length) return;

    const baseline = strategies.find(s => s.key === 'baseline');
    const baselineCo2 = baseline
      ? toFiniteNumber(baseline.training_co2_kg ?? baseline.baseline_co2_kg ?? baseline.co2_kg)
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

  const sizeData = strategies.map(s => ({
    name: straightLabel(s.name, 30),
    fullName: straightLabel(s.name, 120),
    'Size (MB)': s.size_MB,
    'Reduction (%)': s.size_reduction,
  }));

  const baseline = strategies.find(s => s.key === 'baseline');
  const baselineCo2 = baseline
    ? toFiniteNumber(baseline.training_co2_kg ?? baseline.baseline_co2_kg ?? baseline.co2_kg)
    : null;
  const compressedOnly = strategies.filter(s => s.key !== 'baseline');
  const carbonRows = compressedOnly.length > 0 ? compressedOnly : strategies;

  const carbonData = carbonRows.map((s) => {
    const baselineValue = strategyBaselineCo2(s, baselineCo2);
    const compressedValue = s.key === 'baseline' ? baselineCo2 : strategyCompressedCo2(s);
    const reduction = reductionPercent(baselineValue, compressedValue);

    return {
      name: straightLabel(s.name, 30),
      fullName: straightLabel(s.name, 120),
      'Baseline CO₂ (kg)': baselineValue,
      'Compressed CO₂ (kg)': compressedValue,
      'CO₂ Reduction (%)': reduction,
      'Reduction Label': reduction != null ? `${reduction.toFixed(1)}%` : '',
    };
  });

  const hasCarbonValues = carbonData.some((row) =>
    row['Baseline CO₂ (kg)'] != null || row['Compressed CO₂ (kg)'] != null
  );



  if (!strategies.length) {
    return (
      <div className="card">
        <h3 className="text-lg font-headline font-semibold text-on-surface mb-4">{title}</h3>
        <div className="h-60 flex items-center justify-center">
          <div className="text-center">
            <div className="w-12 h-12 bg-surface-container-high rounded-lg flex items-center justify-center mx-auto mb-3">
              <span className="material-symbols-outlined text-on-surface-variant/40">bar_chart</span>
            </div>
            <p className="text-sm text-on-surface-variant">Run a model to see CO2 comparison</p>
            <p className="text-xs text-on-surface-variant/50 mt-1">
              {modelName
                ? 'Run a compression method above to see results here'
                : 'Select a model and compress it to see analysis'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  const radarData = strategies
    .filter(s => s.key !== 'baseline')
    .map(s => {
      const bAcc = baseline?.accuracy || 1;
      return {
        name: straightLabel(s.name, 24),
        fullName: straightLabel(s.name, 120),
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

  const tabs = [
    { key: 'size' as ChartView, label: 'SIZE ANALYSIS' },
    { key: 'carbon' as ChartView, label: 'CARBON FOOTPRINT' },
    { key: 'radar' as ChartView, label: 'RADAR COMPARISON' },
  ];

  return (
    <div className="card">
      <div className="flex flex-wrap items-center justify-between mb-6 gap-4">
        <h3 className="text-lg font-headline font-semibold text-on-surface">
          {title}
        </h3>
        <div className="flex gap-4 overflow-x-auto">
          {tabs.map(tab => (
            <button
              key={tab.key}
              onClick={() => setView(tab.key)}
              className={`text-xs font-bold pb-2 whitespace-nowrap transition-colors ${
                view === tab.key
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-on-surface-variant hover:text-on-surface'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="h-72">
        {view === 'carbon' && !hasCarbonValues ? (
          <div className="h-full flex items-center justify-center text-center bg-surface-container-low rounded-xl">
            <div>
              <p className="text-sm text-on-surface-variant">Run a model to see CO2 comparison</p>
              <p className="text-xs text-on-surface-variant/50 mt-1">
                Baseline and compressed emissions will appear after valid results are loaded.
              </p>
            </div>
          </div>
        ) : (
          <ResponsiveContainer key={chartKey} width="100%" height="100%">
            {view === 'size' ? (
              <BarChart data={sizeData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(66, 73, 62, 0.2)" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 10, fill: '#c2c9bb' }}
                  angle={0}
                  textAnchor="middle"
                  interval={0}
                  height={56}
                />
                <YAxis tick={{ fontSize: 11, fill: '#c2c9bb' }} />
                <Tooltip
                  labelFormatter={(_, payload) => payload?.[0]?.payload?.fullName || ''}
                  contentStyle={{ backgroundColor: '#1e201d', border: '1px solid rgba(66, 73, 62, 0.3)', borderRadius: '0.75rem', color: '#e3e3de' }}
                  itemStyle={{ color: '#e3e3de' }}
                />
                <Legend wrapperStyle={{ color: '#c2c9bb' }} />
                <Bar dataKey="Size (MB)" fill="#5bdda8" radius={[4, 4, 0, 0]} />
              </BarChart>
            ) : view === 'carbon' ? (
              <BarChart data={carbonData} margin={{ top: 5, right: 20, left: 10, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(66, 73, 62, 0.2)" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 10, fill: '#c2c9bb' }}
                  angle={0}
                  textAnchor="middle"
                  interval={0}
                  height={56}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: '#c2c9bb' }}
                  label={{ value: 'CO₂ emissions (kg)', angle: -90, position: 'insideLeft', fill: '#c2c9bb' }}
                />
                <Tooltip
                  labelFormatter={(_, payload) => payload?.[0]?.payload?.fullName || ''}
                  formatter={(value, name) => [formatCo2Value(value), name]}
                  contentStyle={{ backgroundColor: '#1e201d', border: '1px solid rgba(66, 73, 62, 0.3)', borderRadius: '0.75rem', color: '#e3e3de' }}
                  itemStyle={{ color: '#e3e3de' }}
                />
                <Legend wrapperStyle={{ color: '#c2c9bb' }} />
                <Bar dataKey="Baseline CO₂ (kg)" fill="#ffb4ab" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Compressed CO₂ (kg)" fill="#5bdda8" radius={[4, 4, 0, 0]}>
                  <LabelList dataKey="Reduction Label" position="top" fill="#5bdda8" fontSize={10} />
                </Bar>
              </BarChart>
            ) : (
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                <PolarGrid stroke="rgba(66, 73, 62, 0.3)" />
                <PolarAngleAxis dataKey="name" tick={{ fontSize: 10, fill: '#c2c9bb' }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9, fill: '#8c9387' }} />
                <Radar name="Accuracy" dataKey="Accuracy" stroke="#94d3c1" fill="#94d3c1" fillOpacity={0.15} />
                <Radar name="Size Reduction" dataKey="Size Reduction" stroke="#5bdda8" fill="#5bdda8" fillOpacity={0.15} />
                <Legend wrapperStyle={{ color: '#c2c9bb' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e201d', border: '1px solid rgba(66, 73, 62, 0.3)', borderRadius: '0.75rem', color: '#e3e3de' }}
                  itemStyle={{ color: '#e3e3de' }}
                />
              </RadarChart>
            )}
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
