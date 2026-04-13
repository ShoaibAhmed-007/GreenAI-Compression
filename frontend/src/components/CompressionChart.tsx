'use client';

import { Strategy } from '@/lib/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, LabelList, Cell, LineChart, Line, ReferenceLine,
} from 'recharts';
import { useEffect, useMemo, useState } from 'react';

interface ChartProps {
  strategies: Strategy[];
  modelName?: string;
}

type ChartView = 'size' | 'carbon' | 'radar';

type StrategyGroup = 'baseline' | 'pruning' | 'quantization' | 'hybrid' | 'kd' | 'other';

const VIEW_CONFIG: Record<ChartView | 'accuracy' | 'latency', {
  label: string;
  title: string;
  subtitle: string;
}> = {
  accuracy: {
    label: 'Accuracy',
    title: 'Accuracy Comparison',
    subtitle: 'Top-1 accuracy across baseline and compressed models',
  },
  size: {
    label: 'Model Size',
    title: 'Model Size',
    subtitle: 'Compressed model footprint in MB',
  },
  carbon: {
    label: 'CO₂ Emissions',
    title: 'CO₂ Emissions',
    subtitle: 'Baseline training CO₂ vs compressed total CO₂',
  },
  latency: {
    label: 'Latency',
    title: 'Latency Comparison',
    subtitle: 'Baseline inference latency vs compressed latency',
  },
  radar: {
    label: 'Radar',
    title: 'Radar Comparison',
    subtitle: 'Normalized accuracy, size, CO₂, and latency scores',
  },
};

const STRATEGY_COLORS: Record<StrategyGroup, string> = {
  baseline: '#475569',
  pruning: '#16a34a',
  quantization: '#2563eb',
  hybrid: '#ea580c',
  kd: '#0f766e',
  other: '#6b7280',
};

const GRID_COLOR = '#e2e8f0';
const AXIS_TICK_STYLE = { fontSize: 12, fill: '#334155', fontWeight: 600 };
const LEGEND_STYLE = { fontSize: 12, paddingTop: 4 };

type ExtendedChartView = ChartView | 'accuracy' | 'latency';

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

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function shortName(value: string, maxLength: number): string {
  const clean = value.replace(/[→·]/g, '').trim();
  return clean.length > maxLength ? `${clean.slice(0, maxLength - 1)}…` : clean;
}

function detectStrategyGroup(strategy: Strategy): StrategyGroup {
  const token = `${strategy.key} ${strategy.name}`.toLowerCase();
  if (token.includes('baseline')) return 'baseline';
  if (token.includes('prun')) return 'pruning';
  if (token.includes('quant')) return 'quantization';
  if (token.includes('hybrid')) return 'hybrid';
  if (token.includes('distill') || /\bkd\b/.test(token)) return 'kd';
  return 'other';
}

function strategyColor(strategy: Strategy): string {
  return STRATEGY_COLORS[detectStrategyGroup(strategy)];
}

function strategyBaselineCo2(strategy: Strategy, fallbackBaseline: number | null): number | null {
  if (fallbackBaseline != null) {
    return fallbackBaseline;
  }
  return toFiniteNumber(strategy.baseline_co2_kg);
}

function strategyCompressedCo2(strategy: Strategy): number | null {
  return toFiniteNumber(
    strategy.compressed_co2_kg ?? strategy.co2_kg ?? strategy.inference_co2_kg
  );
}

function strategyBaselineLatency(strategy: Strategy, fallbackBaseline: number | null): number | null {
  if (fallbackBaseline != null) {
    return fallbackBaseline;
  }
  return toFiniteNumber(strategy.baseline_latency_ms ?? strategy.latency_ms);
}

function strategyCompressedLatency(strategy: Strategy): number | null {
  return toFiniteNumber(strategy.compressed_latency_ms ?? strategy.latency_ms);
}

function reductionPercent(baseline: number | null, compressed: number | null): number | null {
  if (baseline == null || compressed == null || baseline <= 0) return null;
  return ((baseline - compressed) / baseline) * 100;
}

function formatPercent(value: unknown): string {
  const numeric = toFiniteNumber(value);
  if (numeric == null) return 'Not Available';
  return `${numeric.toFixed(2)}%`;
}

function formatMs(value: unknown): string {
  const numeric = toFiniteNumber(value);
  if (numeric == null) return 'Not Available';
  return `${numeric.toFixed(2)} ms`;
}

function formatCo2Value(value: unknown): string {
  const numeric = toFiniteNumber(value);
  if (numeric == null) return 'Not Available';
  if (numeric > 0 && numeric < 0.000001) return '<0.000001 kg';
  return `${numeric.toFixed(6)} kg`;
}

export function CompressionChart({ strategies, modelName }: ChartProps) {
  const [view, setView] = useState<ExtendedChartView>('accuracy');

  const title = modelName
    ? `Compression Analysis — ${modelName}`
    : 'Compression Analysis';

  const baselineStrategy = useMemo(
    () => strategies.find((s) => s.key === 'baseline' || detectStrategyGroup(s) === 'baseline') ?? null,
    [strategies]
  );

  const baselineAccuracy = toFiniteNumber(baselineStrategy?.accuracy);
  const baselineCo2 = baselineStrategy
    ? toFiniteNumber(
      baselineStrategy.training_co2_kg ?? baselineStrategy.baseline_co2_kg ?? baselineStrategy.co2_kg
    )
    : null;
  const baselineLatency = baselineStrategy
    ? toFiniteNumber(
      baselineStrategy.baseline_latency_ms ?? baselineStrategy.compressed_latency_ms ?? baselineStrategy.latency_ms
    )
    : null;

  const compressedStrategies = useMemo(
    () => strategies.filter((s) => s.key !== 'baseline' && detectStrategyGroup(s) !== 'baseline'),
    [strategies]
  );

  const targetsForComparison = compressedStrategies.length > 0
    ? compressedStrategies
    : strategies.filter((s) => s.key !== 'baseline');

  useEffect(() => {
    if (!strategies.length) return;

    const suspiciousRows = targetsForComparison.filter((s) => {
      const baselineValue = strategyBaselineCo2(s, baselineCo2);
      const compressedValue = strategyCompressedCo2(s);
      const reduction = reductionPercent(baselineValue, compressedValue);
      return reduction != null && reduction > 80;
    });

    if (suspiciousRows.length > 0) {
      console.warn(
        '[CompressionAnalysis] CO2 reduction above 80% detected. Verify size and emissions metadata.',
        suspiciousRows.map(s => s.name)
      );
    }
  }, [targetsForComparison, baselineCo2, strategies.length]);

  const sizeData = strategies.map((s) => ({
    key: s.key,
    name: shortName(s.name, 20),
    color: strategyColor(s),
    'Model Size (MB)': toFiniteNumber(s.size_MB) ?? 0,
    'Reduction (%)': s.size_reduction,
  }));

  const accuracyData = strategies.map((s) => ({
    key: s.key,
    name: shortName(s.name, 20),
    color: strategyColor(s),
    'Accuracy (%)': toFiniteNumber(s.accuracy) ?? 0,
  }));

  const carbonRows = targetsForComparison.length > 0 ? targetsForComparison : strategies;
  const carbonData = carbonRows.map((s) => {
    const baselineValue = strategyBaselineCo2(s, baselineCo2);
    const compressedValue = strategyCompressedCo2(s);
    const reduction = reductionPercent(baselineValue, compressedValue);

    return {
      key: s.key,
      color: strategyColor(s),
      name: shortName(s.name, 24),
      'Baseline CO₂ (kg)': baselineValue,
      'Compressed CO₂ (kg)': compressedValue,
      'CO₂ Reduction (%)': reduction,
      'Reduction Label': reduction != null ? `${reduction.toFixed(1)}%` : '',
    };
  });

  const hasCarbonValues = carbonData.some((row) =>
    row['Baseline CO₂ (kg)'] != null || row['Compressed CO₂ (kg)'] != null
  );

  const latencyRows = targetsForComparison.length > 0 ? targetsForComparison : strategies;
  const latencyData = latencyRows.map((s) => {
    const baselineValue = strategyBaselineLatency(s, baselineLatency);
    const compressedValue = strategyCompressedLatency(s);
    const speedup = reductionPercent(baselineValue, compressedValue);

    return {
      key: s.key,
      color: strategyColor(s),
      name: shortName(s.name, 24),
      'Baseline Latency (ms)': baselineValue,
      'Compressed Latency (ms)': compressedValue,
      'Latency Gain (%)': speedup,
      'Speedup Label': speedup != null ? `${speedup.toFixed(1)}%` : '',
    };
  });

  const hasLatencyValues = latencyData.some((row) =>
    row['Baseline Latency (ms)'] != null || row['Compressed Latency (ms)'] != null
  );

  const radarSeries = targetsForComparison.map((s, index) => {
    const accuracy = toFiniteNumber(s.accuracy);
    const accuracyScore =
      baselineAccuracy != null && baselineAccuracy > 0 && accuracy != null
        ? clamp((accuracy / baselineAccuracy) * 100, 0, 100)
        : clamp(accuracy ?? 0, 0, 100);

    const sizeScore = clamp(toFiniteNumber(s.size_reduction) ?? 0, 0, 100);
    const co2Score = clamp(
      reductionPercent(strategyBaselineCo2(s, baselineCo2), strategyCompressedCo2(s)) ?? 0,
      0,
      100
    );
    const latencyScore = clamp(
      reductionPercent(strategyBaselineLatency(s, baselineLatency), strategyCompressedLatency(s)) ?? 0,
      0,
      100
    );

    return {
      key: `series_${index}`,
      label: shortName((s.name.split('·').pop() || s.name).trim(), 14),
      color: strategyColor(s),
      accuracyScore,
      sizeScore,
      co2Score,
      latencyScore,
    };
  });

  const radarMetricRows = [
    { label: 'Accuracy', key: 'accuracyScore' as const },
    { label: 'Size', key: 'sizeScore' as const },
    { label: 'CO₂', key: 'co2Score' as const },
    { label: 'Latency', key: 'latencyScore' as const },
  ];

  const radarData = radarMetricRows.map((metricRow) => {
    const row: Record<string, string | number> = { metric: metricRow.label };
    radarSeries.forEach((series) => {
      row[series.key] = series[metricRow.key];
    });
    return row;
  });

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

  const chartKey = [
    view,
    strategies
      .map((s) => [
        s.key,
        s.size_MB,
        s.size_reduction,
        s.accuracy,
        s.baseline_co2_kg,
        s.compressed_co2_kg,
        s.training_co2_kg,
        s.inference_co2_kg,
        s.baseline_latency_ms,
        s.compressed_latency_ms,
        s.latency_ms,
      ].join(':'))
      .join('|'),
  ].join('::');

  const activeView = VIEW_CONFIG[view];

  const renderEmptyState = (message: string) => (
    <div className="h-full flex items-center justify-center text-center bg-gray-50 rounded-lg border border-gray-100">
      <div>
        <p className="text-sm text-gray-600 font-medium">{message}</p>
        <p className="text-xs text-gray-400 mt-1">Run compression methods to populate this chart.</p>
      </div>
    </div>
  );

  return (
    <div className="card">
      <div className="flex flex-col gap-3 mb-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              {title}
            </h3>
            <p className="text-xs text-gray-500 mt-1">
              <span className="font-semibold text-gray-700">{activeView.title}</span>
              {' · '}
              {activeView.subtitle}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {[
            { key: 'accuracy' as ExtendedChartView, label: VIEW_CONFIG.accuracy.label },
            { key: 'size' as ChartView, label: 'Size' },
            { key: 'carbon' as ChartView, label: 'CO₂ Emissions' },
            { key: 'latency' as ExtendedChartView, label: VIEW_CONFIG.latency.label },
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

      <div className="h-[21rem] sm:h-[24rem]">
        {(view === 'carbon' && !hasCarbonValues) ? (
          renderEmptyState('CO₂ values are not available yet')
        ) : (view === 'latency' && !hasLatencyValues) ? (
          renderEmptyState('Latency values are not available yet')
        ) : (view === 'radar' && radarSeries.length === 0) ? (
          renderEmptyState('Need at least one compressed model for radar comparison')
        ) : (
          <ResponsiveContainer key={chartKey} width="100%" height="100%">
            {view === 'accuracy' ? (
              <LineChart data={accuracyData} margin={{ top: 14, right: 24, left: 12, bottom: 70 }}>
                <CartesianGrid strokeDasharray="4 4" stroke={GRID_COLOR} />
                <XAxis dataKey="name" tick={AXIS_TICK_STYLE} angle={-30} textAnchor="end" height={74} />
                <YAxis
                  tick={AXIS_TICK_STYLE}
                  domain={[0, 100]}
                  label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', style: AXIS_TICK_STYLE }}
                />
                <Tooltip formatter={(value) => [formatPercent(value), 'Accuracy']} />
                <Legend verticalAlign="top" wrapperStyle={LEGEND_STYLE} />
                {baselineAccuracy != null && (
                  <ReferenceLine
                    y={baselineAccuracy}
                    stroke="#64748b"
                    strokeDasharray="5 4"
                    label={{ value: 'Baseline', position: 'insideTopRight', fill: '#475569', fontSize: 10 }}
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="Accuracy (%)"
                  name="Accuracy (%)"
                  stroke="#0f766e"
                  strokeWidth={3}
                  dot={{ r: 4, fill: '#0f766e' }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            ) : view === 'size' ? (
              <BarChart data={sizeData} margin={{ top: 14, right: 24, left: 12, bottom: 70 }}>
                <CartesianGrid strokeDasharray="4 4" stroke={GRID_COLOR} />
                <XAxis dataKey="name" tick={AXIS_TICK_STYLE} angle={-30} textAnchor="end" height={74} />
                <YAxis
                  tick={AXIS_TICK_STYLE}
                  label={{ value: 'Model Size (MB)', angle: -90, position: 'insideLeft', style: AXIS_TICK_STYLE }}
                />
                <Tooltip formatter={(value) => [`${toFiniteNumber(value)?.toFixed(2) ?? 'N/A'} MB`, 'Model Size']} />
                <Legend verticalAlign="top" wrapperStyle={LEGEND_STYLE} />
                <Bar dataKey="Model Size (MB)" name="Model Size (MB)" radius={[6, 6, 0, 0]}>
                  {sizeData.map((entry) => (
                    <Cell key={entry.key} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            ) : view === 'carbon' ? (
              <BarChart data={carbonData} margin={{ top: 14, right: 24, left: 18, bottom: 70 }}>
                <CartesianGrid strokeDasharray="4 4" stroke={GRID_COLOR} />
                <XAxis dataKey="name" tick={AXIS_TICK_STYLE} angle={-30} textAnchor="end" height={74} />
                <YAxis
                  tick={AXIS_TICK_STYLE}
                  label={{ value: 'CO₂ Emissions (kg)', angle: -90, position: 'insideLeft', style: AXIS_TICK_STYLE }}
                />
                <Tooltip formatter={(value, name) => [formatCo2Value(value), name]} />
                <Legend verticalAlign="top" wrapperStyle={LEGEND_STYLE} />
                <Bar dataKey="Baseline CO₂ (kg)" fill="#64748b" radius={[6, 6, 0, 0]} />
                <Bar dataKey="Compressed CO₂ (kg)" radius={[6, 6, 0, 0]}>
                  {carbonData.map((entry) => (
                    <Cell key={entry.key} fill={entry.color} />
                  ))}
                  <LabelList dataKey="Reduction Label" position="top" fill="#166534" fontSize={10} />
                </Bar>
              </BarChart>
            ) : view === 'latency' ? (
              <BarChart data={latencyData} margin={{ top: 14, right: 24, left: 18, bottom: 70 }}>
                <CartesianGrid strokeDasharray="4 4" stroke={GRID_COLOR} />
                <XAxis dataKey="name" tick={AXIS_TICK_STYLE} angle={-30} textAnchor="end" height={74} />
                <YAxis
                  tick={AXIS_TICK_STYLE}
                  label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', style: AXIS_TICK_STYLE }}
                />
                <Tooltip
                  formatter={(value, name) => {
                    if (String(name).toLowerCase().includes('gain')) {
                      return [formatPercent(value), name];
                    }
                    return [formatMs(value), name];
                  }}
                />
                <Legend verticalAlign="top" wrapperStyle={LEGEND_STYLE} />
                <Bar dataKey="Baseline Latency (ms)" fill="#94a3b8" radius={[6, 6, 0, 0]} />
                <Bar dataKey="Compressed Latency (ms)" radius={[6, 6, 0, 0]}>
                  {latencyData.map((entry) => (
                    <Cell key={entry.key} fill={entry.color} />
                  ))}
                  <LabelList dataKey="Speedup Label" position="top" fill="#0f766e" fontSize={10} />
                </Bar>
              </BarChart>
            ) : (
              <RadarChart data={radarData} cx="50%" cy="52%" outerRadius="67%">
                <PolarGrid stroke={GRID_COLOR} />
                <PolarAngleAxis dataKey="metric" tick={AXIS_TICK_STYLE} />
                <PolarRadiusAxis
                  angle={30}
                  domain={[0, 100]}
                  tick={{ fontSize: 10, fill: '#64748b' }}
                  tickCount={6}
                />
                {radarSeries.map((series) => (
                  <Radar
                    key={series.key}
                    name={series.label}
                    dataKey={series.key}
                    stroke={series.color}
                    fill={series.color}
                    fillOpacity={0.12}
                    strokeWidth={2}
                  />
                ))}
                <Legend verticalAlign="top" wrapperStyle={LEGEND_STYLE} />
                <Tooltip formatter={(value) => [formatPercent(value), 'Score']} />
              </RadarChart>
            )}
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
