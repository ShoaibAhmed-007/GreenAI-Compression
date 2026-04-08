'use client';

import { Strategy } from '@/lib/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar,
} from 'recharts';
import { useState } from 'react';

interface ChartProps {
  strategies: Strategy[];
  modelName?: string;
}

type ChartView = 'size' | 'accuracy' | 'carbon' | 'radar';

export function CompressionChart({ strategies, modelName }: ChartProps) {
  const [view, setView] = useState<ChartView>('size');

  // Prepare carbon emissions data
  const carbonData = strategies.map(s => ({
    name: s.name.replace(/[→·]/g, '').substring(0, 20),
    'CO₂ (kg)': typeof s.co2_kg === 'number' ? s.co2_kg : 0,
  }));

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
            <p className="text-sm text-gray-500">No compression results yet</p>
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

  const accLatData = strategies.map(s => ({
    name: s.name.replace(/[→·]/g, '').substring(0, 20),
    'Accuracy (%)': s.accuracy,
    'Latency (ms)': s.latency_ms,
  }));

  const baseline = strategies.find(s => s.key === 'baseline');
  const radarData = strategies
    .filter(s => s.key !== 'baseline')
    .map(s => {
      const bAcc = baseline?.accuracy || 1;
      const bSize = baseline?.size_MB || 1;
      const bLat = baseline?.latency_ms || 1;
      return {
        name: s.name.replace(/[→·]/g, '').substring(0, 18),
        Accuracy: Math.round((s.accuracy / bAcc) * 100),
        'Size Reduction': Math.round(s.size_reduction),
        'Speed': Math.round((bLat / Math.max(s.latency_ms, 0.1)) * 100),
      };
    });

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          {title}
        </h3>
        <div className="flex gap-1">
          {[
            { key: 'size' as ChartView, label: 'Size' },
            { key: 'accuracy' as ChartView, label: 'Acc/Lat' },
            { key: 'carbon' as ChartView, label: 'CO₂ Emissions' },
            { key: 'radar' as ChartView, label: 'Radar' },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setView(tab.key)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                view === tab.key
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
        <ResponsiveContainer width="100%" height="100%">
          {view === 'size' ? (
            <BarChart data={sizeData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="Size (MB)" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          ) : view === 'accuracy' ? (
            <BarChart data={accLatData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" />
              <YAxis yAxisId="left" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar yAxisId="left" dataKey="Accuracy (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar yAxisId="right" dataKey="Latency (ms)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          ) : view === 'carbon' ? (
            <BarChart data={carbonData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(value) => `${value} kg CO₂`} />
              <Legend />
              <Bar dataKey="CO₂ (kg)" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          ) : (
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid />
              <PolarAngleAxis dataKey="name" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
              <Radar name="Accuracy" dataKey="Accuracy" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} />
              <Radar name="Size Reduction" dataKey="Size Reduction" stroke="#22c55e" fill="#22c55e" fillOpacity={0.15} />
              <Radar name="Speed" dataKey="Speed" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} />
              <Legend />
              <Tooltip />
            </RadarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}
