'use client';

import { Strategy } from '@/lib/api';

interface StatsCardsProps {
  baseline?: Strategy;
  bestStrategy?: Strategy;
  gpuAvailable: boolean;
  totalModels: number;
  dynamicCount?: number;
}

export function StatsCards({ baseline, bestStrategy, gpuAvailable, totalModels, dynamicCount = 0 }: StatsCardsProps) {
  const stats = [
    {
      label: 'Baseline Accuracy',
      value: baseline ? `${baseline.accuracy}%` : '—',
      sub: baseline?.name || 'No baseline yet',
      color: 'text-blue-600',
      bg: 'bg-blue-50',
    },
    {
      label: 'Best Compression',
      value: bestStrategy ? `${bestStrategy.size_reduction}%` : '—',
      sub: bestStrategy?.name || 'Run compress.py',
      color: 'text-green-600',
      bg: 'bg-green-50',
    },
    {
      label: 'Smallest Model',
      value: bestStrategy ? `${bestStrategy.size_MB} MB` : '—',
      sub: baseline ? `vs ${baseline.size_MB} MB baseline` : '',
      color: 'text-purple-600',
      bg: 'bg-purple-50',
    },
    {
      label: 'Device',
      value: gpuAvailable ? 'GPU' : 'CPU',
      sub: `${totalModels} models · ${dynamicCount} compressions`,
      color: gpuAvailable ? 'text-amber-600' : 'text-gray-600',
      bg: gpuAvailable ? 'bg-amber-50' : 'bg-gray-50',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <div key={stat.label} className="card-hover">
          <div className={`w-10 h-10 ${stat.bg} rounded-lg flex items-center justify-center mb-3`}>
            <span className={`text-lg font-bold ${stat.color}`}>
              {stat.label[0]}
            </span>
          </div>
          <p className="text-sm text-gray-500">{stat.label}</p>
          <p className={`text-2xl font-bold ${stat.color} mt-1`}>{stat.value}</p>
          <p className="text-xs text-gray-400 mt-1">{stat.sub}</p>
        </div>
      ))}
    </div>
  );
}
