'use client';

import { Strategy } from '@/lib/api';

interface StatsCardsProps {
  baseline?: Strategy;
  bestStrategy?: Strategy;
  smallestModelName?: string;
  smallestModelSizeMB?: number;
  gpuAvailable: boolean;
  totalModels: number;
  dynamicCount?: number;
}

export function StatsCards({
  baseline,
  bestStrategy,
  smallestModelName,
  smallestModelSizeMB,
  gpuAvailable,
  totalModels,
  dynamicCount = 0,
}: StatsCardsProps) {
  const stats = [
    {
      label: 'Baseline Accuracy',
      value: baseline ? `${baseline.accuracy}%` : '—',
      sub: baseline?.name || 'No baseline yet',
      icon: 'target',
      iconColor: 'text-primary',
    },
    {
      label: 'Best Compression',
      value: bestStrategy ? `${bestStrategy.size_reduction}%` : '—',
      sub: bestStrategy?.name || 'Run compress.py',
      icon: 'compress',
      iconColor: 'text-secondary',
    },
    {
      label: 'Smallest Model',
      value: smallestModelSizeMB != null ? `${smallestModelSizeMB} MB` : '—',
      sub: smallestModelName || 'No ready baseline models',
      icon: 'memory',
      iconColor: 'text-tertiary',
    },
    {
      label: 'Device',
      value: gpuAvailable ? 'GPU' : 'CPU',
      sub: `${totalModels} models · ${dynamicCount} compressions`,
      icon: gpuAvailable ? 'bolt' : 'computer',
      iconColor: gpuAvailable ? 'text-primary' : 'text-on-surface-variant',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      {stats.map((stat) => (
        <div
          key={stat.label}
          className="bg-surface-container p-6 rounded-2xl flex flex-col justify-between transition-all duration-200 hover:ring-1 hover:ring-primary/10"
        >
          <div className="flex justify-between items-start">
            <span className="text-on-surface-variant text-xs font-semibold uppercase tracking-wider">
              {stat.label}
            </span>
            <span className={`material-symbols-outlined ${stat.iconColor}`}>
              {stat.icon}
            </span>
          </div>
          <div className="mt-4">
            <p className="text-3xl font-technical font-bold text-on-surface">
              {stat.value}
            </p>
            <p className="text-xs text-on-surface-variant mt-1">{stat.sub}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
