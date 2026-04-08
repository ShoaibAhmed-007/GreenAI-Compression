'use client';

import { DynamicResult } from '@/lib/api';

interface EnergySectionProps {
  energy: Record<string, any>;
  savedResults?: DynamicResult[];
}

export function EnergySection({ energy, savedResults = [] }: EnergySectionProps) {
  const hasSavings = energy && Object.keys(energy).length > 0;
  const hasResults = savedResults.length > 0;
  const showSection = hasSavings || hasResults;

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
          <span className="text-green-600 text-sm">⚡</span>
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Energy & Carbon Emissions
          </h3>
          <p className="text-xs text-gray-500">Phase 7 — CodeCarbon tracking</p>
        </div>
      </div>

      {!showSection ? (
        <div className="bg-gray-50 rounded-lg p-6 text-center">
          <p className="text-sm text-gray-500 mb-2">
            No energy data available yet.
          </p>
          <p className="text-xs text-gray-400">
            Run energy tracking from the Actions panel or compress a model.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {/* Dynamic compression emissions */}
          {hasResults && (
            <>
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Compression Results</p>
              {savedResults.map((r, i) => {
                const label = `${r.model_name || r.model_key || 'Model'} · ${r.compression_method || r.strategy}`;
                const trainingCo2 = r.training_co2_kg ?? r.training_emissions_kg;
                const inferenceCo2 = r.inference_co2_kg ?? r.inference_emissions_kg ?? r.emissions_kg;
                const inferenceEnergy = r.inference_energy_kwh ?? r.energy_kwh;
                return (
                  <div key={i} className="bg-gray-50 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-700">{label}</p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {trainingCo2 != null ? `Train CO₂: ${trainingCo2.toFixed(6)} kg` : 'Train CO₂: —'}
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {inferenceCo2 != null ? `Infer CO₂: ${inferenceCo2.toFixed(6)} kg` : 'Infer CO₂: —'}
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {inferenceEnergy != null ? `Infer Energy: ${inferenceEnergy.toFixed(8)} kWh` : 'Infer Energy: —'}
                      </p>
                    </div>
                    <div className="text-right">
                      <span className="text-lg font-bold text-green-600">
                        {r.size_reduction_percent?.toFixed(1)}%
                      </span>
                      <p className="text-xs text-green-600">size ↓</p>
                    </div>
                  </div>
                );
              })}
            </>
          )}

          {/* Original energy tracking data */}
          {hasSavings && (
            <>
              {hasResults && (
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mt-4">
                  Baseline Energy Tracking
                </p>
              )}
              {Object.entries(energy).map(([key, data]: [string, any]) => {
                if (key === 'training_compact_vs_baseline') {
                  return (
                    <div key={key} className="bg-green-50 rounded-lg p-4 border border-green-100">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-green-800">
                            Training Energy Savings
                          </p>
                          <p className="text-xs text-green-600 mt-0.5">
                            Compact Student vs ResNet18 Baseline
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-2xl font-bold text-green-700">
                            {data.energy_saving_percent}%
                          </p>
                          <p className="text-xs text-green-600">less energy</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4 mt-3 pt-3 border-t border-green-200">
                        <div>
                          <p className="text-xs text-green-600">Baseline</p>
                          <p className="text-sm font-mono text-green-800">
                            {data.baseline_energy_kWh?.toFixed(6)} kWh
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-green-600">Student</p>
                          <p className="text-sm font-mono text-green-800">
                            {data.student_energy_kWh?.toFixed(6)} kWh
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                }

                return (
                  <div key={key} className="bg-gray-50 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-700">
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {data.inference_energy_kWh !== undefined
                          ? `Energy: ${data.inference_energy_kWh.toFixed(8)} kWh`
                          : ''}
                      </p>
                    </div>
                    <div className="text-right">
                      <span className={`text-lg font-bold ${
                        data.energy_saving_percent > 0 ? 'text-green-600' : 'text-gray-600'
                      }`}>
                        {data.energy_saving_percent !== undefined
                          ? `${data.energy_saving_percent}%`
                          : '—'}
                      </span>
                      {data.energy_saving_percent > 0 && (
                        <p className="text-xs text-green-600">saved</p>
                      )}
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      )}
    </div>
  );
}
