'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  DashboardData, DynamicResult, BaselinesResponse,
  fetchAPI, getBaselines,
  loadSavedResults, saveResult, dynamicResultToStrategy,
} from '@/lib/api';
import { StatsCards } from '@/components/StatsCards';
import { ComparisonTable } from '@/components/ComparisonTable';
import { CompressionChart } from '@/components/CompressionChart';
import { EnergySection } from '@/components/EnergySection';
import { ModelGrid } from '@/components/ModelGrid';
import { ModelDashboard } from '@/components/ModelDashboard';
import { PreparePanel } from '@/components/PreparePanel';

export default function Home() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [baselines, setBaselines] = useState<BaselinesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [savedResults, setSavedResults] = useState<DynamicResult[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  // Load saved results from localStorage on mount
  useEffect(() => {
    setSavedResults(loadSavedResults());
  }, []);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [dashboard, baselineData] = await Promise.all([
        fetchAPI('/api/dashboard'),
        getBaselines().catch(() => null),
      ]);
      setData(dashboard);
      if (baselineData) setBaselines(baselineData);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to connect to API');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleNewResult = useCallback((result: DynamicResult) => {
    const updated = saveResult(result);
    setSavedResults(updated);
  }, []);

  // Get compression results for a specific model
  const getResultsForModel = (modelKey: string): DynamicResult[] => {
    return savedResults.filter(
      r => (r.model_key || r.model_name || '').toLowerCase() === modelKey.toLowerCase()
    );
  };

  // Count compression results per model
  const compressionCounts: Record<string, number> = {};
  savedResults.forEach(r => {
    const key = (r.model_key || r.model_name || '').toLowerCase();
    if (key) compressionCounts[key] = (compressionCounts[key] || 0) + 1;
  });

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-green-200 border-t-green-600 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-500">Loading dashboard...</p>
          <p className="text-xs text-gray-400 mt-1">
            Make sure the FastAPI server is running on port 8000
          </p>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="card max-w-md text-center">
          <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-red-600 text-xl">!</span>
          </div>
          <h2 className="text-lg font-semibold text-gray-900 mb-2">
            Cannot Connect to API
          </h2>
          <p className="text-sm text-gray-500 mb-4">{error}</p>
          <div className="bg-gray-50 rounded-lg p-3 text-left">
            <p className="text-xs text-gray-600 font-mono">
              cd backend<br />
              python main.py
            </p>
          </div>
          <button onClick={loadData} className="btn-primary mt-4">
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  // --- Chart data: only the currently selected model's results ---
  const selectedModelResults = selectedModel ? getResultsForModel(selectedModel) : [];
  const chartStrategies = selectedModelResults.map(dynamicResultToStrategy);

  // --- Table data: ALL accumulated results across all models ---
  const allStrategies = savedResults.map(dynamicResultToStrategy);

  // Stats: pick best across all results
  const bestStrategy = allStrategies.length > 0
    ? allStrategies.sort((a, b) => b.size_reduction - a.size_reduction)[0]
    : undefined;

  const readyCount = baselines?.ready_count || 0;
  const totalCount = baselines?.total_count || 15;

  return (
    <div className="space-y-6">
      {/* Top Stats */}
      <StatsCards
        baseline={undefined}
        bestStrategy={bestStrategy}
        gpuAvailable={data.gpu_available}
        totalModels={totalCount}
        dynamicCount={savedResults.length}
      />

      {/* Prepare Panel */}
      {baselines && (
        <PreparePanel
          readyCount={readyCount}
          totalCount={totalCount}
          onComplete={loadData}
        />
      )}

      {/* Model Grid */}
      {baselines && (
        <ModelGrid
          models={baselines.models}
          selectedModel={selectedModel}
          onSelectModel={(key) => setSelectedModel(selectedModel === key ? null : key)}
          compressionCounts={compressionCounts}
        />
      )}

      {/* Selected Model Dashboard */}
      {selectedModel && baselines?.models[selectedModel] && (
        <ModelDashboard
          model={baselines.models[selectedModel]}
          modelKey={selectedModel}
          compressionResults={getResultsForModel(selectedModel)}
          onNewResult={handleNewResult}
          onClose={() => setSelectedModel(null)}
        />
      )}

      {/* Compression Analysis (selected model only) + Strategy Comparison (all results) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CompressionChart
          strategies={chartStrategies}
          modelName={selectedModel && baselines?.models[selectedModel]
            ? baselines.models[selectedModel].model_name
            : undefined}
        />
        <ComparisonTable strategies={allStrategies} />
      </div>

      {/* Energy */}
      {(data.energy && Object.keys(data.energy).length > 0 || savedResults.length > 0) && (
        <EnergySection energy={data.energy} savedResults={savedResults} />
      )}
    </div>
  );
}
