'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import {
  DashboardData, DynamicResult, BaselinesResponse, Strategy,
  fetchAPI, getBaselines, clearSavedResults, loadSavedResults,
  saveResult, deleteSavedResultByKey, getResultStorageKey,
  dynamicResultToStrategy,
} from '@/lib/api';
import { StatsCards } from '@/components/StatsCards';
import { ComparisonTable } from '@/components/ComparisonTable';
import { CompressionChart } from '@/components/CompressionChart';
import { EnergySection } from '@/components/EnergySection';
import { ModelGrid } from '@/components/ModelGrid';
import { ModelDashboard } from '../components/ModelDashboard';
import { PreparePanel } from '@/components/PreparePanel';

export default function Home() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [baselines, setBaselines] = useState<BaselinesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [savedResults, setSavedResults] = useState<DynamicResult[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const normalizeModelKey = useCallback((result: DynamicResult): string => {
    const key = (result.model_key || result.model_name || '').trim().toLowerCase();
    return key.replace(/\s+/g, '_');
  }, []);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      console.log('[Dashboard] loading latest dashboard and baselines');
      const [dashboardResult, baselinesResult] = await Promise.allSettled([
        fetchAPI('/api/dashboard'),
        getBaselines(),
      ]);

      if (dashboardResult.status !== 'fulfilled') {
        throw dashboardResult.reason;
      }

      const dashboard = dashboardResult.value;
      const baselineData = baselinesResult.status === 'fulfilled' ? baselinesResult.value : null;

      setData(dashboard);
      if (baselineData) setBaselines(baselineData);

      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to connect to API');
      console.error('[Dashboard] failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Load persisted compression history on page startup.
    setSavedResults(loadSavedResults());
    loadData();
    const interval = setInterval(loadData, 15000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleNewResult = useCallback((result: DynamicResult) => {
    console.log('[Dashboard] new compression result received:', {
      model: result.model_key || result.model_name,
      strategy: result.strategy,
    });

    const updated = saveResult(result);
    setSavedResults(updated);
  }, []);

  const handleDeleteResult = useCallback((resultKey: string) => {
    const updated = deleteSavedResultByKey(resultKey);
    setSavedResults(updated);
  }, []);

  const handleResetSessionResults = useCallback(() => {
    setSavedResults([]);
    clearSavedResults();
    setSelectedModel(null);
  }, []);

  useEffect(() => {
    if (selectedModel && baselines && !baselines.models[selectedModel]) {
      setSelectedModel(null);
    }
  }, [selectedModel, baselines]);

  // Get compression results for a specific model
  const getResultsForModel = (modelKey: string): DynamicResult[] => {
    return savedResults.filter(
      r => normalizeModelKey(r) === modelKey.toLowerCase()
    );
  };

  // Count compression results per model
  const compressionCounts: Record<string, number> = {};
  savedResults.forEach(r => {
    const key = normalizeModelKey(r);
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

  // --- Chart data ---
  // If a model is selected, show baseline + that model's compression runs.
  // Otherwise, show all saved compression runs so the chart still renders after refresh.
  let chartStrategies: Strategy[] = [];
  if (selectedModel && baselines?.models[selectedModel]) {
    const baselineModel = baselines.models[selectedModel];
    const baselineTrainingCo2 =
      typeof baselineModel.training_co2_kg === 'number'
        ? baselineModel.training_co2_kg
        : undefined;
    const baselineTrainingEnergy =
      typeof baselineModel.training_energy_kwh === 'number'
        ? baselineModel.training_energy_kwh
        : undefined;
    const baselineStrategy: Strategy = {
      key: 'baseline',
      name: `${baselineModel.model_name} · Baseline`,
      accuracy: baselineModel.accuracy || 0,
      size_MB: baselineModel.size_MB || 0,
      size_reduction: 0,
      // latency_ms: baselineModel.latency_ms || 0,
      params: baselineModel.total_params || 0,
      baseline_co2_kg: baselineTrainingCo2,
      compressed_co2_kg: baselineTrainingCo2,
      co2_kg: baselineTrainingCo2,
      training_co2_kg: baselineTrainingCo2,
      inference_co2_kg: undefined,
      training_energy_kwh: baselineTrainingEnergy,
      inference_energy_kwh: undefined,
      flops_M: undefined,
      sparsity_percent: undefined,
    };
    const selectedModelResults = getResultsForModel(selectedModel);
    const resultStrategies = selectedModelResults.map(dynamicResultToStrategy);
    // Only add baseline if not already present
    if (!resultStrategies.some(s => s.key === 'baseline')) {
      chartStrategies = [baselineStrategy, ...resultStrategies];
    } else {
      chartStrategies = resultStrategies;
    }
  } else if (savedResults.length > 0) {
    chartStrategies = savedResults.map(dynamicResultToStrategy);
  } else {
    chartStrategies = [];
  }

  // --- Table data: ALL accumulated results across all models ---
  const allStrategies = savedResults.map((result) => ({
    ...dynamicResultToStrategy(result),
    key: getResultStorageKey(result),
  }));

  // Stats: pick best across all results
  const bestStrategy = allStrategies.length > 0
    ? [...allStrategies].sort((a, b) => b.size_reduction - a.size_reduction)[0]
    : undefined;

  const readyCount = baselines?.ready_count || 0;
  const totalCount = baselines?.total_count || 15;

  const readyBaselineModels = Object.values(baselines?.models || {}).filter(
    (m) => m.status === 'ready' && typeof m.size_MB === 'number'
  );
  const smallestBaselineModel = readyBaselineModels.length > 0
    ? [...readyBaselineModels].sort((a, b) => (a.size_MB || 0) - (b.size_MB || 0))[0]
    : null;

  return (
    <div className="space-y-6">
      {/* Top Stats */}
      <StatsCards
        baseline={undefined}
        bestStrategy={bestStrategy}
        smallestModelName={smallestBaselineModel?.model_name}
        smallestModelSizeMB={smallestBaselineModel?.size_MB}
        gpuAvailable={data.gpu_available}
        totalModels={totalCount}
        dynamicCount={savedResults.length}
      />

      <div className="flex flex-wrap justify-end gap-2">
        <Link
          href="/model-comparison"
          className="btn-primary"
        >
          Compare Models on Images
        </Link>
        <button
          onClick={handleResetSessionResults}
          disabled={savedResults.length === 0}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${
            savedResults.length === 0
              ? 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed'
              : 'bg-red-50 text-red-700 border-red-200 hover:bg-red-100'
          }`}
        >
          Clear History
        </button>
      </div>

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
        <ComparisonTable
          strategies={allStrategies}
          onDeleteStrategy={handleDeleteResult}
        />
      </div>

      {/* Energy */}
      <EnergySection
        energy={data.energy || {}}
        savedResults={savedResults}
        baselines={baselines?.models || {}}
        onDeleteResult={handleDeleteResult}
      />
    </div>
  );
}
