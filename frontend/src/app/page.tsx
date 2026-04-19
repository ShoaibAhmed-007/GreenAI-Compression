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
import { ModelSelector } from '@/components/ModelSelector';
import { CompressionDialog } from '@/components/CompressionDialog';
import { PreparePanel } from '@/components/PreparePanel';
import { DefenseStrategyPanel } from '@/components/DefenseStrategyPanel';

export default function Home() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [baselines, setBaselines] = useState<BaselinesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [savedResults, setSavedResults] = useState<DynamicResult[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [isCompressionDialogOpen, setCompressionDialogOpen] = useState(false);

  const normalizeModelKey = useCallback((result: DynamicResult): string => {
    const key = (result.model_key || result.model_name || '').trim().toLowerCase();
    return key.replace(/\s+/g, '_');
  }, []);

  const mapResultToStrategy = useCallback((result: DynamicResult): Strategy => {
    const strategy = dynamicResultToStrategy(result);
    const modelKey = normalizeModelKey(result);
    const baselineModel = baselines?.models?.[modelKey];
    const immutableBaselineCo2 =
      baselineModel && typeof baselineModel.training_co2_kg === 'number'
        ? baselineModel.training_co2_kg
        : undefined;

    if (immutableBaselineCo2 == null) {
      return strategy;
    }

    return {
      ...strategy,
      baseline_co2_kg: immutableBaselineCo2,
    };
  }, [baselines, normalizeModelKey]);

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
    setCompressionDialogOpen(false);
  }, []);

  useEffect(() => {
    if (selectedModel && baselines && !baselines.models[selectedModel]) {
      setSelectedModel(null);
      setCompressionDialogOpen(false);
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
          <div className="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full animate-spin mx-auto mb-4" />
          <p className="text-on-surface-variant">Loading dashboard...</p>
          <p className="text-xs text-on-surface-variant/50 mt-1">
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
          <div className="w-12 h-12 bg-error-container/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="material-symbols-outlined text-error">error</span>
          </div>
          <h2 className="text-lg font-headline font-semibold text-on-surface mb-2">
            Cannot Connect to API
          </h2>
          <p className="text-sm text-on-surface-variant mb-4">{error}</p>
          <div className="bg-surface-container-low rounded-xl p-3 text-left">
            <p className="text-xs text-on-surface-variant font-technical">
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
    const resultStrategies = selectedModelResults.map(mapResultToStrategy);
    // Only add baseline if not already present
    if (!resultStrategies.some(s => s.key === 'baseline')) {
      chartStrategies = [baselineStrategy, ...resultStrategies];
    } else {
      chartStrategies = resultStrategies;
    }
  } else if (savedResults.length > 0) {
    chartStrategies = savedResults.map(mapResultToStrategy);
  } else {
    chartStrategies = [];
  }

  // --- Table data: ALL accumulated results across all models ---
  const allStrategies = savedResults.map((result) => ({
    ...mapResultToStrategy(result),
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

  // Best baseline accuracy — highest accuracy among all ready models (dynamic)
  const readyAccuracyModels = Object.values(baselines?.models || {}).filter(
    (m) => m.status === 'ready' && typeof m.accuracy === 'number' && m.accuracy != null
  );
  const bestBaselineModel = readyAccuracyModels.length > 0
    ? [...readyAccuracyModels].sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0))[0]
    : null;
  const bestBaselineStrategy = bestBaselineModel
    ? {
        key: 'baseline',
        name: bestBaselineModel.model_name,
        accuracy: bestBaselineModel.accuracy || 0,
        size_MB: bestBaselineModel.size_MB || 0,
        size_reduction: 0,
        params: bestBaselineModel.total_params || 0,
      } as import('@/lib/api').Strategy
    : undefined;

  return (
    <div className="space-y-10">
      {/* Dashboard Header & Actions */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
        <div>
          <h2 className="text-4xl font-headline font-bold text-on-surface tracking-tight">
            System Overview
          </h2>
          <p className="text-on-surface-variant mt-2 max-w-xl">
            Monitor your machine learning efficiency metrics and coordinate model compression pipelines for sustainable edge deployment.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleResetSessionResults}
            disabled={savedResults.length === 0}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all ${
              savedResults.length === 0
                ? 'bg-surface-container-high text-on-surface-variant/40 cursor-not-allowed'
                : 'bg-surface-container-high text-on-surface-variant hover:bg-surface-container-highest hover:text-on-surface'
            }`}
          >
            <span className="material-symbols-outlined text-lg">delete_sweep</span>
            Clear History
          </button>
          <Link
            href="/model-comparison"
            className="btn-primary flex items-center gap-2"
          >
            <span className="material-symbols-outlined text-lg">analytics</span>
            Compare Models on Images
          </Link>
        </div>
      </div>

      {/* Top Stats */}
      <StatsCards
        baseline={bestBaselineStrategy}
        bestStrategy={bestStrategy}
        smallestModelName={smallestBaselineModel?.model_name}
        smallestModelSizeMB={smallestBaselineModel?.size_MB}
        gpuAvailable={data.gpu_available}
        totalModels={totalCount}
        dynamicCount={savedResults.length}
      />

      <DefenseStrategyPanel />

      {/* Prepare Panel */}
      {baselines && (
        <PreparePanel
          readyCount={readyCount}
          totalCount={totalCount}
          onComplete={loadData}
        />
      )}

      {/* Model Grid + Analysis Sidebar */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        {/* Model Library Grid (2/3 width) */}
        <div className="xl:col-span-2 space-y-6">
          {baselines && (
            <ModelSelector
              models={baselines.models}
              selectedModel={selectedModel}
              onSelectModel={(key) => {
                setSelectedModel(key);
                setCompressionDialogOpen(true);
              }}
              compressionCounts={compressionCounts}
            />
          )}

          {/* Inline Expanded Model Details */}
          {selectedModel && baselines?.models[selectedModel] && (
            <CompressionDialog
              open={isCompressionDialogOpen}
              modelKey={selectedModel}
              model={baselines.models[selectedModel]}
              onClose={() => setCompressionDialogOpen(false)}
              onNewResult={handleNewResult}
            />
          )}
        </div>

        {/* Analysis Sidebar (1/3 width) */}
        <div className="space-y-8">
          {/* Energy */}
          <EnergySection
            energy={data.energy || {}}
            savedResults={savedResults}
            baselines={baselines?.models || {}}
            onDeleteResult={handleDeleteResult}
          />

          {/* Compression Analysis + Strategy Comparison */}
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
      </div>
    </div>
  );
}
