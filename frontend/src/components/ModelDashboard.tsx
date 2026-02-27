'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  BaselineModel, DynamicResult, CompressionStatus,
  compressPreloaded, getCompressionStatus, triggerPrepare,
} from '@/lib/api';

interface ModelDashboardProps {
  model: BaselineModel;
  modelKey: string;
  compressionResults: DynamicResult[];
  onNewResult: (result: DynamicResult) => void;
  onClose: () => void;
}

const METHODS = [
  { value: 'pruning', label: 'Pruning (70%)', desc: 'Remove 70% of smallest weights + fine-tune + gzip', icon: '✂️' },
  { value: 'quantization', label: 'Quantization (INT8)', desc: 'Dynamic INT8 quantization for weights', icon: '📦' },
  { value: 'hybrid', label: 'Hybrid', desc: 'Prune 50% + fine-tune + quantize INT8', icon: '🔗' },
  { value: 'kd', label: 'Knowledge Distillation', desc: 'Distill to a compact MobileNet-style student', icon: '🎓' },
];

const DATASETS = [
  { value: 'CIFAR10', label: 'CIFAR-10 (10 classes)' },
  { value: 'CIFAR100', label: 'CIFAR-100 (100 classes)' },
];

const STEP_ICONS: Record<string, string> = {
  loading_model: '\u2B07',
  loading_data: '\uD83D\uDCCA',
  compressing: '\u2699\uFE0F',
  energy_tracking: '\u26A1',
  evaluating: '\uD83D\uDCCF',
  complete: '\u2705',
};

const STRATEGY_LABELS: Record<string, string> = {
  pruning: 'Pruning',
  quantization: 'Quantization',
  hybrid: 'Hybrid',
  kd: 'Knowledge Distillation',
};

export function ModelDashboard({ model, modelKey, compressionResults, onNewResult, onClose }: ModelDashboardProps) {
      // Ensure baseline is included in results for graph
      const baselineResult = {
        strategy: 'baseline',
        model_name: model.model_name,
        model_key: modelKey,
        compression_method: 'baseline',
        dataset: model.dataset,
        input_size: model.input_size,
        baseline_accuracy: model.accuracy || 0,
        compressed_accuracy: model.accuracy || 0,
        size_MB: model.size_MB || 0,
        baseline_size_MB: model.size_MB || 0,
        size_reduction_percent: 0,
        latency_ms: model.latency_ms || 0,
        emissions_kg: 0,
        flops: 0,
        flops_M: 0,
        sparsity_percent: 0,
        total_params: model.total_params || 0,
        nonzero_params: model.total_params || 0,
        pruning_amount: 0,
        quantization_type: '',
        pipeline: '',
        student_params: 0,
        teacher_params: 0,
        param_reduction_percent: 0,
        kd_epochs: 0,
        fine_tune_epochs: 0,
      };
      const resultsWithBaseline = [baselineResult, ...compressionResults];
    // For compress all methods
    const [allLoading, setAllLoading] = useState(false);
    const [allStep, setAllStep] = useState<number>(0);
    const [allError, setAllError] = useState<string | null>(null);
    const [allStatus, setAllStatus] = useState<CompressionStatus | null>(null);
    const [allCurrentMethod, setAllCurrentMethod] = useState<string | null>(null);
    const [allSuccess, setAllSuccess] = useState(false);
    // Methods to run for Compress All
    const ALL_METHODS = ['pruning', 'quantization', 'hybrid', 'kd'];

    // Sequentially compress all methods
    const handleCompressAll = async () => {
      setAllLoading(true);
      setAllStep(0);
      setAllError(null);
      setAllStatus(null);
      setAllCurrentMethod(null);
      setAllSuccess(false);
      for (let i = 0; i < ALL_METHODS.length; ++i) {
        const m = ALL_METHODS[i];
        setAllStep(i);
        setAllCurrentMethod(m);
        setAllStatus(null);
        try {
          await compressPreloaded(modelKey, m, dataset, epochs);
          // Poll for completion of this method
          await new Promise<void>((resolve, reject) => {
            let tries = 0;
            const poll = async () => {
              try {
                const s = await getCompressionStatus();
                setAllStatus(s);
                if (!s.running) {
                  if (s.error) {
                    setAllError(s.error);
                    setAllLoading(false);
                    reject(new Error(s.error));
                    return;
                  } else if (s.result) {
                    onNewResult(s.result);
                    resolve();
                    return;
                  }
                }
                tries++;
                // Unlimited timeout: do not stop polling
                setTimeout(poll, 1500);
              } catch (e: any) {
                setAllError(e.message || 'Polling failed');
                setAllLoading(false);
                reject(e);
              }
            };
            poll();
          });
        } catch (err: any) {
          setAllError(err.message || 'Compression failed');
          setAllLoading(false);
          return;
        }
      }
      setAllLoading(false);
      setAllSuccess(true);
      setAllCurrentMethod(null);
      setTimeout(() => setAllSuccess(false), 3500);
      // Ensure baseline is included in results for graph
      const baselineResult = {
        strategy: 'baseline',
        model_name: model.model_name,
        model_key: modelKey,
        compression_method: 'baseline',
        dataset: model.dataset,
        input_size: model.input_size,
        baseline_accuracy: model.accuracy || 0,
        compressed_accuracy: model.accuracy || 0,
        size_MB: model.size_MB || 0,
        baseline_size_MB: model.size_MB || 0,
        size_reduction_percent: 0,
        latency_ms: model.latency_ms || 0,
        emissions_kg: 0,
        flops: 0,
        flops_M: 0,
        sparsity_percent: 0,
        total_params: model.total_params || 0,
        nonzero_params: model.total_params || 0,
        pruning_amount: 0,
        quantization_type: '',
        pipeline: '',
        student_params: 0,
        teacher_params: 0,
        param_reduction_percent: 0,
        kd_epochs: 0,
        fine_tune_epochs: 0,
      };

      const resultsWithBaseline = [baselineResult, ...compressionResults];
    };
  const [method, setMethod] = useState('pruning');
  const [dataset, setDataset] = useState('CIFAR10');
  const [epochs, setEpochs] = useState(5);
  const [loading, setLoading] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<CompressionStatus | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const isReady = model.status === 'ready';

  // Cleanup polling
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getCompressionStatus();
        setStatus(s);
        if (!s.running) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          if (s.error) {
            setError(s.error);
            setLoading(false);
          } else if (s.result) {
            onNewResult(s.result);
            setLoading(false);
          } else {
            setLoading(false);
            setStatus(null);
          }
        }
      } catch { /* ignore */ }
    }, 1500);
  }, [onNewResult]);

  const handleCompress = async () => {
    setLoading(true);
    setError(null);
    setStatus(null);
    try {
      await compressPreloaded(modelKey, method, dataset, epochs);
      startPolling();
    } catch (err: any) {
      setError(err.message || 'Compression failed');
      setLoading(false);
    }
  };

  const handlePrepare = async () => {
    setPreparing(true);
    setError(null);
    try {
      await triggerPrepare([modelKey]);
    } catch (err: any) {
      setError(err.message || 'Preparation failed');
    } finally {
      setPreparing(false);
    }
  };

  // Step indicators
  const steps = status?.steps || [];
  const currentStepIdx = steps.findIndex(s => s.key === status?.step);

  // Group results by method
  const resultsByMethod: Record<string, DynamicResult> = {};
  compressionResults.forEach(r => {
    const m = r.strategy || r.compression_method || 'unknown';
    resultsByMethod[m] = r; // keep latest
  });

  return (
    <div className="card border-2 border-green-200 bg-green-50/30">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
            <span className="text-green-700 font-bold text-lg">
              {model.model_name.charAt(0)}
            </span>
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">{model.model_name}</h2>
            <p className="text-sm text-gray-500">
              {model.params_label} params · {model.input_size}×{model.input_size} input · {model.dataset}
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors text-gray-400 hover:text-gray-700"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Baseline Metrics */}
      {isReady ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          <MetricCard label="Baseline Accuracy" value={`${model.accuracy}%`} color="text-blue-600" />
          <MetricCard label="Model Size" value={`${model.size_MB} MB`} color="text-purple-600" />
          <MetricCard label="Latency" value={`${model.latency_ms} ms`} color="text-amber-600" />
          <MetricCard
            label="Parameters"
            value={model.total_params ? `${(model.total_params / 1e6).toFixed(1)}M` : model.params_label}
            color="text-gray-600"
          />
        </div>
      ) : (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6 text-center">
          <p className="text-sm text-amber-800 mb-2">
            This model has not been prepared yet. Prepare it to see baseline metrics.
          </p>
          <button
            onClick={handlePrepare}
            disabled={preparing}
            className={`btn-primary ${preparing ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {preparing ? 'Preparing...' : `Prepare ${model.model_name}`}
          </button>
        </div>
      )}

      {/* Compression Options */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
        <h3 className="text-sm font-semibold text-gray-900 mb-3">Compress This Model</h3>

        {/* Method selector */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
          {METHODS.map((m) => {
            const hasResult = !!resultsByMethod[m.value];
            return (
              <button
                key={m.value}
                onClick={() => setMethod(m.value)}
                className={`relative p-3 rounded-lg border text-left transition-all ${
                  method === m.value
                    ? 'border-green-400 bg-green-50 ring-1 ring-green-300'
                    : 'border-gray-200 bg-white hover:border-gray-300'
                }`}
              >
                <span className="text-base">{m.icon}</span>
                <p className="text-xs font-medium text-gray-900 mt-1">{m.label}</p>
                <p className="text-[10px] text-gray-500 mt-0.5 line-clamp-2">{m.desc}</p>
                {hasResult && (
                  <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-green-400" />
                )}
              </button>
            );
          })}
        </div>

        {/* Dataset + Epochs */}
        <div className="flex flex-wrap gap-3 mb-4">
          <div className="flex-1 min-w-[150px]">
            <label className="text-xs text-gray-500 mb-1 block">Dataset</label>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg bg-white focus:ring-2 focus:ring-green-300 focus:border-green-400 outline-none"
            >
              {DATASETS.map(d => (
                <option key={d.value} value={d.value}>{d.label}</option>
              ))}
            </select>
          </div>
          <div className="w-24">
            <label className="text-xs text-gray-500 mb-1 block">Epochs</label>
            <input
              type="number"
              min={1}
              max={20}
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg bg-white focus:ring-2 focus:ring-green-300 focus:border-green-400 outline-none"
            />
          </div>
        </div>

        {/* Compress buttons */}
        <div className="flex flex-col gap-2">
          <button
            onClick={handleCompress}
            disabled={loading || allLoading}
            className={`w-full ${loading || allLoading ? 'btn-secondary opacity-60 cursor-not-allowed' : 'btn-primary'} rounded-full`}
          >
            {allLoading && allCurrentMethod
              ? `Compressing with ${METHODS.find(m => m.value === allCurrentMethod)?.label || allCurrentMethod}`
              : loading
                ? 'Compressing...'
                : `Compress with ${METHODS.find(m => m.value === method)?.label}`}
          </button>
          <button
            onClick={handleCompressAll}
            disabled={loading || allLoading}
            className={`w-full ${allLoading || loading ? 'btn-secondary opacity-60 cursor-not-allowed' : 'btn-primary'} rounded-full`}
          >
            {allLoading ? (
              <span className="flex items-center gap-2 justify-center">
                <span className="animate-spin inline-block w-4 h-4 border-2 border-green-400 border-t-transparent rounded-full"></span>
                Compress by All Methods
              </span>
            ) : 'Compress by All Methods'}
          </button>
        </div>
            {/* Compress All progress */}
            {allLoading && allCurrentMethod && (
              <div className="bg-white rounded-lg border border-green-200 p-4 mb-6">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium text-gray-700">Compress All Progress</span>
                  <span className="text-xs text-green-600 font-mono">
                    {allStep + 1}/{ALL_METHODS.length}
                  </span>
                </div>
                <div className="mb-2">
                  <span className="inline-flex items-center gap-2 text-xs font-semibold text-green-700">
                    <span className="animate-spin inline-block w-4 h-4 border-2 border-green-400 border-t-transparent rounded-full"></span>
                    {METHODS.find(m => m.value === allCurrentMethod)?.label || allCurrentMethod}
                  </span>
                </div>
                {allStatus && allStatus.steps && allStatus.steps.length > 0 && (
                  <div className="space-y-2">
                    {allStatus.steps.map((step, idx) => {
                      const isDone = idx < allStatus.steps.findIndex(s => s.key === allStatus.step) || allStatus.step === 'complete';
                      const isActive = idx === allStatus.steps.findIndex(s => s.key === allStatus.step) && allStatus.step !== 'complete';
                      return (
                        <div key={step.key} className="flex items-center gap-3">
                          <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs flex-shrink-0 transition-all duration-300 ${
                            isDone ? 'bg-green-500 text-white' :
                            isActive ? 'bg-green-100 border-2 border-green-500 text-green-700' :
                            'bg-gray-200 text-gray-400'
                          }`}>
                            {isDone ? '\u2713' : STEP_ICONS[step.key] || (idx + 1)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className={`text-sm font-medium truncate ${
                              isDone ? 'text-green-700' : isActive ? 'text-gray-900' : 'text-gray-400'
                            }`}>{step.label}</p>
                            {isActive && allStatus?.detail && (
                              <p className="text-xs text-green-600 truncate animate-pulse">{allStatus.detail}</p>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Success notification */}
            {allSuccess && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-6 animate-fade-in">
                <p className="text-green-700 font-semibold flex items-center gap-2">
                  <span className="inline-block w-5 h-5 text-green-600">✅</span>
                  All compression methods completed successfully!
                </p>
              </div>
            )}

            {/* Error for Compress All */}
            {allError && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-6">
                <p className="text-sm text-red-700">{allError}</p>
              </div>
            )}
      </div>

      {/* Progress Tracker */}
      {loading && status && steps.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-700">Compression Progress</span>
            <span className="text-xs text-green-600 font-mono">
              {currentStepIdx >= 0 ? `${currentStepIdx + 1}/${steps.length}` : '...'}
            </span>
          </div>
          <div className="space-y-2">
            {steps.map((step, idx) => {
              const isDone = idx < currentStepIdx || status?.step === 'complete';
              const isActive = idx === currentStepIdx && status?.step !== 'complete';
              return (
                <div key={step.key} className="flex items-center gap-3">
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs flex-shrink-0 transition-all duration-300 ${
                    isDone ? 'bg-green-500 text-white' :
                    isActive ? 'bg-green-100 border-2 border-green-500 text-green-700' :
                    'bg-gray-200 text-gray-400'
                  }`}>
                    {isDone ? '\u2713' : STEP_ICONS[step.key] || (idx + 1)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-medium truncate ${
                      isDone ? 'text-green-700' : isActive ? 'text-gray-900' : 'text-gray-400'
                    }`}>{step.label}</p>
                    {isActive && status?.detail && (
                      <p className="text-xs text-green-600 truncate animate-pulse">{status.detail}</p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-6">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Compression Results for this model */}
      {resultsWithBaseline.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="text-sm font-semibold text-gray-900 mb-3">
            Compression Results
            <span className="ml-2 text-gray-400 font-normal">({resultsWithBaseline.length})</span>
          </h3>

          <div className="space-y-3">
            {[...resultsWithBaseline].reverse().map((r, i) => {
              const methodLabel = STRATEGY_LABELS[r.strategy] || r.strategy;
              const accDiff = r.compressed_accuracy - r.baseline_accuracy;
              return (
                <ResultCard key={i} result={r} methodLabel={methodLabel} accDiff={accDiff} />
              );
            })}
          </div>

          {/* Comparison summary */}
          {resultsWithBaseline.length >= 2 && (
            <div className="mt-4 pt-4 border-t border-gray-100">
              <h4 className="text-xs font-medium text-gray-500 mb-2 uppercase tracking-wide">
                Method Comparison
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-100">
                      <th className="text-left py-2 pr-3 font-medium text-gray-500">Method</th>
                      <th className="text-right py-2 px-2 font-medium text-gray-500">Acc%</th>
                      <th className="text-right py-2 px-2 font-medium text-gray-500">Size</th>
                      <th className="text-right py-2 px-2 font-medium text-gray-500">↓ Size</th>
                      <th className="text-right py-2 px-2 font-medium text-gray-500">CO₂</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...resultsWithBaseline].reverse().map((r, i) => (
                      <tr key={i} className="border-b border-gray-50">
                        <td className="py-2 pr-3 font-medium text-gray-900">
                          {STRATEGY_LABELS[r.strategy] || r.strategy}
                        </td>
                        <td className="text-right py-2 px-2 font-mono text-gray-700">
                          {r.compressed_accuracy}%
                        </td>
                        <td className="text-right py-2 px-2 font-mono text-gray-700">
                          {r.size_MB} MB
                        </td>
                        <td className="text-right py-2 px-2 font-mono text-green-600">
                          ↓{r.size_reduction_percent.toFixed(1)}%
                        </td>
                        <td className="text-right py-2 px-2 font-mono text-gray-500">
                          {r.emissions_kg?.toFixed(6) || '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-white rounded-lg border border-gray-100 p-3 text-center">
      <p className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</p>
      <p className={`text-lg font-bold ${color} mt-1`}>{value}</p>
    </div>
  );
}

function ResultCard({
  result,
  methodLabel,
  accDiff,
}: {
  result: DynamicResult;
  methodLabel: string;
  accDiff: number;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-gray-100 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-gray-50 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <span className={`inline-flex px-2 py-0.5 rounded text-[10px] font-medium ${
            result.size_reduction_percent > 50 ? 'bg-green-100 text-green-700' :
            result.size_reduction_percent > 20 ? 'bg-blue-100 text-blue-700' :
            'bg-gray-100 text-gray-700'
          }`}>
            {methodLabel}
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="font-mono text-green-600">↓{result.size_reduction_percent.toFixed(1)}%</span>
          <span className={`font-mono ${accDiff >= -1 ? 'text-blue-600' : 'text-red-500'}`}>
            {result.compressed_accuracy}%
          </span>
          <span className="font-mono text-gray-500">{result.size_MB} MB</span>
          <svg className={`w-3.5 h-3.5 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {expanded && (
        <div className="px-3 pb-3 border-t border-gray-50">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-2">
            <MiniMetric label="Baseline Acc." value={`${result.baseline_accuracy}%`} />
            <MiniMetric label="Compressed Acc." value={`${result.compressed_accuracy}%`} />
            <MiniMetric label="Baseline Size" value={`${result.baseline_size_MB} MB`} />
            <MiniMetric label="Latency" value={`${result.latency_ms} ms`} />
          </div>

          {/* Size bar */}
          <div className="mt-2">
            <div className="flex justify-between text-[10px] text-gray-400 mb-0.5">
              <span>{result.baseline_size_MB} MB</span>
              <span>{result.size_MB} MB</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div
                className="bg-green-500 h-1.5 rounded-full transition-all"
                style={{
                  width: `${Math.max(5, (result.size_MB / result.baseline_size_MB) * 100)}%`,
                }}
              />
            </div>
          </div>

          {/* Details */}
          <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] bg-gray-50 rounded p-2">
            {result.sparsity_percent != null && result.sparsity_percent > 0 && (
              <DetailRow label="Sparsity" value={`${result.sparsity_percent}%`} />
            )}
            {result.quantization_type && (
              <DetailRow label="Quantization" value={result.quantization_type} />
            )}
            {result.emissions_kg != null && (
              <DetailRow label="CO₂" value={`${result.emissions_kg.toFixed(6)} kg`} />
            )}
            {result.total_params != null && (
              <DetailRow label="Params" value={result.total_params.toLocaleString()} />
            )}
            {result.dataset && (
              <DetailRow label="Dataset" value={result.dataset} />
            )}
            {result.pipeline && (
              <div className="col-span-2">
                <DetailRow label="Pipeline" value={result.pipeline} />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 rounded p-1.5 text-center">
      <p className="text-[9px] text-gray-500">{label}</p>
      <p className="text-[11px] font-bold text-gray-700">{value}</p>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-900 font-medium text-right">{value}</span>
    </>
  );
}
