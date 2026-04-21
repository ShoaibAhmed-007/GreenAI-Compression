'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BaselineModel,
  DynamicResult,
  CompressionStatus,
  compressPreloaded,
  getCompressionStatus,
  getCompressionHistory,
  getModelComparisonOptions,
  loadSavedResults,
} from '@/lib/api';
import { CompressionResults } from './CompressionResults';

interface CompressionDialogProps {
  open: boolean;
  modelKey: string | null;
  model: BaselineModel | null;
  onClose: () => void;
  onNewResult: (result: DynamicResult) => void;
}

const TECHNIQUES = [
  { key: 'smart', label: 'Auto-Green (Smart)' },
  { key: 'maximize_speed', label: 'Preset: Max Speed' },
  { key: 'minimize_size', label: 'Preset: Min Size' },
  { key: 'preserve_accuracy', label: 'Preset: Preserve Acc.' },
  { key: 'pruning', label: 'Pruning' },
  { key: 'quantization', label: 'Quantization' },
  { key: 'hybrid', label: 'Hybrid' },
  { key: 'kd', label: 'Distillation' },
] as const;

const TECHNIQUE_RESULT_FALLBACKS: Record<string, string[]> = {
  smart: ['smart', 'maximize_speed', 'minimize_size', 'preserve_accuracy', 'quantization', 'hybrid', 'kd', 'pruning'],
  maximize_speed: ['maximize_speed', 'quantization', 'hybrid'],
  minimize_size: ['minimize_size', 'hybrid', 'kd', 'pruning', 'quantization'],
  preserve_accuracy: ['preserve_accuracy', 'kd', 'quantization', 'hybrid', 'pruning'],
  pruning: ['pruning'],
  quantization: ['quantization'],
  hybrid: ['hybrid'],
  kd: ['kd'],
};

function normalizeModelKey(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/-/g, '_')
    .replace(/\s+/g, '_')
    .replace(/[^a-z0-9_]/g, '');
}

function normalizeTechnique(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/apply_/g, '')
    .replace(/-/g, '_')
    .replace(/\s+/g, '_')
    .replace(/[^a-z0-9_]/g, '');
}

function canonicalTechnique(value: string): string {
  const normalized = normalizeTechnique(value);

  const aliases: Record<string, string> = {
    smart_router: 'smart',
    auto_green: 'smart',
    auto: 'smart',
    max_speed: 'maximize_speed',
    speed: 'maximize_speed',
    fastest: 'maximize_speed',
    min_size: 'minimize_size',
    smallest: 'minimize_size',
    preserve_acc: 'preserve_accuracy',
    preserve: 'preserve_accuracy',
    prune: 'pruning',
    pruned: 'pruning',
    pruning_sparse: 'pruning',
    magnitude_pruning: 'pruning',
    quantized: 'quantization',
    quantization_static: 'quantization',
    quantization_dynamic: 'quantization',
    static_quantization: 'quantization',
    dynamic_quantization: 'quantization',
    int8_quantization: 'quantization',
    knowledge_distillation: 'kd',
    distillation: 'kd',
    kd_compact_student: 'kd',
    hybrid_student_quant: 'hybrid',
  };

  return aliases[normalized] || normalized;
}

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

function scoreResultQuality(result: DynamicResult): number {
  let score = 0;
  if (toFiniteNumber(result.compressed_accuracy) != null) score += 2;
  if (toFiniteNumber(result.size_MB) != null || toFiniteNumber(result.compressed_size_MB) != null) score += 2;
  if (
    toFiniteNumber(result.compressed_total_emissions_kg) != null ||
    toFiniteNumber(result.training_co2_kg) != null ||
    toFiniteNumber(result.inference_co2_kg) != null ||
    toFiniteNumber(result.co2_kg) != null ||
    toFiniteNumber(result.emissions_kg) != null
  ) {
    score += 2;
  }
  if (toFiniteNumber(result.compressed_latency_ms) != null || toFiniteNumber(result.latency_ms) != null) score += 1;
  return score;
}

function pickLatestResultPerTechnique(results: DynamicResult[]): Record<string, DynamicResult> {
  const latestByTechnique: Record<string, DynamicResult> = {};

  for (const result of results) {
    const key = canonicalTechnique(String(result.strategy || result.compression_method || ''));
    if (!key) continue;

    const current = latestByTechnique[key];
    if (!current) {
      latestByTechnique[key] = result;
      continue;
    }

    const incomingFromDirect = typeof result.source_result_file === 'string' && result.source_result_file.trim() !== '';
    const currentFromDirect = typeof current.source_result_file === 'string' && current.source_result_file.trim() !== '';
    if (incomingFromDirect && !currentFromDirect) {
      latestByTechnique[key] = result;
      continue;
    }
    if (!incomingFromDirect && currentFromDirect) {
      continue;
    }

    const currentTs = Date.parse(String(current.saved_at || ''));
    const incomingTs = Date.parse(String(result.saved_at || ''));

    if (Number.isFinite(incomingTs) && !Number.isFinite(currentTs)) {
      latestByTechnique[key] = result;
      continue;
    }
    if (Number.isFinite(incomingTs) && Number.isFinite(currentTs) && incomingTs >= currentTs) {
      latestByTechnique[key] = result;
      continue;
    }

    if (!Number.isFinite(incomingTs) && !Number.isFinite(currentTs)) {
      if (scoreResultQuality(result) >= scoreResultQuality(current)) {
        latestByTechnique[key] = result;
      }
    }
  }

  return latestByTechnique;
}

function resolveTechniqueResult(
  techniqueKey: string,
  resultsByTechnique: Record<string, DynamicResult>
): DynamicResult | null {
  const candidates = TECHNIQUE_RESULT_FALLBACKS[techniqueKey] || [techniqueKey];
  for (const key of candidates) {
    const candidate = resultsByTechnique[key];
    if (candidate) {
      return candidate;
    }
  }
  return null;
}

function formatPercent(value?: number): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Not Available';
  return `${value.toFixed(2)}%`;
}

function formatSize(value?: number): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Not Available';
  return `${value.toFixed(2)} MB`;
}

function formatCo2(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Not Available';
  if (value > 0 && value < 0.000001) return '<0.000001 kg';
  return `${value.toFixed(6)} kg`;
}

export function CompressionDialog({
  open,
  modelKey,
  model,
  onClose,
  onNewResult,
}: CompressionDialogProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultsByTechnique, setResultsByTechnique] = useState<Record<string, DynamicResult>>({});
  const [selectedTechnique, setSelectedTechnique] = useState<string>('smart');
  const [runningMode, setRunningMode] = useState<'single' | 'all' | null>(null);
  const [runningMethod, setRunningMethod] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number>(0);
  const elapsedRef = useRef<NodeJS.Timeout | null>(null);

  const allTechniqueKeys = useMemo(() => TECHNIQUES.map((t) => t.key), []);
  const techniqueKeySet = useMemo(() => new Set<string>(allTechniqueKeys), [allTechniqueKeys]);
  const isReadyModel = model?.status === 'ready';
  const isRunning = runningMode !== null;

  const wait = useCallback((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)), []);

  const startElapsedTimer = useCallback(() => {
    setElapsed(0);
    if (elapsedRef.current) clearInterval(elapsedRef.current);
    elapsedRef.current = setInterval(() => {
      setElapsed((s) => s + 1);
    }, 1000);
  }, []);

  const stopElapsedTimer = useCallback(() => {
    if (elapsedRef.current) {
      clearInterval(elapsedRef.current);
      elapsedRef.current = null;
    }
    setElapsed(0);
  }, []);

  // Cleanup timer on unmount
  useEffect(() => () => stopElapsedTimer(), [stopElapsedTimer]);

  const pollCompressionResult = useCallback(async (): Promise<DynamicResult> => {
    // No timeout — poll indefinitely until backend signals done or error.
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const status: CompressionStatus = await getCompressionStatus();
      setRunStatus(status.detail || status.progress || null);

      if (status.running) {
        await wait(1500);
        continue;
      }

      if (status.error) {
        throw new Error(status.error);
      }

      if (status.result) {
        return status.result;
      }

      // Backend finished but no result — wait one more cycle before giving up
      await wait(1500);
      const retry = await getCompressionStatus();
      if (retry.result) return retry.result;
      throw new Error('Compression finished without a result payload.');
    }
  }, [wait]);

  const loadCompressionData = useCallback(async () => {
    if (!modelKey) return;

    try {
      setLoading(true);
      setError(null);

      const [historyPayloadResult, comparisonOptionsResult] = await Promise.allSettled([
        getCompressionHistory(),
        getModelComparisonOptions(),
      ]);

      const normalizedTarget = normalizeModelKey(modelKey);

      const historyEntries = historyPayloadResult.status === 'fulfilled'
        ? Object.entries(historyPayloadResult.value.history || {})
          .filter(([key]) => normalizeModelKey(key) === normalizedTarget)
          .flatMap(([, entries]) => entries || [])
        : [];

      const persistedEntries = loadSavedResults().filter((entry) => {
        const entryModelKey = normalizeModelKey(String(entry.model_key || entry.model_name || ''));
        return entryModelKey === normalizedTarget;
      });

      const optionFallbackEntries: DynamicResult[] = comparisonOptionsResult.status === 'fulfilled'
        ? (comparisonOptionsResult.value.compressed_models || [])
          .filter((option) => normalizeModelKey(String(option.model_key || '')) === normalizedTarget)
          .map((option) => {
            const canonical = canonicalTechnique(String(option.strategy || ''));
            const sizeMB =
              toFiniteNumber(option.compressed_size_MB) ??
              toFiniteNumber(option.size_MB) ??
              Number.NaN;
            const baselineSize =
              toFiniteNumber(option.baseline_size_MB) ??
              toFiniteNumber(model?.size_MB);
            const baselineAccuracy =
              toFiniteNumber(option.baseline_accuracy) ??
              toFiniteNumber(option.observed_baseline_accuracy_percent) ??
              toFiniteNumber(model?.accuracy) ??
              Number.NaN;
            const compressedAccuracy =
              toFiniteNumber(option.compressed_accuracy) ??
              toFiniteNumber(option.observed_compressed_accuracy_percent) ??
              toFiniteNumber(option.expected_compressed_accuracy_percent) ??
              Number.NaN;
            const baselineLatency = toFiniteNumber(option.baseline_latency_ms);
            const compressedLatency =
              toFiniteNumber(option.compressed_latency_ms) ??
              toFiniteNumber(option.latency_ms);
            const sizeReduction =
              toFiniteNumber(option.size_reduction_percent) ??
              (baselineSize != null && Number.isFinite(sizeMB) && baselineSize > 0
                ? ((sizeMB - baselineSize) / baselineSize) * -100
                : Number.NaN);
            const baselineTrainingCo2 =
              toFiniteNumber(option.training_co2_kg) ??
              toFiniteNumber(model?.training_co2_kg);
            const baselineTotalCo2 =
              toFiniteNumber(option.baseline_total_emissions_kg) ??
              baselineTrainingCo2;
            const compressedTotalCo2 =
              toFiniteNumber(option.compressed_total_emissions_kg) ??
              toFiniteNumber(option.co2_kg);
            const baselineTrainingEnergy =
              toFiniteNumber(option.training_energy_kwh) ??
              toFiniteNumber(model?.training_energy_kwh);
            const baselineTotalEnergy =
              toFiniteNumber(option.baseline_total_energy_kwh) ??
              baselineTrainingEnergy;
            const compressedTotalEnergy =
              toFiniteNumber(option.compressed_total_energy_kwh) ??
              toFiniteNumber(option.energy_kwh);

            return {
              strategy: canonical,
              compression_method: canonical,
              model_name: option.model_name,
              model_key: option.model_key,
              source_result_file: option.source_result_file || undefined,
              saved_at: option.saved_at || option.summary_generated_at || undefined,
              baseline_accuracy: baselineAccuracy,
              compressed_accuracy: compressedAccuracy,
              size_MB: sizeMB,
              compressed_size_MB: sizeMB,
              baseline_size_MB: baselineSize ?? Number.NaN,
              size_reduction_percent: sizeReduction,
              baseline_latency_ms: baselineLatency ?? undefined,
              compressed_latency_ms: compressedLatency ?? undefined,
              latency_ms: compressedLatency ?? undefined,
              latency_speedup_percent: toFiniteNumber(option.latency_speedup_percent) ?? undefined,
              baseline_training_co2_kg: baselineTrainingCo2 ?? undefined,
              baseline_total_emissions_kg: baselineTotalCo2 ?? undefined,
              compressed_total_emissions_kg: compressedTotalCo2 ?? undefined,
              emissions_kg: compressedTotalCo2 ?? Number.NaN,
              co2_kg: compressedTotalCo2 ?? undefined,
              baseline_training_energy_kwh: baselineTrainingEnergy ?? undefined,
              baseline_total_energy_kwh: baselineTotalEnergy ?? undefined,
              compressed_total_energy_kwh: compressedTotalEnergy ?? undefined,
              energy_kwh: compressedTotalEnergy ?? undefined,
              training_energy_kwh: baselineTrainingEnergy ?? undefined,
              inference_co2_kg: toFiniteNumber(option.inference_co2_kg) ?? undefined,
              inference_energy_kwh: toFiniteNumber(option.inference_energy_kwh) ?? undefined,
              emissions_reduction_percent: toFiniteNumber(option.emissions_reduction_percent) ?? undefined,
              energy_reduction_percent: toFiniteNumber(option.energy_reduction_percent) ?? undefined,
              co2_method: option.co2_method || undefined,
              bottleneck_analysis: option.bottleneck_analysis || undefined,
            };
          })
        : [];

      const mergedEntries: DynamicResult[] = [
        ...historyEntries,
        ...persistedEntries,
        ...optionFallbackEntries,
      ];

      const normalizedResults = mergedEntries
        .map((entry) => {
          const canonical = canonicalTechnique(String(entry.strategy || entry.compression_method || ''));
          return {
            ...entry,
            strategy: canonical,
            compression_method: canonical,
          };
        })
        .filter((entry) => {
          const key = canonicalTechnique(String(entry.strategy || entry.compression_method || ''));
          return key !== 'baseline' && techniqueKeySet.has(key);
        });

      const latest = pickLatestResultPerTechnique(normalizedResults);
      setResultsByTechnique(latest);

      const firstAvailable = TECHNIQUES.find((t) => Boolean(latest[t.key]))?.key || TECHNIQUES[0].key;
      setSelectedTechnique(firstAvailable);

      if (historyPayloadResult.status === 'rejected' && comparisonOptionsResult.status === 'rejected' && persistedEntries.length === 0) {
        setError('Failed to fetch compression results for selected model.');
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch compression results for selected model.');
      setResultsByTechnique({});
    } finally {
      setLoading(false);
    }
  }, [model, modelKey, techniqueKeySet]);

  useEffect(() => {
    if (!open || !modelKey) return;

    loadCompressionData();
  }, [open, modelKey, loadCompressionData]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  const selectedResult = useMemo(
    () => resolveTechniqueResult(selectedTechnique, resultsByTechnique),
    [resultsByTechnique, selectedTechnique]
  );

  const handleCompressSelected = useCallback(async () => {
    if (!modelKey || !model || !isReadyModel || isRunning) return;

    try {
      setRunError(null);
      setRunStatus('Starting compression...');
      setRunningMode('single');
      setRunningMethod(selectedTechnique);
      startElapsedTimer();

      await compressPreloaded(modelKey, selectedTechnique, model.dataset || 'CIFAR10', 5);
      const result = await pollCompressionResult();

      onNewResult(result);
      await loadCompressionData();
      setRunStatus('Compression complete.');
    } catch (err: any) {
      setRunError(err?.message || 'Compression failed.');
    } finally {
      stopElapsedTimer();
      setRunningMode(null);
      setRunningMethod(null);
    }
  }, [isReadyModel, isRunning, loadCompressionData, model, modelKey, onNewResult, pollCompressionResult, selectedTechnique, startElapsedTimer, stopElapsedTimer]);

  const handleCompressAll = useCallback(async () => {
    if (!modelKey || !model || !isReadyModel || isRunning) return;

    try {
      setRunError(null);
      setRunningMode('all');
      startElapsedTimer();

      for (const technique of allTechniqueKeys) {
        setRunningMethod(technique);
        setRunStatus(`Starting ${technique}...`);

        await compressPreloaded(modelKey, technique, model.dataset || 'CIFAR10', 5);
        const result = await pollCompressionResult();
        onNewResult(result);
      }

      await loadCompressionData();
      setRunStatus('All compression techniques completed.');
    } catch (err: any) {
      setRunError(err?.message || 'Compress-by-all failed.');
    } finally {
      stopElapsedTimer();
      setRunningMode(null);
      setRunningMethod(null);
    }
  }, [allTechniqueKeys, isReadyModel, isRunning, loadCompressionData, model, modelKey, onNewResult, pollCompressionResult, startElapsedTimer, stopElapsedTimer]);

  if (!modelKey || !model) return null;

  return (
    <section
      className={`overflow-hidden transition-all duration-300 ease-out ${
        open ? 'max-h-[2200px] opacity-100 translate-y-0' : 'max-h-0 opacity-0 -translate-y-2 pointer-events-none'
      }`}
      aria-hidden={!open}
    >
      <div className="bg-surface-container rounded-2xl overflow-hidden ghost-border ring-1 ring-primary/10">
        {/* Header */}
        <div className="sticky top-0 z-10 bg-surface-container/95 backdrop-blur px-6 py-4 flex items-center justify-between" style={{ borderBottom: '1px solid rgba(66, 73, 62, 0.15)' }}>
          <div>
            <h3 className="text-lg font-headline font-semibold text-on-surface">{model.model_name} Details</h3>
            <p className="text-xs text-on-surface-variant/60 mt-0.5 font-technical">Model key: {modelKey}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-surface-container-high text-on-surface-variant hover:text-on-surface transition-colors"
            aria-label="Collapse model details"
          >
            <span className="material-symbols-outlined">keyboard_arrow_up</span>
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Baseline Details */}
          <section className="space-y-3">
            <h4 className="text-xs font-bold text-on-surface-variant uppercase tracking-widest">Baseline Inference Details</h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="bg-surface-container-low p-4 rounded-xl">
                <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Accuracy</p>
                <p className="text-sm font-technical font-semibold text-on-surface mt-1">{formatPercent(model.accuracy)}</p>
              </div>
              <div className="bg-surface-container-low p-4 rounded-xl">
                <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Size</p>
                <p className="text-sm font-technical font-semibold text-on-surface mt-1">{formatSize(model.size_MB)}</p>
              </div>
              <div className="bg-surface-container-low p-4 rounded-xl">
                <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-technical">Training CO2</p>
                <p className="text-sm font-technical font-semibold text-primary mt-1">{formatCo2(model.training_co2_kg)}</p>
              </div>
            </div>
          </section>

          {/* Technique Selector */}
          <section className="space-y-3">
            <h4 className="text-xs font-bold text-on-surface-variant uppercase tracking-widest">Compression Techniques</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
              {TECHNIQUES.map((technique) => {
                const hasResult = Boolean(resolveTechniqueResult(technique.key, resultsByTechnique));
                const isActive = selectedTechnique === technique.key;
                return (
                  <button
                    key={technique.key}
                    type="button"
                    onClick={() => setSelectedTechnique(technique.key)}
                    className={`rounded-xl p-3 text-left transition-all ${
                      isActive
                        ? 'bg-primary text-on-primary'
                        : 'bg-surface-container-low hover:bg-surface-container-high text-on-surface-variant'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <p className="text-sm font-semibold">{technique.label}</p>
                      {isActive && hasResult && (
                        <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                      )}
                    </div>
                    <p className={`text-xs mt-0.5 ${isActive ? 'text-on-primary/70' : hasResult ? 'text-primary' : 'text-on-surface-variant/50'}`}>
                      {hasResult ? 'Result available' : 'No result yet'}
                    </p>
                  </button>
                );
              })}
            </div>

            <div className="bg-surface-container-low rounded-xl p-4 space-y-3">
              {!isReadyModel && (
                <p className="text-xs text-secondary">
                  This model is not ready yet. Prepare baselines first, then run compression.
                </p>
              )}

              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={handleCompressSelected}
                  disabled={!isReadyModel || isRunning}
                  className={`btn-primary ${!isReadyModel || isRunning ? 'opacity-60 cursor-not-allowed' : ''}`}
                >
                  {runningMode === 'single'
                    ? `Compressing ${runningMethod || selectedTechnique}...`
                    : `Compress ${TECHNIQUES.find((t) => t.key === selectedTechnique)?.label || selectedTechnique}`}
                </button>

                <button
                  type="button"
                  onClick={handleCompressAll}
                  disabled={!isReadyModel || isRunning}
                  className={`btn-secondary ${!isReadyModel || isRunning ? 'opacity-60 cursor-not-allowed' : ''}`}
                >
                  {runningMode === 'all'
                    ? `Compress by ALL (${runningMethod || 'starting'}...)`
                    : 'Compress by ALL'}
                </button>
              </div>

              {(runStatus || (isRunning && elapsed > 0)) && (
                <div className="flex items-center gap-3">
                  {isRunning && (
                    <div className="w-3.5 h-3.5 border-2 border-primary/30 border-t-primary rounded-full animate-spin flex-shrink-0" />
                  )}
                  <div className="min-w-0">
                    {runStatus && (
                      <p className="text-xs text-on-surface-variant font-technical">{runStatus}</p>
                    )}
                    {isRunning && elapsed > 0 && (
                      <p className="text-[10px] text-on-surface-variant/50 font-technical mt-0.5">
                        Elapsed: {Math.floor(elapsed / 60)}m {elapsed % 60}s — this may take several minutes, please wait
                      </p>
                    )}
                  </div>
                </div>
              )}

              {runError && (
                <p className="text-xs text-error">{runError}</p>
              )}
            </div>
          </section>

          {/* Results */}
          <section className="space-y-3">
            <h4 className="text-xs font-bold text-on-surface-variant uppercase tracking-widest">Compression Results</h4>

            {loading && (
              <div className="bg-surface-container-low rounded-xl p-4 flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                <p className="text-sm text-on-surface-variant">Loading compressed model results...</p>
              </div>
            )}

            {!loading && error && (
              <div className="bg-error-container/10 rounded-xl p-3 text-sm text-on-error-container ghost-border">
                {error}
              </div>
            )}

            {!loading && !error && (
              <CompressionResults result={selectedResult} />
            )}
          </section>
        </div>
      </div>
    </section>
  );
}
