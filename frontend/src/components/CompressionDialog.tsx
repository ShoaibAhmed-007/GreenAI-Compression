'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  BaselineModel,
  DynamicResult,
  CompressionStatus,
  compressPreloaded,
  getCompressionStatus,
  getCompressionHistory,
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
  { key: 'pruning', label: 'Pruning' },
  { key: 'quantization', label: 'Quantization' },
  { key: 'hybrid', label: 'Hybrid' },
  { key: 'kd', label: 'Distillation' },
] as const;

function normalizeModelKey(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, '_');
}

function normalizeTechnique(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, '_');
}

function pickLatestResultPerTechnique(results: DynamicResult[]): Record<string, DynamicResult> {
  const latestByTechnique: Record<string, DynamicResult> = {};

  for (const result of results) {
    const key = normalizeTechnique(String(result.strategy || result.compression_method || ''));
    if (!key) continue;

    const current = latestByTechnique[key];
    if (!current) {
      latestByTechnique[key] = result;
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
    }
  }

  return latestByTechnique;
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
  const [selectedTechnique, setSelectedTechnique] = useState<string>('pruning');
  const [runningMode, setRunningMode] = useState<'single' | 'all' | null>(null);
  const [runningMethod, setRunningMethod] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const allTechniqueKeys = useMemo(() => TECHNIQUES.map((t) => t.key), []);
  const isReadyModel = model?.status === 'ready';
  const isRunning = runningMode !== null;

  const wait = useCallback((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)), []);

  const pollCompressionResult = useCallback(async (): Promise<DynamicResult> => {
    const maxPolls = 800;

    for (let attempt = 0; attempt < maxPolls; attempt += 1) {
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

      throw new Error('Compression finished without a result payload.');
    }

    throw new Error('Compression timed out. Please try again.');
  }, [wait]);

  const loadCompressionData = useCallback(async () => {
    if (!modelKey) return;

    try {
      setLoading(true);
      setError(null);

      const historyPayload = await getCompressionHistory();
      const normalizedTarget = normalizeModelKey(modelKey);

      const matchingEntries = Object.entries(historyPayload.history || {})
        .filter(([key]) => normalizeModelKey(key) === normalizedTarget)
        .flatMap(([, entries]) => entries || []);

      const normalizedResults = matchingEntries.filter(
        (entry) => normalizeTechnique(String(entry.strategy || entry.compression_method || '')) !== 'baseline'
      );

      const latest = pickLatestResultPerTechnique(normalizedResults);
      setResultsByTechnique(latest);

      const firstAvailable = TECHNIQUES.find((t) => Boolean(latest[t.key]))?.key || TECHNIQUES[0].key;
      setSelectedTechnique(firstAvailable);
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch compression results for selected model.');
      setResultsByTechnique({});
    } finally {
      setLoading(false);
    }
  }, [modelKey]);

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
    () => resultsByTechnique[selectedTechnique] || null,
    [resultsByTechnique, selectedTechnique]
  );

  const handleCompressSelected = useCallback(async () => {
    if (!modelKey || !model || !isReadyModel || isRunning) return;

    try {
      setRunError(null);
      setRunStatus('Starting compression...');
      setRunningMode('single');
      setRunningMethod(selectedTechnique);

      await compressPreloaded(modelKey, selectedTechnique, model.dataset || 'CIFAR10', 5);
      const result = await pollCompressionResult();

      onNewResult(result);
      await loadCompressionData();
      setRunStatus('Compression complete.');
    } catch (err: any) {
      setRunError(err?.message || 'Compression failed.');
    } finally {
      setRunningMode(null);
      setRunningMethod(null);
    }
  }, [isReadyModel, isRunning, loadCompressionData, model, modelKey, onNewResult, pollCompressionResult, selectedTechnique]);

  const handleCompressAll = useCallback(async () => {
    if (!modelKey || !model || !isReadyModel || isRunning) return;

    try {
      setRunError(null);
      setRunningMode('all');

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
      setRunningMode(null);
      setRunningMethod(null);
    }
  }, [allTechniqueKeys, isReadyModel, isRunning, loadCompressionData, model, modelKey, onNewResult, pollCompressionResult]);

  if (!modelKey || !model) return null;

  return (
    <section
      className={`overflow-hidden transition-all duration-300 ease-out ${
        open ? 'max-h-[2200px] opacity-100 translate-y-0' : 'max-h-0 opacity-0 -translate-y-2 pointer-events-none'
      }`}
      aria-hidden={!open}
    >
      <div className="mx-auto w-full max-w-5xl rounded-2xl border border-gray-200 bg-white shadow-lg">
        <div className="sticky top-0 z-10 bg-white/95 backdrop-blur border-b border-gray-200 px-5 py-4 flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{model.model_name} Details</h3>
            <p className="text-xs text-gray-500 mt-0.5">Model key: {modelKey}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700"
            aria-label="Collapse model details"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-5 space-y-5">
          <section className="space-y-3">
            <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">Baseline Inference Details</h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <div className="rounded-lg border border-gray-200 bg-white p-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-500">Accuracy</p>
                <p className="text-sm font-semibold text-gray-900 mt-1">{formatPercent(model.accuracy)}</p>
              </div>
              <div className="rounded-lg border border-gray-200 bg-white p-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-500">Size</p>
                <p className="text-sm font-semibold text-gray-900 mt-1">{formatSize(model.size_MB)}</p>
              </div>
              <div className="rounded-lg border border-gray-200 bg-white p-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-500">Training CO2</p>
                <p className="text-sm font-semibold text-emerald-700 mt-1">{formatCo2(model.training_co2_kg)}</p>
              </div>
            </div>
          </section>

          <section className="space-y-3">
            <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">Compression Techniques</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
              {TECHNIQUES.map((technique) => {
                const hasResult = Boolean(resultsByTechnique[technique.key]);
                const isActive = selectedTechnique === technique.key;
                return (
                  <button
                    key={technique.key}
                    type="button"
                    onClick={() => setSelectedTechnique(technique.key)}
                    className={[
                      'rounded-lg border px-3 py-2 text-left transition-colors',
                      isActive
                        ? 'border-green-500 bg-green-50 text-green-900'
                        : 'border-gray-200 bg-white hover:border-green-300',
                    ].join(' ')}
                  >
                    <p className="text-sm font-semibold">{technique.label}</p>
                    <p className={`text-xs mt-0.5 ${hasResult ? 'text-green-700' : 'text-gray-500'}`}>
                      {hasResult ? 'Result available' : 'No result yet'}
                    </p>
                  </button>
                );
              })}
            </div>

            <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-3">
              {!isReadyModel && (
                <p className="text-xs text-amber-700">
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

              {runStatus && (
                <p className="text-xs text-gray-600">{runStatus}</p>
              )}

              {runError && (
                <p className="text-xs text-red-600">{runError}</p>
              )}
            </div>
          </section>

          <section className="space-y-3">
            <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">Compression Results</h4>

            {loading && (
              <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-green-200 border-t-green-600 rounded-full animate-spin" />
                <p className="text-sm text-gray-700">Loading compressed model results...</p>
              </div>
            )}

            {!loading && error && (
              <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
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
