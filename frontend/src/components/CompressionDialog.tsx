'use client';

import { useEffect, useMemo, useState } from 'react';
import { BaselineModel, DynamicResult, getCompressionHistory } from '@/lib/api';
import { CompressionResults } from './CompressionResults';

interface CompressionDialogProps {
  open: boolean;
  modelKey: string | null;
  model: BaselineModel | null;
  onClose: () => void;
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

export function CompressionDialog({
  open,
  modelKey,
  model,
  onClose,
}: CompressionDialogProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultsByTechnique, setResultsByTechnique] = useState<Record<string, DynamicResult>>({});
  const [selectedTechnique, setSelectedTechnique] = useState<string>('pruning');

  useEffect(() => {
    if (!open || !modelKey) return;

    const loadCompressionData = async () => {
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
    };

    loadCompressionData();
  }, [open, modelKey]);

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

  if (!open || !modelKey || !model) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true">
      <div className="absolute inset-0 bg-black/45" onClick={onClose} />

      <div className="relative w-full max-w-4xl max-h-[88vh] overflow-y-auto rounded-2xl bg-white shadow-2xl border border-gray-200">
        <div className="sticky top-0 z-10 bg-white/95 backdrop-blur border-b border-gray-200 px-5 py-4 flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{model.model_name} Compression Details</h3>
            <p className="text-xs text-gray-500 mt-0.5">Model key: {modelKey}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700"
            aria-label="Close compression dialog"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-5 space-y-5">
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
    </div>
  );
}
