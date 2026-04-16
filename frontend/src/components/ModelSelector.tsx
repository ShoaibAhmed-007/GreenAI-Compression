'use client';

import { useMemo } from 'react';
import { BaselineModel } from '@/lib/api';
import { ModelGrid } from './ModelGrid';
import { useSearch } from './TopNavbar';

/**
 * Canonical technique tokens supported by every model in the backend.
 * All 11 preloaded models can be compressed with all 4 methods —
 * see run_compression() in backend/compress.py.
 *
 * Additionally we include common aliases so that searching:
 *   "kd"           → matches 'knowledge distillation'
 *   "distill"      → matches 'knowledge distillation'
 *   "quant"        → matches 'quantization'
 *   "prune"        → matches 'pruning'
 */
const ALL_TECHNIQUES = [
  'pruning',
  'quantization',
  'hybrid',
  'knowledge distillation',
  'kd',
  'distillation',
] as const;

/**
 * Normalize a raw query/token:
 *  • trim whitespace
 *  • collapse internal spaces
 *  • lowercase
 */
function normalize(raw: string): string {
  return raw.trim().replace(/\s+/g, ' ').toLowerCase();
}

/**
 * Returns true when `model` (or any of the universal compression techniques)
 * matches the normalized query.
 *
 * Matching rules:
 *  1. Model name / key: substring match (so 'res' finds 'resnet18').
 *  2. Technique tokens: a technique matches only when the query is a
 *     PREFIX of the technique token OR the token is a PREFIX of the query.
 *     This prevents cross-contamination (e.g. 'kd' must not match 'densenet').
 */
export function modelMatchesQuery(
  key: string,
  model: BaselineModel,
  query: string,
): boolean {
  // Empty / whitespace-only query → show everything
  const q = normalize(query);
  if (!q) return true;

  // 1. Match by model name or model key (substring)
  if (normalize(model.model_name).includes(q)) return true;
  if (normalize(key).includes(q)) return true;

  // 2. Match by compression technique (prefix match on both sides)
  return ALL_TECHNIQUES.some(
    (technique) => technique.startsWith(q) || q.startsWith(technique),
  );
}

interface ModelSelectorProps {
  models: Record<string, BaselineModel>;
  selectedModel: string | null;
  compressionCounts: Record<string, number>;
  onSelectModel: (modelKey: string) => void;
}

export function ModelSelector({
  models,
  selectedModel,
  compressionCounts,
  onSelectModel,
}: ModelSelectorProps) {
  const { query } = useSearch();

  const filteredModels = useMemo(() => {
    if (!query.trim()) return models;
    return Object.fromEntries(
      Object.entries(models).filter(([key, model]) => modelMatchesQuery(key, model, query))
    );
  }, [models, query]);

  const noResults = query.trim() !== '' && Object.keys(filteredModels).length === 0;

  return (
    <div>
      {/* Active search query chip */}
      {query.trim() && (
        <p className="mb-3 text-xs text-on-surface-variant">
          Showing{' '}
          <span className="font-semibold text-primary">{Object.keys(filteredModels).length}</span>
          {' '}of{' '}
          <span className="font-semibold">{Object.keys(models).length}</span>
          {' '}models for{' '}
          <span className="italic">&ldquo;{query}&rdquo;</span>
        </p>
      )}

      {noResults ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <span className="material-symbols-outlined text-4xl text-on-surface-variant/30 mb-3">search_off</span>
          <p className="text-on-surface-variant font-medium">No models match &ldquo;{query}&rdquo;</p>
          <p className="text-xs text-on-surface-variant/50 mt-1">
            Try searching by model name (e.g. &ldquo;resnet&rdquo;) or technique (e.g. &ldquo;quantization&rdquo;)
          </p>
        </div>
      ) : (
        <ModelGrid
          models={filteredModels}
          selectedModel={selectedModel}
          compressionCounts={compressionCounts}
          onSelectModel={onSelectModel}
        />
      )}
    </div>
  );
}
