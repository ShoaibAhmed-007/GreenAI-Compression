'use client';

import { BaselineModel } from '@/lib/api';
import { ModelGrid } from './ModelGrid';

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
  return (
    <ModelGrid
      models={models}
      selectedModel={selectedModel}
      compressionCounts={compressionCounts}
      onSelectModel={onSelectModel}
    />
  );
}
