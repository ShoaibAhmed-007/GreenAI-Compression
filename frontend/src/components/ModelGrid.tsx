'use client';

import { BaselineModel } from '@/lib/api';

// Model family color mapping
const MODEL_COLORS: Record<string, { icon: string; iconColor: string }> = {
  resnet18:       { icon: 'filter_vintage', iconColor: 'text-primary' },
  resnet34:       { icon: 'filter_vintage', iconColor: 'text-primary' },
  mobilenet_v2:   { icon: 'bolt', iconColor: 'text-secondary' },
  efficientnet_b0:{ icon: 'auto_awesome', iconColor: 'text-tertiary' },
  efficientnet_b1:{ icon: 'auto_awesome', iconColor: 'text-tertiary' },
  densenet121:    { icon: 'hub', iconColor: 'text-secondary' },
  densenet169:    { icon: 'hub', iconColor: 'text-secondary' },
  squeezenet:     { icon: 'compress', iconColor: 'text-error' },
  shufflenet_v2:  { icon: 'shuffle', iconColor: 'text-tertiary' },
  inception_v3:   { icon: 'diamond', iconColor: 'text-primary' },
  googlenet:      { icon: 'neurology', iconColor: 'text-secondary' },
};

interface ModelGridProps {
  models: Record<string, BaselineModel>;
  selectedModel: string | null;
  onSelectModel: (key: string) => void;
  compressionCounts: Record<string, number>;
}

function formatCo2(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 'Not Available';
  }
  return `${value.toFixed(6)} kg`;
}

export function ModelGrid({ models, selectedModel, onSelectModel, compressionCounts }: ModelGridProps) {
  const modelKeys = Object.keys(models);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-headline font-bold text-on-surface">
          Model Library
          <span className="ml-2 text-sm font-normal text-on-surface-variant">
            {modelKeys.length} models
          </span>
        </h2>
        <div className="flex items-center gap-2 text-xs text-on-surface-variant">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" /> Ready
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-on-surface-variant/30" /> Not Prepared
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {modelKeys.map((key) => {
          const m = models[key];
          const isReady = m.status === 'ready';
          const isSelected = selectedModel === key;
          const config = MODEL_COLORS[key] || { icon: 'smart_toy', iconColor: 'text-on-surface-variant' };
          const resultCount = compressionCounts[key] || 0;

          return (
            <button
              key={key}
              onClick={() => onSelectModel(key)}
              className={`relative text-left p-5 rounded-2xl transition-all duration-200 group flex flex-col h-full overflow-hidden ${
                isSelected
                  ? 'bg-surface-container-high ring-2 ring-primary/40 shadow-lg shadow-primary/5 scale-[1.01]'
                  : isReady
                    ? 'bg-surface-container-low hover:bg-surface-container ghost-border hover:ring-1 hover:ring-primary/20 hover:scale-[1.03] hover:shadow-xl hover:shadow-primary/10'
                    : 'bg-surface-container-low/50 ghost-border hover:bg-surface-container-low hover:scale-[1.02]'
              }`}
            >
              {/* Status dot */}
              <div className={`absolute top-3 right-3 w-2.5 h-2.5 rounded-full ${
                isReady ? 'bg-primary animate-pulse' : m.status === 'error' ? 'bg-error' : 'bg-on-surface-variant/30'
              }`} />

              {/* Icon */}
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center mb-3 ${
                isReady ? 'bg-primary/10' : 'bg-surface-container'
              }`}>
                <span className={`material-symbols-outlined ${isReady ? config.iconColor : 'text-on-surface-variant/40'}`}>
                  {config.icon}
                </span>
              </div>

              {/* Name */}
              <p className={`text-sm font-bold truncate ${
                isReady ? 'text-on-surface' : 'text-on-surface-variant/60'
              }`}>
                {m.model_name}
              </p>

              {/* Params */}
              <p className="text-xs text-on-surface-variant/60 mt-0.5 font-technical">{m.params_label} params</p>

              {/* Metrics (if ready) */}
              {isReady && (
                <>
                  <div className="mt-3 flex flex-col gap-2 w-full">
                    {/* Accuracy stat box */}
                    <div className="bg-surface-container-lowest px-4 py-2 rounded-md w-full flex justify-between items-center gap-3">
                      <p className="text-[10px] text-on-surface-variant uppercase font-technical leading-none flex-shrink-0">
                        Accuracy
                      </p>
                      <p className="text-xs font-technical text-on-surface whitespace-nowrap font-semibold">
                        {m.accuracy != null ? `${m.accuracy}%` : 'N/A'}
                      </p>
                    </div>
                    {/* Size stat box */}
                    <div className="bg-surface-container-lowest px-4 py-2 rounded-md w-full flex justify-between items-center gap-3">
                      <p className="text-[10px] text-on-surface-variant uppercase font-technical leading-none flex-shrink-0">
                        Size
                      </p>
                      <p className="text-xs font-technical text-on-surface whitespace-nowrap font-semibold">
                        {m.size_MB != null ? `${m.size_MB} MB` : 'N/A'}
                      </p>
                    </div>
                  </div>
                  <div className="mt-auto w-full flex flex-col gap-2 pt-3" style={{ borderTop: '1px solid rgba(66, 73, 62, 0.1)' }}>
                    {/* CO2 row */}
                    <div className="flex items-center gap-1.5 w-full">
                      <span className="material-symbols-outlined text-xs text-secondary flex-shrink-0">cloud_queue</span>
                      <span className="text-[10px] font-technical text-secondary break-all leading-tight">
                        {formatCo2(m.training_co2_kg)}
                      </span>
                    </div>
                    {/* COMPRESS button — full width */}
                    <span className="w-full text-xs font-bold text-primary flex items-center justify-center gap-0.5 group-hover:gap-1.5 transition-all bg-primary/10 hover:bg-primary/20 px-2 py-1.5 rounded-lg">
                      COMPRESS
                      <span className="material-symbols-outlined text-sm">chevron_right</span>
                    </span>
                  </div>
                </>
              )}

              {/* Not ready prompt */}
              {!isReady && m.status !== 'error' && (
                <p className="text-[10px] text-on-surface-variant/40 mt-2 italic">
                  Click to prepare
                </p>
              )}

              {/* Error */}
              {m.status === 'error' && (
                <p className="text-[10px] text-error mt-2 truncate" title={m.error}>
                  Error: {m.error}
                </p>
              )}

              {/* Compression result count badge */}
              {resultCount > 0 && (
                <div className="absolute top-3 left-3">
                  <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-primary/20 text-primary text-[10px] font-bold font-technical">
                    {resultCount}
                  </span>
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
