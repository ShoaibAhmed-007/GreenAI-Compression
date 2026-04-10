'use client';

import { BaselineModel } from '@/lib/api';

// Model family color mapping
const MODEL_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  resnet18:       { bg: 'bg-blue-50',    text: 'text-blue-700',    border: 'border-blue-200' },
  resnet34:       { bg: 'bg-blue-50',    text: 'text-blue-700',    border: 'border-blue-200' },
  mobilenet_v2:   { bg: 'bg-green-50',   text: 'text-green-700',   border: 'border-green-200' },
  efficientnet_b0:{ bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
  efficientnet_b1:{ bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
  densenet121:    { bg: 'bg-amber-50',   text: 'text-amber-700',   border: 'border-amber-200' },
  densenet169:    { bg: 'bg-amber-50',   text: 'text-amber-700',   border: 'border-amber-200' },
  squeezenet:     { bg: 'bg-rose-50',    text: 'text-rose-700',    border: 'border-rose-200' },
  shufflenet_v2:  { bg: 'bg-cyan-50',    text: 'text-cyan-700',    border: 'border-cyan-200' },
  inception_v3:   { bg: 'bg-indigo-50',  text: 'text-indigo-700',  border: 'border-indigo-200' },
  googlenet:      { bg: 'bg-teal-50',    text: 'text-teal-700',    border: 'border-teal-200' },
};

const MODEL_ICONS: Record<string, string> = {
  resnet18: 'R18', resnet34: 'R34',
  mobilenet_v2: 'Mv2',
  efficientnet_b0: 'Eb0', efficientnet_b1: 'Eb1',
  densenet121: 'D121', densenet169: 'D169',
  squeezenet: 'Sq', shufflenet_v2: 'Sh',
  inception_v3: 'Iv3', googlenet: 'Gn',
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
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900">
          Model Library
          <span className="ml-2 text-sm font-normal text-gray-400">
            {modelKeys.length} models
          </span>
        </h2>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-400" /> Ready
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-gray-300" /> Not Prepared
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
        {modelKeys.map((key) => {
          const m = models[key];
          const isReady = m.status === 'ready';
          const isSelected = selectedModel === key;
          const colors = MODEL_COLORS[key] || { bg: 'bg-gray-50', text: 'text-gray-700', border: 'border-gray-200' };
          const icon = MODEL_ICONS[key] || key.substring(0, 3).toUpperCase();
          const resultCount = compressionCounts[key] || 0;

          return (
            <button
              key={key}
              onClick={() => onSelectModel(key)}
              className={`relative text-left p-4 rounded-xl border-2 transition-all duration-200 hover:shadow-md ${
                isSelected
                  ? `${colors.border} ${colors.bg} shadow-md ring-2 ring-offset-1 ring-green-400`
                  : isReady
                    ? `border-gray-100 bg-white hover:${colors.border} hover:${colors.bg}`
                    : 'border-gray-100 bg-gray-50/50 hover:border-gray-200'
              }`}
            >
              {/* Status dot */}
              <div className={`absolute top-2 right-2 w-2.5 h-2.5 rounded-full ${
                isReady ? 'bg-green-400' : m.status === 'error' ? 'bg-red-400' : 'bg-gray-300'
              }`} />

              {/* Icon */}
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center mb-2 ${
                isReady ? colors.bg : 'bg-gray-100'
              }`}>
                <span className={`text-xs font-bold ${isReady ? colors.text : 'text-gray-400'}`}>
                  {icon}
                </span>
              </div>

              {/* Name */}
              <p className={`text-sm font-semibold truncate ${
                isReady ? 'text-gray-900' : 'text-gray-500'
              }`}>
                {m.model_name}
              </p>

              {/* Params */}
              <p className="text-xs text-gray-400 mt-0.5">{m.params_label} params</p>

              {/* Metrics (if ready) */}
              {isReady && (
                <>
                  <div className="mt-2 pt-2 border-t border-gray-100 grid grid-cols-2 gap-1">
                    <div>
                      <p className="text-[10px] text-gray-400">Accuracy</p>
                      <p className="text-xs font-bold text-green-700">
                        {m.accuracy != null ? `${m.accuracy}%` : 'Not Available'}
                      </p>
                    </div>
                    <div>
                      <p className="text-[10px] text-gray-400">Size</p>
                      <p className="text-xs font-bold text-gray-700">
                        {m.size_MB != null ? `${m.size_MB} MB` : 'Not Available'}
                      </p>
                    </div>
                  </div>
                  <div className="mt-2 pt-2 border-t border-gray-100">
                    <p className="text-[10px] text-gray-400">Training CO2</p>
                    <p className="text-xs font-bold text-emerald-700">
                      {formatCo2(m.training_co2_kg)}
                    </p>
                  </div>
                </>
              )}

              {/* Not ready prompt */}
              {!isReady && m.status !== 'error' && (
                <p className="text-[10px] text-gray-400 mt-2 italic">
                  Click to prepare
                </p>
              )}

              {/* Error */}
              {m.status === 'error' && (
                <p className="text-[10px] text-red-500 mt-2 truncate" title={m.error}>
                  Error: {m.error}
                </p>
              )}

              {/* Compression result count badge */}
              {resultCount > 0 && (
                <div className="absolute top-2 left-2">
                  <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-green-100 text-green-700 text-[10px] font-bold">
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
