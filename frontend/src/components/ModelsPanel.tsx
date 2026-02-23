'use client';

interface ModelsPanelProps {
  models: { filename: string; size_MB: number }[];
}

export function ModelsPanel({ models }: ModelsPanelProps) {
  const totalSize = models.reduce((sum, m) => sum + m.size_MB, 0);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Saved Models</h3>
        <span className="text-xs text-gray-400">
          {models.length} files · {totalSize.toFixed(1)} MB total
        </span>
      </div>

      {models.length === 0 ? (
        <p className="text-sm text-gray-500">No models found.</p>
      ) : (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {models.map((model) => {
            const isCompressed = model.filename.endsWith('.gz');
            const isQuantized = model.filename.includes('quantized');
            const isStudent = model.filename.includes('student') || model.filename.includes('hybrid');
            const isUltra = model.filename.includes('ultra');

            let badgeClass = 'badge bg-gray-100 text-gray-600';
            if (isUltra) badgeClass = 'badge-green';
            else if (isCompressed) badgeClass = 'badge bg-purple-100 text-purple-700';
            else if (isQuantized) badgeClass = 'badge-blue';
            else if (isStudent) badgeClass = 'badge bg-amber-100 text-amber-700';

            return (
              <div
                key={model.filename}
                className="flex items-center justify-between p-2.5 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-gray-400 text-xs">📦</span>
                  <span className="text-sm font-mono text-gray-700 truncate">
                    {model.filename}
                  </span>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {isCompressed && (
                    <span className={badgeClass}>gz</span>
                  )}
                  {isQuantized && (
                    <span className={badgeClass}>INT8</span>
                  )}
                  <span className="text-sm font-mono text-gray-600 w-20 text-right">
                    {model.size_MB} MB
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
