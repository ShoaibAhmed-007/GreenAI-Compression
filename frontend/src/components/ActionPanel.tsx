'use client';

interface ActionPanelProps {
  taskStatus: Record<string, { running: boolean; last_run: string | null; error: string | null }>;
  onAction: (endpoint: string) => void;
  onRefresh: () => void;
}

export function ActionPanel({ taskStatus, onAction, onRefresh }: ActionPanelProps) {
  const actions = [
    {
      label: 'Run Compression',
      endpoint: '/api/compress',
      key: 'compress',
      description: 'Run all 5 compression strategies on baseline model',
      icon: '🗜️',
    },
    {
      label: 'Run Evaluation',
      endpoint: '/api/evaluate',
      key: 'evaluate',
      description: 'Evaluate all models (accuracy, FLOPs, latency)',
      icon: '📊',
    },
    {
      label: 'Track Energy',
      endpoint: '/api/energy/track',
      key: 'energy',
      description: 'Measure energy consumption & CO2 emissions',
      icon: '⚡',
    },
  ];

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Actions</h3>
        <button onClick={onRefresh} className="btn-secondary text-xs">
          ↻ Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {actions.map((action) => {
          const status = taskStatus[action.key];
          const isRunning = status?.running;
          const hasError = status?.error;
          const lastRun = status?.last_run;

          return (
            <div key={action.key} className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-start justify-between mb-2">
                <span className="text-xl">{action.icon}</span>
                {isRunning && (
                  <span className="badge bg-yellow-100 text-yellow-700 animate-pulse">
                    Running...
                  </span>
                )}
                {hasError && !isRunning && (
                  <span className="badge bg-red-100 text-red-700" title={hasError}>
                    Error
                  </span>
                )}
                {lastRun && !isRunning && !hasError && (
                  <span className="badge-green">Done</span>
                )}
              </div>
              <p className="text-sm font-medium text-gray-900">{action.label}</p>
              <p className="text-xs text-gray-500 mt-1 mb-3">{action.description}</p>
              {lastRun && (
                <p className="text-xs text-gray-400 mb-2">
                  Last: {lastRun}
                </p>
              )}
              <button
                onClick={() => onAction(action.endpoint)}
                disabled={isRunning}
                className={`w-full ${isRunning ? 'btn-secondary opacity-50 cursor-not-allowed' : 'btn-primary'}`}
              >
                {isRunning ? 'Running...' : action.label}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
