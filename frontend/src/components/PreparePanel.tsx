'use client';

import { useState, useEffect, useRef } from 'react';
import { PrepareStatus, triggerPrepare, getPrepareStatus } from '@/lib/api';

interface PreparePanelProps {
  readyCount: number;
  totalCount: number;
  onComplete: () => void;
}

export function PreparePanel({ readyCount, totalCount, onComplete }: PreparePanelProps) {
  const [status, setStatus] = useState<PrepareStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const allReady = readyCount === totalCount;

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getPrepareStatus();
        setStatus(s);
        if (!s.running) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          if (s.error) {
            setError(s.error);
          } else {
            onComplete();
          }
        }
      } catch { /* ignore */ }
    }, 2000);
  };

  const handlePrepareAll = async () => {
    setIsStarting(true);
    setError(null);
    try {
      await triggerPrepare();
      startPolling();
    } catch (err: any) {
      setError(err.message || 'Failed to start preparation');
    } finally {
      setIsStarting(false);
    }
  };

  const isRunning = status?.running === true;
  const percent = status && status.total > 0
    ? Math.round((status.completed / status.total) * 100)
    : 0;

  return (
    <div className={`card ${allReady ? 'border-green-200 bg-green-50/30' : ''}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
            allReady ? 'bg-green-100' : 'bg-amber-100'
          }`}>
            <span className="text-lg">{allReady ? '✅' : '📥'}</span>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-gray-900">
              {allReady ? 'All Models Ready' : 'Prepare Models'}
            </h3>
            <p className="text-xs text-gray-500">
              {readyCount}/{totalCount} models have baseline metrics
            </p>
          </div>
        </div>

        {!allReady && !isRunning && (
          <button
            onClick={handlePrepareAll}
            disabled={isStarting}
            className={`btn-primary text-sm ${isStarting ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isStarting ? 'Starting...' : 'Prepare All'}
          </button>
        )}
      </div>

      {/* Progress bar during preparation */}
      {isRunning && status && (
        <div className="mt-4">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-600">
              {status.current_model && `Preparing ${status.current_model}...`}
            </span>
            <span className="text-xs font-mono text-green-600">
              {status.completed}/{status.total} ({percent}%)
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-green-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${percent}%` }}
            />
          </div>
          <p className="text-[10px] text-gray-400 mt-1">{status.progress}</p>
        </div>
      )}

      {error && (
        <div className="mt-3 bg-red-50 border border-red-200 rounded p-2">
          <p className="text-xs text-red-700">{error}</p>
        </div>
      )}
    </div>
  );
}
