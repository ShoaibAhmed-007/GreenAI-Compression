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
    <section
      className={`asymmetric-hero bg-gradient-to-r from-surface-container to-surface-container-high p-8 ${
        allReady ? 'ring-1 ring-primary/20' : ''
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
            <span className="material-symbols-outlined text-primary">
              {allReady ? 'check_circle' : 'settings_b_roll'}
            </span>
          </div>
          <div>
            <h3 className="text-xl font-headline font-semibold text-on-surface">
              {allReady ? 'All Models Ready' : 'Prepare Models'}
            </h3>
            <p className="text-sm text-on-surface-variant">
              {readyCount}/{totalCount} models have baseline metrics
            </p>
          </div>
        </div>

        {!allReady && !isRunning && (
          <button
            onClick={handlePrepareAll}
            disabled={isStarting}
            className={`btn-primary ${isStarting ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isStarting ? 'Starting...' : 'Prepare All'}
          </button>
        )}
      </div>

      {/* Progress bar during preparation */}
      {isRunning && status && (
        <div className="mt-6 space-y-2">
          <div className="flex justify-between items-end mb-1">
            <span className="text-xs font-technical text-primary uppercase tracking-wider">
              {status.current_model && `Preparing ${status.current_model}...`}
            </span>
            <span className="text-xs font-technical text-on-surface">
              {status.completed}/{status.total} ({percent}%)
            </span>
          </div>
          <div className="w-full bg-surface-container-lowest h-3 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-primary/40 to-primary h-full relative transition-all duration-500"
              style={{ width: `${percent}%` }}
            >
              <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_25%,rgba(255,255,255,0.1)_50%,transparent_75%)] bg-[length:20px_20px]" />
            </div>
          </div>
          <p className="text-[10px] text-on-surface-variant">{status.progress}</p>
        </div>
      )}

      {error && (
        <div className="mt-4 flex items-start gap-3 bg-error-container/10 p-3 rounded-xl ghost-border">
          <span className="material-symbols-outlined text-error text-lg">warning</span>
          <p className="text-xs text-on-error-container leading-relaxed">{error}</p>
        </div>
      )}
    </section>
  );
}
