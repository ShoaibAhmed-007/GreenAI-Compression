'use client';

import { useMemo, useState } from 'react';
import { CompareImageResponse, compareImage } from '@/lib/api';

const LAST_COMPARE_KEY = 'greenai_compare_image_last_result';
const COMPARE_HISTORY_KEY = 'greenai_compare_image_history';
const MAX_HISTORY = 20;

export interface CompareImageHistoryItem {
  id: string;
  createdAt: string;
  result: CompareImageResponse;
}

function safelyReadStorage<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function safelyWriteStorage<T>(key: string, value: T) {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore storage quota or parsing failures.
  }
}

export function loadLastCompareResult(): CompareImageResponse | null {
  return safelyReadStorage<CompareImageResponse | null>(LAST_COMPARE_KEY, null);
}

export function loadCompareHistory(): CompareImageHistoryItem[] {
  return safelyReadStorage<CompareImageHistoryItem[]>(COMPARE_HISTORY_KEY, []);
}

function appendHistory(result: CompareImageResponse): CompareImageHistoryItem[] {
  const existing = loadCompareHistory();
  const next: CompareImageHistoryItem = {
    id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    createdAt: new Date().toISOString(),
    result,
  };
  const updated = [next, ...existing].slice(0, MAX_HISTORY);
  safelyWriteStorage(COMPARE_HISTORY_KEY, updated);
  return updated;
}

export default function useCompareImage() {
  const [result, setResult] = useState<CompareImageResponse | null>(() => loadLastCompareResult());
  const [history, setHistory] = useState<CompareImageHistoryItem[]>(() => loadCompareHistory());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasResult = useMemo(() => Boolean(result), [result]);

  const runCompare = async (payload: {
    baseline_model_key: string;
    compressed_model_key: string;
    sample_image_path?: string;
    image_file?: File;
    enable_tta?: boolean;
  }) => {
    try {
      setLoading(true);
      setError(null);
      const response = await compareImage(payload);
      setResult(response);
      safelyWriteStorage(LAST_COMPARE_KEY, response);
      setHistory(appendHistory(response));
      return response;
    } catch (err: any) {
      setError(err?.message || 'Image comparison failed.');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const useHistoryResult = (item: CompareImageHistoryItem) => {
    setResult(item.result);
    safelyWriteStorage(LAST_COMPARE_KEY, item.result);
  };

  const clearHistory = () => {
    setHistory([]);
    safelyWriteStorage(COMPARE_HISTORY_KEY, []);
  };

  return {
    result,
    history,
    loading,
    error,
    hasResult,
    runCompare,
    setResult,
    setError,
    useHistoryResult,
    clearHistory,
  };
}
