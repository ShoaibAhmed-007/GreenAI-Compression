'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import {
  ModelComparisonOptionsResponse,
  ModelComparisonResult,
  ModelComparisonSample,
  compareModelsOnImage,
  getModelComparisonOptions,
  getModelComparisonSamples,
} from '@/lib/api';

function formatPercent(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 'Not Available';
  }
  return `${value.toFixed(2)}%`;
}

function formatSize(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 'Not Available';
  }
  return `${value.toFixed(2)} MB`;
}

function formatCo2(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 'Not Available';
  }
  if (value > 0 && value < 0.000001) {
    return '<0.000001 kg';
  }
  return `${value.toFixed(6)} kg`;
}

function toConfidence(value?: number | null): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, value));
}

function confidenceBarTone(value?: number | null): string {
  const confidence = toConfidence(value);
  if (confidence < 30) return 'bg-red-500';
  if (confidence < 50) return 'bg-amber-500';
  return 'bg-green-500';
}

function confidenceTextTone(value?: number | null): string {
  const confidence = toConfidence(value);
  if (confidence < 30) return 'text-red-700';
  if (confidence < 50) return 'text-amber-700';
  return 'text-green-700';
}

function confidenceHint(value?: number | null): string {
  const confidence = toConfidence(value);
  if (confidence < 30) return 'Very low confidence';
  if (confidence < 50) return 'Low confidence';
  return 'Reliable confidence';
}

function ConfidenceBar({ value }: { value?: number | null }) {
  const confidence = toConfidence(value);
  return (
    <div className="mt-2">
      <div className="w-full h-2 rounded-full bg-white/70 border border-gray-200 overflow-hidden">
        <div
          className={`h-full ${confidenceBarTone(confidence)} transition-all duration-300`}
          style={{ width: `${Math.max(2, confidence)}%` }}
        />
      </div>
      <p className={`text-[11px] mt-1 ${confidenceTextTone(confidence)}`}>
        {confidenceHint(confidence)}
      </p>
    </div>
  );
}

function confidenceTone(delta?: number | null): string {
  if (typeof delta !== 'number' || !Number.isFinite(delta)) return 'text-gray-700';
  if (delta > 0) return 'text-green-700';
  if (delta < 0) return 'text-amber-700';
  return 'text-gray-700';
}

function reductionTone(delta?: number | null): string {
  if (typeof delta !== 'number' || !Number.isFinite(delta)) return 'text-gray-700';
  if (delta > 0) return 'text-green-700';
  if (delta < 0) return 'text-red-700';
  return 'text-gray-700';
}

export default function ModelComparisonPage() {
  const [samples, setSamples] = useState<ModelComparisonSample[]>([]);
  const [options, setOptions] = useState<ModelComparisonOptionsResponse>({
    baseline_models: [],
    compressed_models: [],
  });
  const [selectedSampleId, setSelectedSampleId] = useState<number | null>(null);
  const [baselineModelKey, setBaselineModelKey] = useState('');
  const [compressedModelKey, setCompressedModelKey] = useState('');
  const [result, setResult] = useState<ModelComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const loadInitialData = async () => {
      try {
        setInitialLoading(true);
        const [sampleData, optionData] = await Promise.all([
          getModelComparisonSamples(10),
          getModelComparisonOptions(),
        ]);

        if (cancelled) return;

        setSamples(sampleData);
        setOptions(optionData);

        if (sampleData.length > 0) {
          setSelectedSampleId(sampleData[0].id);
        }

        if (optionData.baseline_models.length > 0) {
          setBaselineModelKey(optionData.baseline_models[0].key);
        }

        if (optionData.compressed_models.length > 0) {
          setCompressedModelKey(optionData.compressed_models[0].key);
        }

        setError(null);
      } catch (err: any) {
        if (cancelled) return;
        setError(err?.message || 'Failed to load model comparison data.');
      } finally {
        if (!cancelled) setInitialLoading(false);
      }
    };

    loadInitialData();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedSample = useMemo(
    () => samples.find((item) => item.id === selectedSampleId) || null,
    [samples, selectedSampleId]
  );

  const selectedBaseline = useMemo(
    () => options.baseline_models.find((item) => item.key === baselineModelKey) || null,
    [options.baseline_models, baselineModelKey]
  );

  const selectedCompressed = useMemo(
    () => options.compressed_models.find((item) => item.key === compressedModelKey) || null,
    [options.compressed_models, compressedModelKey]
  );

  const runComparison = async () => {
    if (selectedSampleId == null) {
      setError('Please select an image before running comparison.');
      return;
    }
    if (!baselineModelKey) {
      setError('Please select a baseline model.');
      return;
    }
    if (!compressedModelKey) {
      setError('Please select a compressed model.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const response = await compareModelsOnImage({
        sample_id: selectedSampleId,
        baseline_model_key: baselineModelKey,
        compressed_model_key: compressedModelKey,
      });
      setResult(response);
    } catch (err: any) {
      setError(err?.message || 'Comparison failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (initialLoading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center min-h-[320px]">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-green-200 border-t-green-600 rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm text-gray-600">Loading model comparison workspace...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Image-Based Model Comparison</h2>
          <p className="text-sm text-gray-500 mt-1">
            Compare baseline and compressed models on the same image input.
          </p>
        </div>
        <Link href="/" className="btn-secondary">
          Back to Dashboard
        </Link>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="card space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">1. Image Selection</h3>
          <span className="text-xs text-gray-500">Select one image from the sample grid</span>
        </div>

        {samples.length === 0 ? (
          <p className="text-sm text-gray-500">No sample images available.</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {samples.map((sample) => {
              const selected = sample.id === selectedSampleId;
              return (
                <button
                  key={sample.id}
                  type="button"
                  onClick={() => setSelectedSampleId(sample.id)}
                  className={`rounded-lg border p-2 text-left transition-all ${
                    selected
                      ? 'border-green-500 ring-2 ring-green-200 bg-green-50'
                      : 'border-gray-200 hover:border-green-300 bg-white'
                  }`}
                >
                  <div className="cifar-zoom-wrap w-28 h-28 mx-auto bg-gray-50 border border-gray-100 rounded-md">
                    <img
                      src={sample.image_data_url}
                      alt={sample.label}
                      className="cifar-pixelated"
                    />
                  </div>
                  <p className="text-xs font-medium text-gray-700 mt-2 capitalize">{sample.label}</p>
                  <p className="text-[11px] text-gray-500">Sample #{sample.id}</p>
                </button>
              );
            })}
          </div>
        )}
      </div>

      <div className="card space-y-4">
        <h3 className="text-lg font-semibold text-gray-900">2. Model Selection</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium text-gray-700">Baseline Model</label>
            <select
              value={baselineModelKey}
              onChange={(e) => setBaselineModelKey(e.target.value)}
              className="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg bg-white focus:ring-2 focus:ring-green-300 focus:border-green-400 outline-none"
            >
              {options.baseline_models.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.name}
                </option>
              ))}
            </select>
            {selectedBaseline && (
              <p className="text-xs text-gray-500 mt-2">
                Size: {formatSize(selectedBaseline.size_MB)} | CO2: {formatCo2(selectedBaseline.co2_kg)}
              </p>
            )}
          </div>

          <div>
            <label className="text-sm font-medium text-gray-700">Compressed Model</label>
            <select
              value={compressedModelKey}
              onChange={(e) => setCompressedModelKey(e.target.value)}
              className="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg bg-white focus:ring-2 focus:ring-green-300 focus:border-green-400 outline-none"
            >
              {options.compressed_models.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.label}
                </option>
              ))}
            </select>
            {selectedCompressed && (
              <p className="text-xs text-gray-500 mt-2">
                Size: {formatSize(selectedCompressed.size_MB)} | CO2: {formatCo2(selectedCompressed.co2_kg)}
              </p>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
          <div className="text-xs text-gray-500">
            Both models will run on the exact same selected image.
          </div>
          <button
            type="button"
            onClick={runComparison}
            disabled={loading || samples.length === 0 || options.compressed_models.length === 0}
            className={`btn-primary ${loading ? 'opacity-70 cursor-not-allowed' : ''}`}
          >
            {loading ? 'Comparing Models...' : 'Compare Models'}
          </button>
        </div>
      </div>

      {result && (
        <>
          <div className="card space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">3. Output Display</h3>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
              <div className="rounded-lg border border-gray-200 p-3 bg-gray-50">
                <p className="text-xs uppercase tracking-wide text-gray-500 mb-2">Selected Image</p>
                <div className="cifar-zoom-wrap w-44 h-44 mx-auto bg-white border border-gray-200 rounded-md">
                  <img
                    src={result.sample.image_data_url}
                    alt={result.sample.true_label}
                    className="cifar-pixelated"
                  />
                </div>
                <p className="text-sm text-gray-700 mt-2 capitalize">
                  True Label: <span className="font-semibold">{result.sample.true_label}</span>
                </p>
                <p className="text-[11px] text-gray-500 mt-1">
                  CIFAR-10 source is 32x32; preview is bicubic-upscaled for clearer inspection.
                </p>
              </div>

              <div className="rounded-lg border border-blue-200 p-4 bg-blue-50">
                <p className="text-xs uppercase tracking-wide text-blue-700">Baseline Model</p>
                <h4 className="text-base font-semibold text-blue-900 mt-1">{result.baseline.model_name}</h4>
                <div className="mt-3 space-y-1 text-sm text-blue-900">
                  <p>
                    Predicted class: <span className="font-semibold capitalize">{result.baseline.prediction.predicted_class}</span>
                  </p>
                  <p>
                    Confidence: <span className="font-semibold">{formatPercent(result.baseline.prediction.confidence)}</span>
                  </p>
                  <ConfidenceBar value={result.baseline.prediction.confidence} />
                  {result.baseline.prediction.top_k && result.baseline.prediction.top_k.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs font-semibold text-blue-800">Top-3 probabilities</p>
                      <ul className="text-xs text-blue-900 mt-1 space-y-0.5">
                        {result.baseline.prediction.top_k.map((item, rank) => (
                          <li key={`baseline-top-${item.class_index}`} className="flex justify-between gap-2">
                            <span className="capitalize">#{rank + 1} {item.class_name}</span>
                            <span className="font-mono">{formatPercent(item.probability)}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {typeof result.baseline.prediction.temperature === 'number' && (
                    <p className="text-[11px] text-blue-700 mt-1">
                      Temperature scaling: {result.baseline.prediction.temperature.toFixed(2)}
                    </p>
                  )}
                  <p>Model size: <span className="font-semibold">{formatSize(result.baseline.size_MB)}</span></p>
                  <p>CO2 emission: <span className="font-semibold">{formatCo2(result.baseline.co2_kg)}</span></p>
                </div>
              </div>

              <div className="rounded-lg border border-green-200 p-4 bg-green-50">
                <p className="text-xs uppercase tracking-wide text-green-700">Compressed Model</p>
                <h4 className="text-base font-semibold text-green-900 mt-1">
                  {result.compressed.model_name} - {result.compressed.strategy_label}
                </h4>
                <div className="mt-3 space-y-1 text-sm text-green-900">
                  <p>
                    Predicted class: <span className="font-semibold capitalize">{result.compressed.prediction.predicted_class}</span>
                  </p>
                  <p>
                    Confidence: <span className="font-semibold">{formatPercent(result.compressed.prediction.confidence)}</span>
                  </p>
                  <ConfidenceBar value={result.compressed.prediction.confidence} />
                  {result.compressed.prediction.top_k && result.compressed.prediction.top_k.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs font-semibold text-green-800">Top-3 probabilities</p>
                      <ul className="text-xs text-green-900 mt-1 space-y-0.5">
                        {result.compressed.prediction.top_k.map((item, rank) => (
                          <li key={`compressed-top-${item.class_index}`} className="flex justify-between gap-2">
                            <span className="capitalize">#{rank + 1} {item.class_name}</span>
                            <span className="font-mono">{formatPercent(item.probability)}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {typeof result.compressed.prediction.temperature === 'number' && (
                    <p className="text-[11px] text-green-700 mt-1">
                      Temperature scaling: {result.compressed.prediction.temperature.toFixed(2)}
                    </p>
                  )}
                  <p>Model size: <span className="font-semibold">{formatSize(result.compressed.size_MB)}</span></p>
                  <p>CO2 emission: <span className="font-semibold">{formatCo2(result.compressed.co2_kg)}</span></p>
                </div>
              </div>
            </div>
          </div>

          <div className="card space-y-3">
            <h3 className="text-lg font-semibold text-gray-900">4. Performance Comparison</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="rounded-lg border border-gray-200 p-3 bg-white">
                <p className="text-xs text-gray-500 uppercase tracking-wide">Confidence Change</p>
                <p className={`text-lg font-bold mt-1 ${confidenceTone(result.comparison.confidence_delta_percent)}`}>
                  {result.comparison.confidence_delta_percent >= 0 ? '+' : ''}
                  {formatPercent(result.comparison.confidence_delta_percent)}
                </p>
              </div>
              <div className="rounded-lg border border-gray-200 p-3 bg-white">
                <p className="text-xs text-gray-500 uppercase tracking-wide">Size Reduction</p>
                <p className={`text-lg font-bold mt-1 ${reductionTone(result.comparison.size_reduction_percent)}`}>
                  {formatPercent(result.comparison.size_reduction_percent)}
                </p>
              </div>
              <div className="rounded-lg border border-gray-200 p-3 bg-white">
                <p className="text-xs text-gray-500 uppercase tracking-wide">CO2 Reduction</p>
                <p className={`text-lg font-bold mt-1 ${reductionTone(result.comparison.co2_reduction_percent)}`}>
                  {formatPercent(result.comparison.co2_reduction_percent)}
                </p>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg border border-gray-200 p-3">
              <p className="text-sm text-gray-700 font-medium">{result.comparison.summary}.</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
