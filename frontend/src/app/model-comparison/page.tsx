'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import {
  ModelComparisonOptionsResponse,
  ModelComparisonSample,
  getModelComparisonOptions,
  getModelComparisonSamples,
} from '@/lib/api';
import ImageSourceTabs from '@/components/model-comparison/ImageSourceTabs';
import SampleImageSelector from '@/components/model-comparison/SampleImageSelector';
import ImageUploadDropzone from '@/components/model-comparison/ImageUploadDropzone';
import SelectedImagePreview from '@/components/model-comparison/SelectedImagePreview';
import PredictionComparisonResults from '@/components/model-comparison/PredictionComparisonResults';
import useCompareImage from '@/lib/hooks/useCompareImage';

type SourceMode = 'sample' | 'upload';

export default function ModelComparisonPage() {
  const [samples, setSamples] = useState<ModelComparisonSample[]>([]);
  const [options, setOptions] = useState<ModelComparisonOptionsResponse>({
    baseline_models: [],
    compressed_models: [],
  });
  const [initialLoading, setInitialLoading] = useState(true);
  const [initialError, setInitialError] = useState<string | null>(null);

  const [sourceMode, setSourceMode] = useState<SourceMode>('sample');
  const [selectedSamplePath, setSelectedSamplePath] = useState('');
  const [selectedSample, setSelectedSample] = useState<ModelComparisonSample | null>(null);

  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = useState<string | null>(null);

  const [baselineModelKey, setBaselineModelKey] = useState('');
  const [compressedModelKey, setCompressedModelKey] = useState('');
  const [enableTTA, setEnableTTA] = useState(true);
  const [formError, setFormError] = useState<string | null>(null);

  const {
    result,
    history,
    loading,
    error,
    runCompare,
    useHistoryResult,
    clearHistory,
  } = useCompareImage();

  useEffect(() => {
    let cancelled = false;

    const loadData = async () => {
      try {
        setInitialLoading(true);
        const [sampleData, optionData] = await Promise.all([
          getModelComparisonSamples(20),
          getModelComparisonOptions(),
        ]);

        if (cancelled) return;

        setSamples(sampleData);
        setOptions(optionData);

        if (sampleData.length > 0) {
          setSelectedSample(sampleData[0]);
          setSelectedSamplePath(sampleData[0].source_path || '');
        }

        if (optionData.baseline_models.length > 0) {
          const readyBaseline = optionData.baseline_models.find((item) => item.status === 'ready');
          setBaselineModelKey((readyBaseline || optionData.baseline_models[0]).key);
        }

        if (optionData.compressed_models.length > 0) {
          setCompressedModelKey(optionData.compressed_models[0].key);
        }

        setInitialError(null);
      } catch (err: any) {
        if (!cancelled) {
          setInitialError(err?.message || 'Failed to load model comparison data.');
        }
      } finally {
        if (!cancelled) {
          setInitialLoading(false);
        }
      }
    };

    loadData();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!uploadedFile) {
      setUploadedPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(uploadedFile);
    setUploadedPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [uploadedFile]);

  const readyBaselineOptions = useMemo(() => {
    const ready = options.baseline_models.filter((item) => item.status === 'ready');
    return ready.length > 0 ? ready : options.baseline_models;
  }, [options.baseline_models]);

  const noReadyBaseline = readyBaselineOptions.length === 0;

  const selectedPreviewUrl = sourceMode === 'sample'
    ? (selectedSample?.image_data_url || null)
    : uploadedPreviewUrl;

  const selectedSourceLabel = sourceMode === 'sample'
    ? 'Sample Image'
    : (uploadedFile?.name ? 'Uploaded Image' : 'Upload Pending');

  const selectedSourcePath = sourceMode === 'sample'
    ? selectedSamplePath
    : uploadedFile?.name || null;

  const onSelectSample = (sample: ModelComparisonSample) => {
    setSelectedSample(sample);
    setSelectedSamplePath(sample.source_path || '');
    setSourceMode('sample');
    setFormError(null);
  };

  const onSelectUpload = (file: File | null) => {
    setUploadedFile(file);
    if (file) {
      setSourceMode('upload');
      setFormError(null);
    }
  };

  const runCompareAction = async () => {
    if (!baselineModelKey) {
      setFormError('Please select a baseline model.');
      return;
    }
    if (!compressedModelKey) {
      setFormError('Please select a compressed model.');
      return;
    }

    if (sourceMode === 'sample') {
      if (!selectedSamplePath) {
        setFormError('Please select a sample image.');
        return;
      }
      setFormError(null);
      try {
        await runCompare({
          baseline_model_key: baselineModelKey,
          compressed_model_key: compressedModelKey,
          sample_image_path: selectedSamplePath,
          enable_tta: enableTTA,
        });
      } catch {
        // Error state is handled in hook.
      }
      return;
    }

    if (!uploadedFile) {
      setFormError('Please upload an image first.');
      return;
    }

    setFormError(null);
    try {
      await runCompare({
        baseline_model_key: baselineModelKey,
        compressed_model_key: compressedModelKey,
        image_file: uploadedFile,
        enable_tta: enableTTA,
      });
    } catch {
      // Error state is handled in hook.
    }
  };

  if (initialLoading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center min-h-[320px]">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-green-200 border-t-green-600 rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm text-gray-600">Loading image comparison workspace...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Compare Models on Images</h2>
          <p className="text-sm text-gray-500 mt-1">
            Run baseline vs compressed inference on either project sample images or your own uploaded image.
          </p>
        </div>
        <Link href="/" className="btn-secondary">
          Back to Dashboard
        </Link>
      </div>

      {(initialError || formError || error) && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {initialError || formError || error}
        </div>
      )}

      <div className="card space-y-4">
        <h3 className="text-lg font-semibold text-gray-900">1. Select Models</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium text-gray-700">Baseline Model</label>
            <select
              value={baselineModelKey}
              onChange={(e) => setBaselineModelKey(e.target.value)}
              className="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg bg-white focus:ring-2 focus:ring-green-300 focus:border-green-400 outline-none"
            >
              {readyBaselineOptions.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.name}
                </option>
              ))}
            </select>
            {noReadyBaseline && (
              <p className="text-xs text-amber-700 mt-2">
                No ready baseline found. Prepare baseline models first.
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
          </div>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-3">
          <label className="inline-flex items-center gap-2 text-sm font-medium text-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={enableTTA}
              onChange={(e) => setEnableTTA(e.target.checked)}
              className="h-4 w-4 rounded border-gray-300 text-green-600 focus:ring-green-500"
            />
            Enable test-time augmentation (TTA)
          </label>
          <p className="text-xs text-gray-500 mt-1">
            Improves stability for confusing classes (for example, cat vs dog) by averaging predictions over augmented variants.
          </p>
        </div>
      </div>

      <div className="card space-y-4">
        <h3 className="text-lg font-semibold text-gray-900">2. Select Image Input</h3>
        <ImageSourceTabs value={sourceMode} onChange={setSourceMode} />

        {sourceMode === 'sample' ? (
          <SampleImageSelector
            samples={samples}
            selectedPath={selectedSamplePath}
            onSelect={onSelectSample}
          />
        ) : (
          <ImageUploadDropzone
            fileName={uploadedFile?.name}
            onFileSelected={onSelectUpload}
          />
        )}
      </div>

      <div className="card space-y-4">
        <h3 className="text-lg font-semibold text-gray-900">3. Preview and Run Inference</h3>

        <SelectedImagePreview
          imageUrl={selectedPreviewUrl}
          sourceLabel={selectedSourceLabel}
          sourcePath={selectedSourcePath}
        />

        <div className="flex items-center justify-end">
          <button
            type="button"
            onClick={runCompareAction}
            disabled={loading || noReadyBaseline || options.compressed_models.length === 0}
            className={`btn-primary ${loading ? 'opacity-70 cursor-not-allowed' : ''}`}
          >
            {loading ? 'Running Inference...' : 'Compare Image'}
          </button>
        </div>
      </div>

      {result && <PredictionComparisonResults result={result} />}

      {history.length > 0 && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-lg font-semibold text-gray-900">Comparison History</h3>
            <button type="button" className="btn-secondary" onClick={clearHistory}>
              Clear History
            </button>
          </div>

          <div className="space-y-2">
            {history.slice(0, 8).map((entry) => (
              <button
                key={entry.id}
                type="button"
                onClick={() => useHistoryResult(entry)}
                className="w-full text-left rounded-lg border border-gray-200 bg-white p-3 hover:border-green-300"
              >
                <p className="text-sm font-medium text-gray-900">
                  Baseline: <span className="capitalize">{entry.result.baseline.class}</span>{' '}
                  ({entry.result.baseline.confidence.toFixed(2)}%) | Compressed:{' '}
                  <span className="capitalize">{entry.result.compressed.class}</span>{' '}
                  ({entry.result.compressed.confidence.toFixed(2)}%)
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {new Date(entry.createdAt).toLocaleString()} | Source: {entry.result.input.source}
                </p>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
