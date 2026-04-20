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
import { DefenseStrategyPanel } from '@/components/DefenseStrategyPanel';
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
      <div className="flex items-center justify-center min-h-[320px]">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-on-surface-variant">Loading image comparison workspace...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <header className="mb-2">
        <Link
          href="/"
          className="inline-flex items-center text-primary hover:text-primary-fixed transition-colors gap-2 mb-4 group"
        >
          <span className="material-symbols-outlined text-[18px] group-hover:-translate-x-1 transition-transform">arrow_back</span>
          <span className="text-sm font-medium">Back to Dashboard</span>
        </Link>
        <h1 className="text-4xl font-headline font-bold text-on-surface tracking-tight mb-2">
          Compare Models on Images
        </h1>
        <p className="text-on-surface-variant font-light max-w-2xl">
          Execute side-by-side inference testing to validate performance and accuracy of your compressed edge models against high-precision baselines.
        </p>
      </header>

      {/* <DefenseStrategyPanel compact /> */}

      {(initialError || formError || error) && (
        <div className="bg-error-container/10 rounded-xl p-3 text-sm text-on-error-container ghost-border">
          {initialError || formError || error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Workflow Section (Left) */}
        <div className="lg:col-span-8 space-y-8">
          {/* Step 1: Select Models */}
          <section className="bg-surface-container rounded-xl p-8">
            <div className="flex items-center gap-4 mb-6">
              <span className="font-technical w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center font-bold text-sm">01</span>
              <h2 className="text-xl font-headline font-semibold text-on-surface">Select Models</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-xs font-technical text-on-surface-variant uppercase tracking-widest">Baseline Model</label>
                <select
                  value={baselineModelKey}
                  onChange={(e) => setBaselineModelKey(e.target.value)}
                  className="w-full bg-surface-container-low border-none rounded-lg text-on-surface py-3 px-4 focus:ring-2 focus:ring-primary/30 outline-none"
                >
                  {readyBaselineOptions.map((option) => (
                    <option key={option.key} value={option.key}>
                      {option.name}
                    </option>
                  ))}
                </select>
                {noReadyBaseline && (
                  <p className="text-xs text-secondary mt-2">
                    No ready baseline found. Prepare baseline models first.
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <label className="text-xs font-technical text-on-surface-variant uppercase tracking-widest">Compressed Model</label>
                <select
                  value={compressedModelKey}
                  onChange={(e) => setCompressedModelKey(e.target.value)}
                  className="w-full bg-surface-container-low border-none rounded-lg text-on-surface py-3 px-4 focus:ring-2 focus:ring-primary/30 outline-none"
                >
                  {options.compressed_models.map((option) => (
                    <option key={option.key} value={option.key}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* TTA Toggle */}
            <div className="mt-6 flex items-center justify-between p-4 bg-surface-container-low rounded-lg">
              <div className="flex flex-col">
                <span className="text-sm font-medium text-on-surface">Test-Time Augmentation (TTA)</span>
                <span className="text-xs text-on-surface-variant">
                  Increases accuracy by running inference on multiple transformed versions of the image.
                </span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableTTA}
                  onChange={(e) => setEnableTTA(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-surface-variant peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary" />
              </label>
            </div>
          </section>

          {/* Step 2: Select Image Input */}
          <section className="bg-surface-container rounded-xl p-8">
            <div className="flex items-center gap-4 mb-6">
              <span className="font-technical w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center font-bold text-sm">02</span>
              <h2 className="text-xl font-headline font-semibold text-on-surface">Select Image Input</h2>
            </div>
            <ImageSourceTabs value={sourceMode} onChange={setSourceMode} />

            <div className="mt-6">
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
          </section>

          {/* Step 3: Preview and Run */}
          <section className="bg-surface-container rounded-xl p-8 relative overflow-hidden">
            <div className="absolute top-4 right-4">
              <div className="flex items-center gap-2 bg-primary/20 text-primary px-3 py-1 rounded-full text-xs font-technical">
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                READY
              </div>
            </div>
            <div className="flex items-center gap-4 mb-6">
              <span className="font-technical w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center font-bold text-sm">03</span>
              <h2 className="text-xl font-headline font-semibold text-on-surface">Preview and Run</h2>
            </div>

            <div className="mb-6">
              <SelectedImagePreview
                imageUrl={selectedPreviewUrl}
                sourceLabel={selectedSourceLabel}
                sourcePath={selectedSourcePath}
              />
            </div>

            <button
              type="button"
              onClick={runCompareAction}
              disabled={loading || noReadyBaseline || options.compressed_models.length === 0}
              className={`w-full py-4 bg-gradient-to-r from-primary to-primary-container text-on-primary font-bold text-lg rounded-xl flex items-center justify-center gap-3 hover:shadow-lg transition-all active:scale-95 ${
                loading ? 'opacity-70 cursor-not-allowed' : ''
              }`}
            >
              <span className="material-symbols-outlined">bolt</span>
              {loading ? 'Running Inference...' : 'Run Comparison'}
            </button>
          </section>

          {/* Results */}
          {result && <PredictionComparisonResults result={result} />}
        </div>

        {/* Sidebar: Comparison History */}
        <aside className="lg:col-span-4 space-y-6">
          <div className="bg-surface-container rounded-xl p-6 h-fit sticky top-24">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-headline font-bold text-on-surface">Recent Runs</h3>
              {history.length > 0 && (
                <button
                  type="button"
                  className="text-xs font-technical text-error hover:underline transition-all"
                  onClick={clearHistory}
                >
                  Clear History
                </button>
              )}
            </div>

            {history.length === 0 ? (
              <div className="text-center py-8">
                <span className="material-symbols-outlined text-3xl text-on-surface-variant/30">history</span>
                <p className="text-sm text-on-surface-variant/50 mt-2">No comparison runs yet</p>
              </div>
            ) : (
              <div className="space-y-3">
                {history.slice(0, 8).map((entry) => {
                  const isMatch = entry.result.comparison?.prediction_match;
                  return (
                    <button
                      key={entry.id}
                      type="button"
                      onClick={() => useHistoryResult(entry)}
                      className="w-full text-left group"
                    >
                      <div className="flex gap-3 p-3 rounded-lg hover:bg-surface-container-low transition-colors ghost-border">
                        <div className="flex flex-col justify-between py-1 min-w-0">
                          <span className="text-sm font-bold text-on-surface truncate">
                            <span className="capitalize">{entry.result.baseline.class}</span>
                          </span>
                          <span className={`text-[10px] font-technical ${isMatch ? 'text-primary' : 'text-error'}`}>
                            {isMatch ? 'MATCH' : 'MISMATCH'} · {entry.result.baseline.confidence.toFixed(1)}% Conf
                          </span>
                          <span className="text-[10px] font-technical text-on-surface-variant/50">
                            {new Date(entry.createdAt).toLocaleString()}
                          </span>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            <div className="mt-8 pt-6 text-center" style={{ borderTop: '1px solid rgba(66, 73, 62, 0.1)' }}>
              <p className="text-xs text-on-surface-variant/40 italic font-light">
                &ldquo;Edge-ready compression for a sustainable digital ecosystem.&rdquo;
              </p>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
