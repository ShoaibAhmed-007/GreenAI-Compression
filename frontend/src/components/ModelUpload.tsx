'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { compressPreloaded, getCompressionStatus, fetchAPI, DynamicResult, CompressionStatus } from '@/lib/api';

interface ModelUploadProps {
  onResult: (result: DynamicResult) => void;
}

interface PreloadedModel {
  key: string;
  name: string;
  params: string;
  input_size: number;
  dataset: string;
}

const METHODS = [
  { value: 'pruning', label: 'Pruning (70%)', description: 'Remove 70% of smallest weights + fine-tune + gzip' },
  { value: 'quantization', label: 'Quantization (INT8)', description: 'Dynamic INT8 quantization for weights' },
  { value: 'hybrid', label: 'Hybrid (Prune + Quantize)', description: 'Prune 50% → fine-tune → quantize INT8' },
  { value: 'kd', label: 'Knowledge Distillation', description: 'Distill to compact MobileNet-style student' },
];

const DATASETS = [
  { value: 'CIFAR10', label: 'CIFAR-10 (10 classes)' },
  { value: 'CIFAR100', label: 'CIFAR-100 (100 classes)' },
];

// Step icons
const STEP_ICONS: Record<string, string> = {
  loading_model: '\u2B07',
  loading_data: '\uD83D\uDCCA',
  compressing: '\u2699\uFE0F',
  energy_tracking: '\u26A1',
  evaluating: '\uD83D\uDCCF',
  complete: '\u2705',
};

export function ModelUpload({ onResult }: ModelUploadProps) {
  const [models, setModels] = useState<PreloadedModel[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [method, setMethod] = useState('pruning');
  const [dataset, setDataset] = useState('CIFAR10');
  const [epochs, setEpochs] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [status, setStatus] = useState<CompressionStatus | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch available models on mount
  useEffect(() => {
    fetchAPI('/api/preloaded-models')
      .then((data) => {
        setModels(data.models);
        if (data.models.length > 0) {
          setSelectedModel(data.models[0].key);
        }
      })
      .catch(() => {
        const fallback: PreloadedModel[] = [
          { key: 'resnet18', name: 'ResNet18', params: '11.2M', input_size: 224, dataset: 'CIFAR-10 / ImageNet' },
          { key: 'resnet34', name: 'ResNet34', params: '21.8M', input_size: 224, dataset: 'CIFAR-10 / ImageNet' },
          { key: 'resnet50', name: 'ResNet50', params: '25.6M', input_size: 224, dataset: 'ImageNet' },
          { key: 'vgg16', name: 'VGG16', params: '138M', input_size: 224, dataset: 'CIFAR-10 / ImageNet' },
          { key: 'vgg19', name: 'VGG19', params: '143M', input_size: 224, dataset: 'CIFAR-10 / ImageNet' },
          { key: 'mobilenet_v2', name: 'MobileNetV2', params: '3.4M', input_size: 224, dataset: 'CIFAR-10 / ImageNet' },
          { key: 'efficientnet_b0', name: 'EfficientNet-B0', params: '5.3M', input_size: 224, dataset: 'ImageNet' },
          { key: 'efficientnet_b1', name: 'EfficientNet-B1', params: '7.8M', input_size: 240, dataset: 'ImageNet' },
          { key: 'densenet121', name: 'DenseNet121', params: '8.0M', input_size: 224, dataset: 'CIFAR-10 / ImageNet' },
          { key: 'densenet169', name: 'DenseNet169', params: '14.3M', input_size: 224, dataset: 'CIFAR-10 / ImageNet' },
          { key: 'squeezenet', name: 'SqueezeNet 1.1', params: '1.2M', input_size: 224, dataset: 'CIFAR-10 / ImageNet' },
          { key: 'shufflenet_v2', name: 'ShuffleNet V2', params: '2.3M', input_size: 224, dataset: 'CIFAR-10 / ImageNet' },
          { key: 'alexnet', name: 'AlexNet', params: '61M', input_size: 224, dataset: 'CIFAR-10 / ImageNet' },
          { key: 'inception_v3', name: 'Inception V3', params: '23.8M', input_size: 299, dataset: 'ImageNet' },
          { key: 'googlenet', name: 'GoogLeNet', params: '6.8M', input_size: 224, dataset: 'ImageNet' },
        ];
        setModels(fallback);
        setSelectedModel(fallback[0].key);
      })
      .finally(() => setModelsLoading(false));
  }, []);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getCompressionStatus();
        setStatus(s);

        // Task finished (success or error)
        if (!s.running) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;

          if (s.error) {
            setError(s.error);
            setLoading(false);
          } else if (s.result) {
            onResult(s.result);
            setLoading(false);
          } else {
            // No result and no error — stale/initial state, stop polling
            setLoading(false);
            setStatus(null);
          }
        }
      } catch {
        // Ignore transient network errors during polling
      }
    }, 1500);
  }, [onResult]);

  const handleSubmit = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    setLoading(true);
    setError(null);
    setStatus(null);

    try {
      await compressPreloaded(selectedModel, method, dataset, epochs);
      // Start polling for progress
      startPolling();
    } catch (err: any) {
      setError(err.message || 'Compression failed');
      setLoading(false);
    }
  };

  const selectedModelInfo = models.find(m => m.key === selectedModel);
  const selectedMethod = METHODS.find(m => m.value === method);

  // Determine which step index is active
  const steps = status?.steps || [];
  const currentStepIdx = steps.findIndex(s => s.key === status?.step);

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Model Compression
      </h3>
      <p className="text-sm text-gray-500 mb-6">
        Select a pretrained model and compression method to compare results.
      </p>

      {/* Progress Tracker — shown while compressing */}
      {loading && status && steps.length > 0 && (
        <div className="mb-6 bg-gray-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-700">Compression Progress</span>
            <span className="text-xs text-green-600 font-mono">
              {currentStepIdx >= 0 ? `${currentStepIdx + 1}/${steps.length}` : '...'}
            </span>
          </div>

          {/* Step indicators */}
          <div className="space-y-2">
            {steps.map((step, idx) => {
              const isDone = idx < currentStepIdx || status?.step === 'complete';
              const isActive = idx === currentStepIdx && status?.step !== 'complete';
              const isPending = idx > currentStepIdx;

              return (
                <div key={step.key} className="flex items-center gap-3">
                  {/* Circle indicator */}
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs flex-shrink-0 transition-all duration-300 ${
                    isDone ? 'bg-green-500 text-white' :
                    isActive ? 'bg-green-100 border-2 border-green-500 text-green-700' :
                    'bg-gray-200 text-gray-400'
                  }`}>
                    {isDone ? '\u2713' : STEP_ICONS[step.key] || (idx + 1)}
                  </div>

                  {/* Label + detail */}
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-medium truncate ${
                      isDone ? 'text-green-700' :
                      isActive ? 'text-gray-900' :
                      'text-gray-400'
                    }`}>
                      {step.label}
                    </p>
                    {isActive && status?.detail && (
                      <p className="text-xs text-green-600 truncate animate-pulse">
                        {status.detail}
                      </p>
                    )}
                  </div>

                  {/* Spinner for active step */}
                  {isActive && (
                    <div className="w-4 h-4 border-2 border-green-500 border-t-transparent rounded-full animate-spin flex-shrink-0" />
                  )}
                </div>
              );
            })}
          </div>

          {/* Progress bar */}
          <div className="mt-3 w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-green-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${steps.length > 0 ? Math.max(5, ((currentStepIdx + 1) / steps.length) * 100) : 5}%` }}
            />
          </div>
        </div>
      )}

      {/* Model Selector */}
      <div className="mb-5">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Model
        </label>
        {modelsLoading ? (
          <div className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-400">
            Loading models...
          </div>
        ) : (
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={loading}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-green-500 focus:border-green-500 disabled:opacity-50"
          >
            {models.map(m => (
              <option key={m.key} value={m.key}>
                {m.name} ({m.params} params)
              </option>
            ))}
          </select>
        )}
        {selectedModelInfo && !loading && (
          <div className="mt-2 bg-gray-50 rounded-lg px-3 py-2 text-xs text-gray-600 flex gap-4">
            <span>&#128202; {selectedModelInfo.params} params</span>
            <span>&#128393; {selectedModelInfo.input_size}x{selectedModelInfo.input_size} input</span>
            <span>&#128218; {selectedModelInfo.dataset}</span>
          </div>
        )}
      </div>

      {/* Compression Method Selector */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Compression Method
        </label>
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value)}
          disabled={loading}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-green-500 focus:border-green-500 disabled:opacity-50"
        >
          {METHODS.map(m => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
        {selectedMethod && !loading && (
          <p className="text-xs text-gray-500 mt-1">{selectedMethod.description}</p>
        )}
      </div>

      {/* Dataset Selector */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Evaluation Dataset
        </label>
        <select
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
          disabled={loading}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-green-500 focus:border-green-500 disabled:opacity-50"
        >
          {DATASETS.map(d => (
            <option key={d.value} value={d.value}>{d.label}</option>
          ))}
        </select>
      </div>

      {/* Fine-tune Epochs */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Fine-tune Epochs: {epochs}
        </label>
        <input
          type="range"
          min={0}
          max={20}
          value={epochs}
          onChange={(e) => setEpochs(Number(e.target.value))}
          disabled={loading}
          className="w-full accent-green-600"
        />
        <div className="flex justify-between text-xs text-gray-400">
          <span>0</span>
          <span>10</span>
          <span>20</span>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        disabled={!selectedModel || loading}
        className={`w-full mt-5 ${
          loading || !selectedModel
            ? 'btn-secondary opacity-60 cursor-not-allowed'
            : 'btn-primary'
        }`}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Compressing...
          </span>
        ) : (
          'Compress Model'
        )}
      </button>
    </div>
  );
}
