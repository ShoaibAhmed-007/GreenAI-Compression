'use client';

import { useRef, useState } from 'react';

interface ImageUploadDropzoneProps {
  fileName?: string;
  onFileSelected: (file: File | null) => void;
}

function isValidImage(file: File): boolean {
  return file.type.startsWith('image/');
}

export default function ImageUploadDropzone({
  fileName,
  onFileSelected,
}: ImageUploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleFile = (file: File | null) => {
    if (!file) {
      setLocalError(null);
      onFileSelected(null);
      return;
    }

    if (!isValidImage(file)) {
      setLocalError('Invalid file type. Please select an image file.');
      onFileSelected(null);
      return;
    }

    setLocalError(null);
    onFileSelected(file);
  };

  return (
    <div className="space-y-2">
      <div
        onDragEnter={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragActive(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          const file = e.dataTransfer.files?.[0] || null;
          handleFile(file);
        }}
        className={`rounded-xl border-2 border-dashed p-8 text-center transition-all ${
          dragActive
            ? 'border-primary bg-primary/5'
            : 'border-outline-variant/30 bg-surface-container-low hover:border-primary/30'
        }`}
      >
        <span className="material-symbols-outlined text-3xl text-on-surface-variant/40 mb-2">cloud_upload</span>
        <p className="text-sm font-medium text-on-surface">Drag and drop image here</p>
        <p className="text-xs text-on-surface-variant/50 mt-1">or</p>
        <button
          type="button"
          className="mt-3 btn-secondary"
          onClick={() => inputRef.current?.click()}
        >
          Choose Image
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] || null)}
        />
        <p className="text-xs text-on-surface-variant/40 mt-3">
          Accepted formats: JPG, PNG, WEBP, BMP
        </p>
      </div>

      {fileName && (
        <p className="text-sm text-on-surface">
          Selected file: <span className="font-semibold font-technical text-primary">{fileName}</span>
        </p>
      )}

      {localError && (
        <div className="rounded-lg bg-error-container/10 px-3 py-2 text-sm text-on-error-container ghost-border">
          {localError}
        </div>
      )}
    </div>
  );
}
