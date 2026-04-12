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
        className={[
          'rounded-lg border-2 border-dashed p-5 text-center transition-colors',
          dragActive
            ? 'border-green-500 bg-green-50'
            : 'border-gray-300 bg-white hover:border-green-400',
        ].join(' ')}
      >
        <p className="text-sm font-medium text-gray-800">Drag and drop image here</p>
        <p className="text-xs text-gray-500 mt-1">or</p>
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
        <p className="text-xs text-gray-500 mt-2">
          Accepted formats: JPG, PNG, WEBP, BMP
        </p>
      </div>

      {fileName && (
        <p className="text-sm text-gray-700">
          Selected file: <span className="font-semibold">{fileName}</span>
        </p>
      )}

      {localError && (
        <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {localError}
        </div>
      )}
    </div>
  );
}
