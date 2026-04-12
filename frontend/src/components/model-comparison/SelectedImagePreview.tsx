'use client';

interface SelectedImagePreviewProps {
  imageUrl: string | null;
  sourceLabel: string;
  sourcePath?: string | null;
}

export default function SelectedImagePreview({
  imageUrl,
  sourceLabel,
  sourcePath,
}: SelectedImagePreviewProps) {
  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
      <p className="text-xs uppercase tracking-wide text-gray-500">Selected Image Preview</p>
      {imageUrl ? (
        <>
          <div className="cifar-zoom-wrap w-56 h-56 mx-auto mt-2 bg-white border border-gray-200 rounded-md">
            <img src={imageUrl} alt={sourceLabel} className="cifar-smooth w-56 h-56" />
          </div>
          <p className="text-sm text-gray-700 mt-2">Source: <span className="font-semibold">{sourceLabel}</span></p>
          {sourcePath && (
            <p className="text-[11px] text-gray-500 mt-1 break-all">{sourcePath}</p>
          )}
        </>
      ) : (
        <p className="text-sm text-gray-500 mt-2">Select a sample image or upload one to preview.</p>
      )}
    </div>
  );
}
