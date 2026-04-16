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
    <div className="relative w-full aspect-[16/9] rounded-xl overflow-hidden bg-surface-container-lowest">
      {imageUrl ? (
        <>
          <img src={imageUrl} alt={sourceLabel} className="w-full h-full object-cover" />
          <div className="absolute inset-0 flex items-end p-4 bg-gradient-to-t from-surface-container/80 to-transparent">
            <div>
              <p className="text-sm text-on-surface">
                Source: <span className="font-semibold">{sourceLabel}</span>
              </p>
              {sourcePath && (
                <p className="text-[10px] text-on-surface-variant/60 mt-0.5 break-all font-technical">{sourcePath}</p>
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <span className="material-symbols-outlined text-4xl text-on-surface-variant/30">image</span>
            <p className="text-sm text-on-surface-variant/50 mt-2">Select a sample image or upload one to preview.</p>
          </div>
        </div>
      )}
    </div>
  );
}
