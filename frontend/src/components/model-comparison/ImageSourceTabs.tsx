'use client';

type SourceMode = 'sample' | 'upload';

interface ImageSourceTabsProps {
  value: SourceMode;
  onChange: (value: SourceMode) => void;
}

export default function ImageSourceTabs({ value, onChange }: ImageSourceTabsProps) {
  const tabs: Array<{ key: SourceMode; label: string }> = [
    { key: 'sample', label: 'Sample Images' },
    { key: 'upload', label: 'Upload Image' },
  ];

  return (
    <div className="flex gap-1 bg-surface-container-low p-1 rounded-xl">
      {tabs.map((tab) => {
        const active = tab.key === value;
        return (
          <button
            key={tab.key}
            type="button"
            onClick={() => onChange(tab.key)}
            className={`flex-1 py-2.5 rounded-lg font-medium text-sm transition-all ${
              active
                ? 'bg-surface-container-high text-primary'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
