'use client';

type SourceMode = 'sample' | 'upload';

interface ImageSourceTabsProps {
  value: SourceMode;
  onChange: (value: SourceMode) => void;
}

export default function ImageSourceTabs({ value, onChange }: ImageSourceTabsProps) {
  const tabs: Array<{ key: SourceMode; label: string; hint: string }> = [
    { key: 'sample', label: 'Sample Images', hint: 'Select from Assets images' },
    { key: 'upload', label: 'Upload Image', hint: 'Choose your local file' },
  ];

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-2">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {tabs.map((tab) => {
          const active = tab.key === value;
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => onChange(tab.key)}
              className={[
                'rounded-md border px-3 py-2 text-left transition-colors',
                active
                  ? 'border-green-500 bg-green-50 text-green-900'
                  : 'border-gray-200 bg-white text-gray-700 hover:border-green-300',
              ].join(' ')}
            >
              <p className="text-sm font-semibold">{tab.label}</p>
              <p className="text-xs text-gray-500 mt-0.5">{tab.hint}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
