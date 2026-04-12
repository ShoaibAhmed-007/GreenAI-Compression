'use client';

import { ModelComparisonSample } from '@/lib/api';

interface SampleImageSelectorProps {
	samples: ModelComparisonSample[];
	selectedPath: string;
	onSelect: (sample: ModelComparisonSample) => void;
}

function sampleKey(sample: ModelComparisonSample): string {
	return sample.source_path || `sample-${sample.id}`;
}

export default function SampleImageSelector({
	samples,
	selectedPath,
	onSelect,
}: SampleImageSelectorProps) {
	if (!samples.length) {
		return (
			<div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm text-gray-500">
				No sample images available.
			</div>
		);
	}

	return (
		<div className="rounded-lg border border-gray-200 bg-white p-3">
			<div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
				{samples.map((sample) => {
					const key = sampleKey(sample);
					const active = selectedPath === (sample.source_path || '');

					return (
						<button
							key={key}
							type="button"
							onClick={() => onSelect(sample)}
							className={[
								'rounded-md border bg-white p-2 text-left transition-colors',
								active
									? 'border-green-500 ring-2 ring-green-200'
									: 'border-gray-200 hover:border-green-300',
							].join(' ')}
						>
							<div className="aspect-square overflow-hidden rounded border border-gray-100 bg-gray-50">
								<img
									src={sample.image_data_url}
									alt={sample.label}
									className="h-full w-full object-cover"
								/>
							</div>
							<p className="mt-2 text-xs font-medium text-gray-700 truncate" title={sample.label}>
								{sample.label}
							</p>
						</button>
					);
				})}
			</div>
		</div>
	);
}
