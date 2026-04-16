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
			<div className="bg-surface-container-low rounded-xl p-4 text-sm text-on-surface-variant">
				No sample images available.
			</div>
		);
	}

	return (
		<div className="grid grid-cols-4 sm:grid-cols-6 gap-3">
			{samples.map((sample) => {
				const key = sampleKey(sample);
				const active = selectedPath === (sample.source_path || '');

				return (
					<button
						key={key}
						type="button"
						onClick={() => onSelect(sample)}
						className={`aspect-square rounded-lg overflow-hidden transition-all cursor-pointer ${
							active
								? 'ring-2 ring-primary ring-offset-4 ring-offset-surface-container opacity-100'
								: 'opacity-60 hover:opacity-100'
						}`}
					>
						<img
							src={sample.image_data_url}
							alt={sample.label}
							className="w-full h-full object-cover"
						/>
					</button>
				);
			})}
		</div>
	);
}
