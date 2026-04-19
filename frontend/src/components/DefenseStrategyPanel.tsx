'use client';

interface DefenseStrategyPanelProps {
  compact?: boolean;
}

const DECISIONS = [
  {
    title: 'Time-Based Benchmarking',
    subtitle: 'Batch-based -> temporal steady-state window',
    reasoning:
      'High-end GPUs process batches too quickly for stable sensor capture. Time-based windows produce more reliable power and CO2 readings.',
    icon: 'schedule',
  },
  {
    title: 'Structured Pruning',
    subtitle: 'Hardware-aware filter removal path',
    reasoning:
      'Unstructured sparsity often keeps dense execution with mask overhead. Structured pruning targets whole filters/channels for practical speed and energy gains.',
    icon: 'tune',
  },
  {
    title: 'Architectural Router',
    subtitle: 'Compression policy by model family',
    reasoning:
      'Lightweight models usually have limited redundancy. Routing them to distillation avoids the sustainability trap of accuracy collapse from over-pruning.',
    icon: 'route',
  },
] as const;

export function DefenseStrategyPanel({ compact = false }: DefenseStrategyPanelProps) {
  return (
    <section className="bg-surface-container rounded-2xl p-6 space-y-5 ring-1 ring-primary/10">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Defense Strategy</p>
          <h3 className="text-xl font-headline font-semibold text-on-surface mt-1">
            Engineering Decisions Behind GreenAI Compression
          </h3>
        </div>
        <span className="material-symbols-outlined text-primary">campaign</span>
      </div>

      <div className={`grid gap-3 ${compact ? 'grid-cols-1' : 'grid-cols-1 xl:grid-cols-3'}`}>
        {DECISIONS.map((decision) => (
          <article key={decision.title} className="bg-surface-container-low rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-[18px]">{decision.icon}</span>
              <p className="text-sm font-semibold text-on-surface">{decision.title}</p>
            </div>
            <p className="text-[11px] uppercase tracking-wider text-on-surface-variant font-technical">{decision.subtitle}</p>
            <p className="text-sm text-on-surface-variant">{decision.reasoning}</p>
          </article>
        ))}
      </div>

      <div className="bg-primary/8 rounded-xl p-4 border border-primary/20">
        <p className="text-[10px] uppercase tracking-widest font-bold text-primary">Final Defense Tip</p>
        <p className="text-sm text-on-surface mt-1">
          If some 3070 runs show slightly higher CO2 after compression, frame it as the energy cost of compression training.
          For short-lived deployments, retraining overhead can exceed downstream savings, which is a valid sustainability tradeoff.
        </p>
      </div>
    </section>
  );
}
