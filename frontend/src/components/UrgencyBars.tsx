import type { UrgencyLabel } from '../services/api';
import { URGENCY_COLORS } from './labels';

const ORDER: UrgencyLabel[] = ['High', 'Medium', 'Low'];

/** Horizontal probability bars for the three urgency classes. */
export default function UrgencyBars({ probabilities }: { probabilities: Record<UrgencyLabel, number> }) {
  return (
    <div className="space-y-2">
      {ORDER.map((label) => {
        const p = probabilities[label] ?? 0;
        return (
          <div key={label} className="flex items-center gap-3">
            <span className="w-16 text-xs font-semibold" style={{ color: URGENCY_COLORS[label] }}>{label}</span>
            <div className="flex-1 h-2.5 rounded-full bg-slate-100 dark:bg-slate-800">
              <div className="h-2.5 rounded-full" style={{ width: `${p * 100}%`, backgroundColor: URGENCY_COLORS[label] }} />
            </div>
            <span className="w-12 text-right text-xs font-mono text-slate-500">{(p * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}
