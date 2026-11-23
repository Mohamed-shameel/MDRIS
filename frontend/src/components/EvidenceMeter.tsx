/** Three-segment evidence-strength bar: Weak < 0.40 <= Fair < 0.70 <= Strong. */
export default function EvidenceMeter({ value, showValue = true }: { value: number; showValue?: boolean }) {
  const level = value >= 0.7 ? 'Strong' : value >= 0.4 ? 'Fair' : 'Weak';
  const filled = level === 'Strong' ? 3 : level === 'Fair' ? 2 : 1;
  const color = level === 'Strong' ? 'bg-emerald-500' : level === 'Fair' ? 'bg-amber-500' : 'bg-slate-400';
  return (
    <div className="flex items-center gap-2" title={`Evidence strength ${value.toFixed(2)} (${level})`}>
      <div className="flex gap-0.5">
        {[1, 2, 3].map((i) => (
          <span key={i} className={`h-2 w-4 rounded-sm ${i <= filled ? color : 'bg-slate-200 dark:bg-slate-700'}`} />
        ))}
      </div>
      <span className="text-xs text-slate-500 whitespace-nowrap">
        {level}
        {showValue && ` · ${value.toFixed(2)}`}
      </span>
    </div>
  );
}
