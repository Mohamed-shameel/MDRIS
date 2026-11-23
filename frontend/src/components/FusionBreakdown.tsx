import type { FusionResult } from '../services/api';

const f = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? '—' : v.toFixed(d));

/**
 * Explainability panel: the fusion formula with the actual numbers substituted,
 * plus a stacked bar of how much vision and text contribute to the score.
 */
export default function FusionBreakdown({ fusion }: { fusion: FusionResult }) {
  const b = fusion.breakdown;
  const total = b.eff_v + b.eff_n;
  const visionPart = total > 0 && b.S_v !== null ? (100 * b.eff_v * b.S_v) / total : 0;
  const textPart = total > 0 && b.S_n !== null ? (100 * b.eff_n * b.S_n) / total : 0;

  const rows = [
    {
      name: 'Vision',
      present: b.S_v !== null,
      s: b.S_v, conf: b.conf_v, rel: b.reliability_v, w: b.weights.vision, eff: b.eff_v,
      formula: `eff_v = ${b.weights.vision} × ${f(b.reliability_v, 2)} × ${f(b.conf_v)} = ${f(b.eff_v)}`,
    },
    {
      name: 'Text',
      present: b.S_n !== null,
      s: b.S_n, conf: b.conf_n, rel: b.reliability_n, w: b.weights.nlp, eff: b.eff_n,
      formula: `eff_n = ${b.weights.nlp} × ${f(b.reliability_n, 2)} × ${f(b.conf_n)} = ${f(b.eff_n)}`,
    },
  ];

  const numerator = [
    b.S_v !== null ? `${f(b.eff_v)}×${f(b.S_v)}` : null,
    b.S_n !== null ? `${f(b.eff_n)}×${f(b.S_n)}` : null,
  ].filter(Boolean).join(' + ');
  const denominator = [b.S_v !== null ? f(b.eff_v) : null, b.S_n !== null ? f(b.eff_n) : null].filter(Boolean).join(' + ');

  return (
    <div className="space-y-4 text-sm">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-slate-500 border-b border-console-line">
              <th className="py-1.5 pr-2">Modality</th>
              <th className="py-1.5 pr-2">Score S</th>
              <th className="py-1.5 pr-2">Confidence</th>
              <th className="py-1.5 pr-2">Reliability</th>
              <th className="py-1.5 pr-2">Weight</th>
              <th className="py-1.5">eff</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((r) => (
              <tr key={r.name} className={`border-b border-slate-100 dark:border-slate-800/60 ${r.present ? '' : 'text-slate-400'}`}>
                <td className="py-1.5 pr-2 font-sans font-semibold">{r.name}{!r.present && ' (absent)'}</td>
                <td className="py-1.5 pr-2">{f(r.s)}</td>
                <td className="py-1.5 pr-2">{f(r.conf)}</td>
                <td className="py-1.5 pr-2">{f(r.rel, 2)}</td>
                <td className="py-1.5 pr-2">{r.w}</td>
                <td className="py-1.5">{f(r.eff)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-lg bg-console-bg border border-console-line p-3 font-mono text-xs space-y-1 overflow-x-auto">
        {rows.filter((r) => r.present).map((r) => <p key={r.name}>{r.formula}</p>)}
        <p>
          priority = 100 × ({numerator}) / ({denominator}) = <strong>{fusion.priority_score.toFixed(1)}</strong>
        </p>
        <p>evidence = {denominator} = <strong>{fusion.evidence_strength.toFixed(3)}</strong> ({fusion.evidence_level})</p>
        {b.S_v !== null && b.S_n !== null && (
          <p>
            |S_v − S_n| = {Math.abs(b.S_v - b.S_n).toFixed(3)} {Math.abs(b.S_v - b.S_n) > 0.5 ? '> 0.5 → conflict' : '≤ 0.5 → no conflict'}
          </p>
        )}
      </div>

      <div>
        <div className="flex justify-between text-xs text-slate-500 mb-1">
          <span>Contribution to the score</span>
          <span>{fusion.priority_score.toFixed(1)} / 100</span>
        </div>
        <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
          <div className="bg-blue-500" style={{ width: `${visionPart}%` }} title={`Vision ${visionPart.toFixed(1)}`} />
          <div className="bg-purple-500" style={{ width: `${textPart}%` }} title={`Text ${textPart.toFixed(1)}`} />
        </div>
        <div className="flex gap-4 mt-1.5 text-xs text-slate-500">
          <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-blue-500" /> Vision {visionPart.toFixed(1)}</span>
          <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-purple-500" /> Text {textPart.toFixed(1)}</span>
        </div>
      </div>
      <p className="text-[11px] text-slate-400">
        Weather is not an input. Weights and reliabilities are documented design choices, not calibrated values.
      </p>
    </div>
  );
}
