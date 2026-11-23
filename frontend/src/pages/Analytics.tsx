import { useEffect, useState, type ReactNode } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList } from 'recharts';
import { Eye, Brain, GitMerge, Database } from 'lucide-react';
import { fetchEvaluation, serverUrl, type EvaluationPayload } from '../services/api';
import DataOriginBadge from '../components/DataOriginBadge';
import LoadError, { errorDetail } from '../components/LoadError';

const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? 'n/a' : v.toFixed(d));

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-console-bg border border-console-line p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-2xl font-bold">{value}</p>
    </div>
  );
}

function Figure({ src, caption }: { src: string | undefined; caption: string }) {
  if (!src) return null;
  return (
    <figure className="space-y-1">
      <a href={serverUrl(src)} target="_blank" rel="noreferrer">
        <img src={serverUrl(src)} alt={caption} className="w-full rounded-lg border border-console-line bg-white" />
      </a>
      <figcaption className="text-xs text-slate-500">{caption}</figcaption>
    </figure>
  );
}

function Section({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <Card className="border-console-line shadow-md">
      <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
        <CardTitle className="flex items-center justify-between text-base">
          <span className="flex items-center gap-2">{icon} {title}</span>
          <DataOriginBadge origin="model" />
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4 space-y-5">{children}</CardContent>
    </Card>
  );
}

export default function Analytics() {
  const [data, setData] = useState<EvaluationPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchEvaluation().then((res) => setData(res.data)).catch((err) => setError(errorDetail(err)));
  }, []);

  if (error) return <LoadError title="Evaluation unavailable" detail={error} />;
  if (!data) return <div className="animate-pulse h-96 w-full bg-slate-100 dark:bg-slate-900 rounded-xl" />;

  const v = data.vision.metrics;
  const n = data.nlp.metrics;
  const fa = data.fusion.analysis;
  const vfig = data.vision.figure_urls;
  const nfig = data.nlp.figure_urls;

  const baselineRows = n
    ? [
        { name: n.baselines.majority?.name ?? 'Majority', f1: n.baselines.majority?.macro_f1 ?? 0 },
        { name: n.baselines.tfidf_lr?.name ?? 'TF-IDF + LR', f1: n.baselines.tfidf_lr?.macro_f1 ?? 0 },
        { name: n.baselines.zero_shot?.name ?? 'NLI baseline', f1: n.baselines.zero_shot?.macro_f1 ?? 0 },
        { name: 'DistilBERT (fine-tuned)', f1: n.metrics.test.macro_f1 },
      ]
    : [];

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="console-title text-[28px]">Model Evaluation</h2>
          <p className="text-slate-500">Held-out test-split results saved by the training scripts (metrics.json), plus fusion behaviour.</p>
        </div>
        <DataOriginBadge origin={data.data_origin} />
      </div>

      {/* Vision */}
      <Section icon={<Eye size={18} className="text-blue-500" />} title="Vision: fine-tuned YOLOv8n (evaluated, not served)">
        <p className="text-xs text-slate-500">
          The backend currently serves the <strong>COCO-pretrained</strong> YOLOv8n (decision D-37): on clean photos it found people the fine-tuned model missed.
          On this flood test split the fine-tuned model scores higher (person AP50 0.365 vs 0.137, same scoring code; see the COCO baseline below).
        </p>
        {!v && <LoadError title="Vision metrics not available" detail={data.vision.reason ?? ''} />}
        {v && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Test mAP50" value={fmt(v.metrics.test.mAP50)} />
              <Stat label="Test mAP50-95" value={fmt(v.metrics.test.mAP50_95)} />
              <Stat label="Epochs run" value={String(v.hyperparameters.epochs_run)} />
              <Stat label="Train time" value={`${v.train_time_minutes} min`} />
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-slate-500 border-b border-console-line">
                  <th className="py-1.5">Class</th><th>Test instances</th><th>Precision</th><th>Recall</th><th>AP50</th>
                </tr>
              </thead>
              <tbody className="font-mono text-xs">
                {Object.entries(v.metrics.test.per_class).map(([cls, m]) => (
                  <tr key={cls} className="border-b border-slate-100 dark:border-slate-800/60">
                    <td className="py-1.5 font-sans">{cls}</td>
                    <td>{m.instances ?? 'n/a'}</td>
                    <td>{fmt(m.precision)}</td>
                    <td>{fmt(m.recall)}</td>
                    <td>{fmt(m.AP50)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {v.notes && (
              <ul className="list-disc pl-5 text-xs text-amber-700 dark:text-amber-400 space-y-1">
                {v.notes.map((note) => <li key={note}>{note}</li>)}
              </ul>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Figure src={vfig['figures/confusion_matrix.png']} caption="Test confusion matrix" />
              <Figure src={vfig['figures/PR_curve.png']} caption="Test precision-recall curve" />
            </div>
            <div>
              <p className="text-sm font-semibold mb-2">Worst 6 test images (ground truth | prediction, ranked by FN + FP)</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(vfig).filter(([k]) => k.includes('failures/')).map(([k, url]) => (
                  <Figure key={k} src={url} caption={k.split('/').pop() ?? k} />
                ))}
              </div>
            </div>
          </>
        )}
      </Section>

      {/* NLP */}
      <Section icon={<Brain size={18} className="text-purple-500" />} title="Urgency NLP: fine-tuned DistilBERT">
        {!n && <LoadError title="NLP metrics not available" detail={data.nlp.reason ?? ''} />}
        {n && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Test accuracy" value={fmt(n.metrics.test.accuracy)} />
              <Stat label="Test macro-F1" value={fmt(n.metrics.test.macro_f1)} />
              <Stat label="Weighted-F1" value={fmt(n.metrics.test.weighted_f1)} />
              <Stat label="Train time" value={`${n.train_time_minutes} min`} />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <table className="w-full text-sm self-start">
                <thead>
                  <tr className="text-left text-xs text-slate-500 border-b border-console-line">
                    <th className="py-1.5">Urgency</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-xs">
                  {Object.entries(n.metrics.test.per_class).map(([cls, m]) => (
                    <tr key={cls} className="border-b border-slate-100 dark:border-slate-800/60">
                      <td className="py-1.5 font-sans">{cls}</td>
                      <td>{fmt(m.precision)}</td><td>{fmt(m.recall)}</td><td>{fmt(m.f1)}</td><td>{m.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div>
                <p className="text-sm font-semibold mb-2">Baselines vs fine-tuned (test macro-F1)</p>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={baselineRows} margin={{ top: 16, right: 8, left: -16, bottom: 0 }}>
                      <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                      <Tooltip formatter={(value) => Number(value).toFixed(3)} />
                      <Bar dataKey="f1" radius={[4, 4, 0, 0]}>
                        {baselineRows.map((row, i) => (
                          <Cell key={row.name} fill={i === baselineRows.length - 1 ? '#2563EB' : '#94A3B8'} />
                        ))}
                        <LabelList dataKey="f1" position="top" formatter={(value) => Number(value).toFixed(3)} style={{ fontSize: 10 }} />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-slate-500 border-b border-console-line">
                  <th className="py-1.5">Model</th><th>Accuracy</th><th>Macro-F1</th><th>F1 Low</th><th>F1 Medium</th><th>F1 High</th>
                </tr>
              </thead>
              <tbody className="font-mono text-xs">
                {Object.values(n.baselines).map((b) => (
                  <tr key={b.name} className="border-b border-slate-100 dark:border-slate-800/60">
                    <td className="py-1.5 font-sans">{b.name}{b.model ? ` (${b.model})` : ''}</td>
                    <td>{fmt(b.accuracy)}</td><td>{fmt(b.macro_f1)}</td>
                    <td>{fmt(b.per_class_f1.Low)}</td><td>{fmt(b.per_class_f1.Medium)}</td><td>{fmt(b.per_class_f1.High)}</td>
                  </tr>
                ))}
                <tr className="font-bold">
                  <td className="py-1.5 font-sans">Fine-tuned DistilBERT</td>
                  <td>{fmt(n.metrics.test.accuracy)}</td><td>{fmt(n.metrics.test.macro_f1)}</td>
                  <td>{fmt(n.metrics.test.per_class.Low?.f1)}</td><td>{fmt(n.metrics.test.per_class.Medium?.f1)}</td>
                  <td>{fmt(n.metrics.test.per_class.High?.f1)}</td>
                </tr>
              </tbody>
            </table>
            <div>
              <p className="text-sm font-semibold mb-1">Per-event macro-F1</p>
              <div className="flex flex-wrap gap-2 text-xs">
                {Object.entries(n.metrics.test.per_event_macro_f1).map(([event, f1]) => (
                  <span key={event} className="rounded-full border border-slate-200 dark:border-slate-700 px-2 py-0.5">
                    {event}: <strong>{fmt(f1)}</strong>
                  </span>
                ))}
              </div>
            </div>
            {n.notes && (
              <ul className="list-disc pl-5 text-xs text-amber-700 dark:text-amber-400 space-y-1">
                {n.notes.map((note) => <li key={note}>{note}</li>)}
              </ul>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Figure src={nfig['figures/confusion_matrix.png']} caption="Test confusion matrix" />
              <Figure src={nfig['figures/training_curve.png']} caption="Training curve (validation macro-F1 drives early stopping)" />
            </div>
          </>
        )}
      </Section>

      {/* Fusion */}
      <Section icon={<GitMerge size={18} className="text-emerald-500" />} title="Fusion: ablation and sensitivity">
        {!fa && <LoadError title="Fusion analysis not available" detail={data.fusion.reason ?? ''} />}
        {fa && (
          <>
            <p className="text-xs text-slate-500">
              No ground-truth priority exists, so the rule is evaluated by behaviour over the {fa.n_incidents} simulated incidents:
              how many change risk band and how similar the ranking stays (Kendall τ, 1 = identical order).
            </p>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-slate-500 border-b border-console-line">
                  <th className="py-1.5">Ablation</th><th>Band changes</th><th>Kendall τ (all)</th><th>Kendall τ (top-10)</th>
                </tr>
              </thead>
              <tbody className="text-xs">
                {fa.ablation.map((row) => (
                  <tr key={row.variant} className="border-b border-slate-100 dark:border-slate-800/60">
                    <td className="py-1.5"><p className="font-semibold">{row.variant}</p><p className="text-slate-500">{row.description}</p></td>
                    <td className="font-mono">{row.band_changes}</td>
                    <td className="font-mono">{fmt(row.kendall_tau)}</td>
                    <td className="font-mono">{fmt(row.kendall_tau_top10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-slate-500 border-b border-console-line">
                  <th className="py-1.5">Sensitivity setting</th><th>Band changes</th><th>Kendall τ (all)</th><th>Kendall τ (top-10)</th>
                </tr>
              </thead>
              <tbody className="text-xs font-mono">
                {[...fa.sensitivity.vision_weight, ...fa.sensitivity.reliability_scale].map((row) => (
                  <tr key={row.setting} className="border-b border-slate-100 dark:border-slate-800/60">
                    <td className="py-1.5 font-sans">{row.setting}</td>
                    <td>{row.band_changes}</td><td>{fmt(row.kendall_tau_all)}</td><td>{fmt(row.kendall_tau_top10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {fa.notes && (
              <ul className="list-disc pl-5 text-xs text-slate-500 space-y-1">
                {fa.notes.map((note) => <li key={note}>{note}</li>)}
              </ul>
            )}
          </>
        )}
      </Section>

      {/* Datasets */}
      <Section icon={<Database size={18} className="text-slate-500" />} title="Datasets">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          {[v && { kind: 'Vision', d: v.dataset }, n && { kind: 'Text', d: n.dataset }].filter(Boolean).map((x) => {
            const { kind, d } = x as { kind: string; d: { name: string; url: string; licence: string; splits: Record<string, number> } };
            return (
              <div key={kind} className="rounded-xl border border-console-line p-3 space-y-1">
                <p className="text-xs uppercase tracking-wide text-slate-500">{kind}</p>
                <a href={d.url} target="_blank" rel="noreferrer" className="font-semibold text-blue-600 dark:text-blue-400 hover:underline">{d.name}</a>
                <p className="text-xs">Licence: {d.licence}</p>
                <p className="text-xs font-mono">
                  {Object.entries(d.splits).map(([k, c]) => `${k} ${c}`).join(' · ')}
                </p>
              </div>
            );
          })}
        </div>
      </Section>
    </div>
  );
}
