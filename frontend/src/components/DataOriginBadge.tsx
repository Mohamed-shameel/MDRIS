import type { DataOrigin } from '../services/api';

const STYLES: Record<DataOrigin, { label: string; className: string; title: string }> = {
  simulated: {
    label: 'SIMULATED',
    className: 'bg-amber-100 text-amber-800 border-amber-300',
    title: 'Simulated incidents: real dataset images and held-out tweets run through the real models; locations and times are synthetic',
  },
  static_demo: {
    label: 'STATIC DEMO',
    className: 'bg-slate-200 text-slate-600 border-slate-300',
    title: 'Fixed demo data; not connected to any live system',
  },
  model: {
    label: 'MODEL OUTPUT',
    className: 'bg-console-soft text-accent-dark border-blue-200',
    title: 'Computed by the MDRIS models (YOLOv8n, fine-tuned DistilBERT) or from their saved evaluation results',
  },
  external_llm: {
    label: 'EXTERNAL LLM',
    className: 'bg-violet-100 text-violet-800 border-violet-300',
    title: 'Second opinion from an external LLM (Groq); not a trained MDRIS model and never used in the priority score',
  },
  live: {
    label: 'LIVE',
    className: 'bg-green-100 text-green-800 border-green-300',
    title: 'Read live from the running system or an external API',
  },
};

/** Pill showing where the data on a card comes from (reads `data_origin`). */
export default function DataOriginBadge({ origin }: { origin: DataOrigin | undefined | null }) {
  if (!origin) return null;
  const s = STYLES[origin];
  return (
    <span
      title={s.title}
      className={`inline-flex items-center px-2 py-0.5 rounded-md border text-[10px] font-bold uppercase tracking-wider font-sans ${s.className}`}
    >
      {s.label}
    </span>
  );
}
