import { useEffect, type ReactNode } from 'react';
import { X, MapPin, Clock, Image as ImageIcon, MessageSquare, Siren } from 'lucide-react';
import { serverUrl, type Incident } from '../services/api';
import DataOriginBadge from './DataOriginBadge';
import RiskBadge from './RiskBadge';
import EvidenceMeter from './EvidenceMeter';
import ConflictFlag from './ConflictFlag';
import SingleSourceChip from './SingleSourceChip';
import FusionBreakdown from './FusionBreakdown';
import UrgencyBars from './UrgencyBars';
import { CLASS_LABELS, SOURCE_LABELS } from './labels';

const Chip = ({ children }: { children: ReactNode }) => (
  <span className="inline-flex items-center rounded-full bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-2 py-0.5 text-[11px] font-medium text-slate-600 dark:text-slate-300">
    {children}
  </span>
);

/** Right-side drawer with everything behind one simulated incident's priority. */
export default function IncidentDrawer({ incident, onClose }: { incident: Incident | null; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!incident) return null;
  const { fusion } = incident;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <aside className="relative h-full w-full max-w-xl overflow-y-auto bg-white dark:bg-slate-950 border-l border-console-line shadow-2xl">
        <div className="sticky top-0 z-10 flex items-center justify-between gap-3 border-b border-console-line bg-white/95 dark:bg-slate-950/95 px-6 py-4 backdrop-blur">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold">{incident.id}</h3>
              <DataOriginBadge origin="simulated" />
            </div>
            <p className="text-xs text-slate-500 flex items-center gap-3 mt-1">
              <span className="flex items-center gap-1"><MapPin size={12} /> {incident.location.name}, {incident.location.district}</span>
              <span className="flex items-center gap-1"><Clock size={12} /> {new Date(incident.created_at).toLocaleString()}</span>
            </p>
          </div>
          <button onClick={onClose} className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800" aria-label="Close">
            <X size={18} />
          </button>
        </div>

        <div className="space-y-6 p-6">
          <section className="flex flex-wrap items-center gap-3">
            <RiskBadge band={fusion.risk_band} size="lg" />
            <span className="text-2xl font-bold">{fusion.priority_score.toFixed(1)}</span>
            <EvidenceMeter value={fusion.evidence_strength} />
            {fusion.conflict && <ConflictFlag withLabel />}
            {fusion.single_source && <SingleSourceChip />}
          </section>

          <section className="rounded-xl border border-console-line p-4">
            <p className="text-xs uppercase tracking-wide text-slate-500 mb-1 flex items-center gap-1"><Siren size={12} /> Suggested action</p>
            <p className="font-semibold">{fusion.suggested_action}</p>
            <p className="text-xs text-slate-500 mt-1">Dominant class: {CLASS_LABELS[fusion.dominant_class]}</p>
          </section>

          <section className="space-y-2">
            <h4 className="text-sm font-semibold flex items-center gap-2"><ImageIcon size={14} /> Image</h4>
            {incident.image && incident.vision ? (
              <>
                <img src={serverUrl(incident.image.url)} alt={`${incident.id} annotated`} className="w-full rounded-xl border border-console-line" />
                <div className="flex flex-wrap gap-2">
                  <Chip>Source: {SOURCE_LABELS[incident.image.source]}</Chip>
                  <Chip>S_v {incident.vision.vision_score.toFixed(3)}</Chip>
                  <Chip>{incident.vision.detections.length} detections</Chip>
                  <Chip>Dataset file: {incident.image.dataset_file}</Chip>
                </div>
              </>
            ) : (
              <p className="text-sm text-slate-400">No image for this incident.</p>
            )}
          </section>

          <section className="space-y-2">
            <h4 className="text-sm font-semibold flex items-center gap-2"><MessageSquare size={14} /> Text</h4>
            {incident.text && incident.nlp ? (
              <>
                <blockquote className="rounded-xl bg-console-bg border border-console-line p-3 text-sm italic">
                  {incident.text.content}
                </blockquote>
                <UrgencyBars probabilities={incident.nlp.probabilities} />
                <div className="flex flex-wrap gap-2">
                  <Chip>Source: {SOURCE_LABELS[incident.text.source]}</Chip>
                  <Chip>Predicted: {incident.nlp.urgency}</Chip>
                  <Chip>HumAID label: {incident.text.humaid_label}</Chip>
                  <Chip>Event: {incident.text.event}</Chip>
                </div>
              </>
            ) : (
              <p className="text-sm text-slate-400">No text for this incident.</p>
            )}
          </section>

          <section className="space-y-2">
            <h4 className="text-sm font-semibold">How the score was computed</h4>
            <FusionBreakdown fusion={fusion} />
          </section>
        </div>
      </aside>
    </div>
  );
}
