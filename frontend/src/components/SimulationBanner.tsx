import { FlaskConical } from 'lucide-react';

/** Sticky, non-dismissable notice shown on every page that displays simulated incidents. */
export default function SimulationBanner() {
  return (
    <div className="sticky top-0 z-30 mb-6 flex items-start gap-3 rounded-lg border border-amber-300 border-l-4 border-l-amber-500 bg-amber-50/95 px-4 py-3 text-[13px] text-amber-900 backdrop-blur">
      <FlaskConical size={18} className="mt-0.5 shrink-0 text-amber-600" />
      <p>
        <strong className="uppercase">Simulated incidents.</strong> public dataset images and held-out crisis tweets processed by the real
        models. Locations and times are synthetic. Not live data.
      </p>
    </div>
  );
}
