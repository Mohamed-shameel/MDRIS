import { AlertTriangle } from 'lucide-react';

/** Amber triangle marking incidents whose image and text disagree. */
export default function ConflictFlag({ withLabel = false }: { withLabel?: boolean }) {
  return (
    <span
      title="Image and text disagree: verify on ground"
      className="inline-flex items-center gap-1 text-amber-500"
    >
      <AlertTriangle size={16} />
      {withLabel && <span className="text-xs font-semibold">Image and text disagree: verify on ground</span>}
    </span>
  );
}
