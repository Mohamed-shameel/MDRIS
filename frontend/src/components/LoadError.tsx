import { AlertOctagon } from 'lucide-react';
import { isAxiosError } from 'axios';

/** Extract the backend's `detail` (e.g. the 503 reason) from a failed request. */
export const errorDetail = (err: unknown): string => {
  if (isAxiosError(err)) {
    if (!err.response) return 'Cannot reach the MDRIS backend. Is it running on port 8000?';
    const d = err.response.data?.detail;
    return `HTTP ${err.response.status}: ${typeof d === 'string' ? d : JSON.stringify(d ?? err.message)}`;
  }
  return String(err);
};

/** Honest error panel: shows the backend's reason instead of placeholder data. */
export default function LoadError({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-red-300 border-l-4 border-l-red-500 bg-red-50 p-4 text-sm">
      <AlertOctagon size={18} className="mt-0.5 shrink-0 text-red-500" />
      <div>
        <p className="font-semibold text-red-700">{title}</p>
        <p className="text-red-600 font-mono text-xs mt-1 break-all">{detail}</p>
      </div>
    </div>
  );
}
