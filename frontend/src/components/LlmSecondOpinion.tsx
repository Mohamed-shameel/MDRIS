import { useEffect, useState } from 'react';
import { Loader2, MessageSquareText, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import DataOriginBadge from './DataOriginBadge';
import { CLASS_LABELS } from './labels';
import { llmVisionCheck, type LlmVisionResult, type VisionClass } from '../services/api';

const PEOPLE: VisionClass[] = ['person_in_water', 'person_in_vehicle', 'person'];
const people = (counts: Record<VisionClass, number>) => PEOPLE.reduce((n, c) => n + (counts[c] ?? 0), 0);

/**
 * Second opinion from an external LLM (Groq), shown beside the YOLOv8n result (D-36).
 * Always labelled with the LLM's name; never used in the priority score.
 */
export default function LlmSecondOpinion({ file, yoloCounts }: {
  file: File | null;
  yoloCounts?: Record<VisionClass, number> | null;
}) {
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState<LlmVisionResult | null>(null);

  // A new image invalidates the previous opinion
  useEffect(() => { setResult(null); }, [file]);

  if (!file) return null;

  const run = async () => {
    setLoading(true);
    try {
      setResult((await llmVisionCheck(file)).data);
    } catch {
      // Interceptor shows the toast (503 not configured, 502 Groq failure)
    } finally {
      setLoading(false);
    }
  };

  const llmPeople  = result ? people(result.counts) : 0;
  const yoloPeople = yoloCounts ? people(yoloCounts) : null;
  const disagree   = result && yoloPeople !== null && llmPeople !== yoloPeople;

  return (
    <Card className="border-violet-200 shadow-sm">
      <CardHeader className="pb-3 border-b border-slate-100">
        <CardTitle className="flex items-center justify-between text-sm">
          <span className="flex items-center gap-2">
            <MessageSquareText size={15} className="text-violet-600" />
            Second opinion: LLM vision {result ? `(${result.model.provider} · ${result.model.name})` : '(Groq)'}
          </span>
          {result && <DataOriginBadge origin={result.data_origin} />}
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4 space-y-3">
        <p className="text-xs text-slate-500">
          An external LLM, not YOLOv8n. It is shown for comparison only and is
          <strong> not used in the priority score</strong>. It gives counts and descriptions, no confidence scores.
        </p>

        {!result && (
          <button
            onClick={run}
            disabled={loading}
            className="w-full py-2 px-4 rounded-xl text-sm font-semibold border border-violet-300 text-violet-800 bg-violet-50 hover:bg-violet-100 disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {loading ? <><Loader2 size={16} className="animate-spin" /> Asking the LLM…</> : 'Get LLM second opinion'}
          </button>
        )}

        {result && (
          <>
            {disagree && (
              <div className="flex items-start gap-2 p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-900 text-xs">
                <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                <span>
                  Disagreement: the LLM sees {llmPeople} {llmPeople === 1 ? 'person' : 'people'}, YOLOv8n detected {yoloPeople}.
                  Verify the image before relying on the vision score.
                </span>
              </div>
            )}
            <p className="text-sm">{result.summary}</p>
            {result.objects.length === 0 ? (
              <p className="text-sm text-slate-500">The LLM reported no people or vehicles.</p>
            ) : (
              <ul className="text-sm space-y-1">
                {result.objects.map((o, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="font-medium whitespace-nowrap">{CLASS_LABELS[o.class_name]}:</span>
                    <span className="text-slate-600">{o.description}</span>
                  </li>
                ))}
              </ul>
            )}
            <p className="text-xs text-slate-400">
              Flood visible: {result.flood_visible ? 'yes' : 'no'} · {result.latency_ms.toFixed(0)} ms
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
