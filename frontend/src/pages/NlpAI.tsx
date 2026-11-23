import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import {
  analyzeNLP, fetchNLPLabels, TEXT_SOURCES,
  type NlpLabels, type NlpResult, type TextSource,
} from '../services/api';
import { toast } from 'sonner';
import { MessageSquare, Loader2, Brain, Zap, BarChart2 } from 'lucide-react';
import DataOriginBadge from '../components/DataOriginBadge';
import UrgencyBars from '../components/UrgencyBars';
import { SOURCE_LABELS, URGENCY_COLORS } from '../components/labels';

const EXAMPLE_TEXTS = [
  'Water entering houses near the lake, elderly people stuck on the first floor, please send a boat.',
  'Road to the district hospital is submerged; avoid the bridge until water levels drop.',
  'Volunteers are collecting food packets and drinking water for flood relief camps this weekend.',
  'Praying for everyone affected by the floods. Stay safe.',
];

// ── Main Component ───────────────────────────────────────────────────────────

export default function NlpAI() {
  const [text, setText]       = useState('');
  const [source, setSource]   = useState<TextSource>('citizen_sms');
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState<NlpResult | null>(null);
  const [labels, setLabels]   = useState<NlpLabels | null>(null);

  useEffect(() => {
    fetchNLPLabels().then((res) => setLabels(res.data)).catch(() => {});
  }, []);

  const handleAnalyze = async () => {
    if (text.trim().length < 3) {
      toast.warning('Please enter at least 3 characters.');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const res = await analyzeNLP(text, source);
      setResult(res.data);
    } catch {
      // Interceptor handles toast (503 if the model is not loaded)
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-console-soft border border-console-line">
            <Brain className="text-accent" size={22} />
          </div>
          <h2 className="console-title text-[28px]">Urgency NLP (DistilBERT)</h2>
        </div>
        <p className="text-slate-500 ml-12">
          Fine-tuned DistilBERT classifies incident text as Low, Medium or High urgency (trained on HumAID flood tweets).
        </p>
      </div>

      {/* Input Card */}
      <Card className="border-console-line shadow-md">
        <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
          <CardTitle className="flex items-center gap-2 text-base">
            <MessageSquare size={18} className="text-purple-500" />
            Incident text
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-4 space-y-4">
          <textarea
            id="nlp-text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter a flood report, control-room transcript or social media post…"
            rows={5}
            maxLength={2000}
            className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-700
              bg-console-bg text-slate-900 dark:text-slate-100
              placeholder-slate-400 text-sm resize-none
              focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent
              transition-all"
          />

          <div>
            <p className="text-xs text-slate-400 mb-2 font-medium">Try an example (written for the demo, not from the dataset):</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_TEXTS.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => setText(ex)}
                  className="text-xs px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-700
                    text-slate-600 dark:text-slate-300 hover:border-blue-400 hover:text-purple-600
                    dark:hover:text-purple-400 transition-colors truncate max-w-[220px]"
                >
                  {ex.slice(0, 40)}…
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
            <label className="block">
              <span className="text-xs font-medium text-slate-500">Text source</span>
              <select
                value={source}
                onChange={(e) => setSource(e.target.value as TextSource)}
                className="mt-1 block rounded-xl border border-slate-200 dark:border-slate-700 bg-console-bg px-3 py-2 text-sm"
              >
                {TEXT_SOURCES.map((s) => <option key={s} value={s}>{SOURCE_LABELS[s]}</option>)}
              </select>
            </label>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400">{text.length} / 2000 chars</span>
              <button
                id="nlp-analyze-btn"
                onClick={handleAnalyze}
                disabled={loading || !text.trim()}
                className="px-6 py-2.5 rounded-xl font-semibold text-sm transition-all duration-200
                  bg-accent hover:bg-accent-dark text-white shadow-md shadow-blue-500/20
                  disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none
                  flex items-center gap-2"
              >
                {loading
                  ? <><Loader2 size={16} className="animate-spin" /> Analyzing…</>
                  : <><Zap size={16} /> Analyze Text</>
                }
              </button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="border-console-line shadow-sm md:col-span-2">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-500 font-medium uppercase tracking-wide flex items-center justify-between">
                <span>Predicted urgency</span>
                <DataOriginBadge origin={result.data_origin} />
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0 space-y-4">
              <span
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-sm font-bold"
                style={{ color: URGENCY_COLORS[result.urgency], borderColor: `${URGENCY_COLORS[result.urgency]}66`, backgroundColor: `${URGENCY_COLORS[result.urgency]}1A` }}
              >
                {result.urgency}
              </span>
              <UrgencyBars probabilities={result.probabilities} />
            </CardContent>
          </Card>

          <Card className="border-console-line shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-500 font-medium uppercase tracking-wide flex items-center gap-2">
                <BarChart2 size={14} /> Scores
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0 space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500">NLP score S_n</span>
                <span className="text-sm font-bold">{result.nlp_score.toFixed(3)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500">Confidence</span>
                <span className="text-sm font-bold">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500">Inference time</span>
                <span className="text-sm font-bold">{result.inference_ms.toFixed(0)} ms</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500">Model</span>
                <span className="text-xs font-mono truncate max-w-[140px] text-right" title={result.model.path}>{result.model.name}</span>
              </div>
              <p className="text-[11px] text-slate-400 pt-1">S_n = Σ p_k · value_k (Low 0.15, Medium 0.50, High 0.90)</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Label mapping */}
      {labels && (
        <Card className="border-console-line shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center justify-between">
              <span>How urgency labels were derived from HumAID categories</span>
              <DataOriginBadge origin={labels.data_origin} />
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <p className="text-xs text-slate-500 mb-3">
              HumAID labels describe information type, not urgency. This mapping is a documented project decision and the
              weakest link of the NLP model.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs">
              {labels.humaid_mapping.map((m) => (
                <div key={m.humaid_label} className="flex justify-between border-b border-slate-100 dark:border-slate-800/60 py-1">
                  <span className="font-mono text-slate-600 dark:text-slate-300">{m.humaid_label}</span>
                  <span className="font-semibold" style={{ color: URGENCY_COLORS[m.urgency] }}>{m.urgency}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
