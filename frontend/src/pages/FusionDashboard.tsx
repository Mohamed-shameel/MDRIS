import { useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import {
  analyzeFusion, analyzeIncident, serverUrl, TEXT_SOURCES, VISION_SOURCES,
  type FusionRequest, type FusionResult, type IncidentAnalysis, type TextSource, type VisionSource,
} from '../services/api';
import { toast } from 'sonner';
import { GitMerge, Loader2, Zap, Upload, X, FlaskConical, Siren } from 'lucide-react';
import DataOriginBadge from '../components/DataOriginBadge';
import RiskBadge from '../components/RiskBadge';
import EvidenceMeter from '../components/EvidenceMeter';
import ConflictFlag from '../components/ConflictFlag';
import SingleSourceChip from '../components/SingleSourceChip';
import FusionBreakdown from '../components/FusionBreakdown';
import LlmSecondOpinion from '../components/LlmSecondOpinion';
import UrgencyBars from '../components/UrgencyBars';
import { CLASS_LABELS, SOURCE_LABELS } from '../components/labels';

// ── Fusion test cases T1–T7 (docs/MDRIS_CONTEXT.md section 8.4) ─────────────

const HI_TEXT = { Low: 0.05, Medium: 0.1, High: 0.85 };
const LO_TEXT = { Low: 0.85, Medium: 0.1, High: 0.05 };

interface TestCase { id: string; title: string; request: FusionRequest; expected: string }

const TEST_CASES: TestCase[] = [
  {
    id: 'T1', title: 'Both strong, officer sources',
    request: { vision: { detections: [{ class_name: 'person_in_water', confidence: 0.9 }], source: 'officer_photo' }, nlp: { probabilities: HI_TEXT, source: 'officer_report' } },
    expected: 'priority 87.0, Critical, evidence 0.88, no conflict',
  },
  {
    id: 'T2', title: 'Vision only',
    request: { vision: { detections: [{ class_name: 'person_in_water', confidence: 0.9 }], source: 'officer_photo' }, nlp: null },
    expected: 'priority 90.0, Critical, evidence 0.54 (renormalized, not capped at 60)',
  },
  {
    id: 'T3', title: 'Text only',
    request: { vision: null, nlp: { probabilities: HI_TEXT, source: 'officer_report' } },
    expected: 'priority 82.2, Critical, evidence 0.34',
  },
  {
    id: 'T4', title: 'Conflicting signals',
    request: { vision: { detections: [{ class_name: 'person_in_water', confidence: 0.9 }], source: 'officer_photo' }, nlp: { probabilities: LO_TEXT, source: 'officer_report' } },
    expected: 'priority 63.8, High, evidence 0.88, conflict = true',
  },
  {
    id: 'T5a', title: 'Reliability: officer photo',
    request: { vision: { detections: [{ class_name: 'person_in_vehicle', confidence: 0.8 }], source: 'officer_photo' }, nlp: { probabilities: LO_TEXT, source: 'citizen_sms' } },
    expected: 'priority 54.4, Moderate, eff_v 0.48, evidence 0.684',
  },
  {
    id: 'T5b', title: 'Same content, citizen photo',
    request: { vision: { detections: [{ class_name: 'person_in_vehicle', confidence: 0.8 }], source: 'citizen_upload' }, nlp: { probabilities: LO_TEXT, source: 'citizen_sms' } },
    expected: 'priority 49.0, Moderate, eff_v 0.288, evidence 0.492 (lower influence, not a blind penalty)',
  },
  {
    id: 'T6', title: 'Image with no detections + urgent text',
    request: { vision: { detections: [], source: 'officer_photo' }, nlp: { probabilities: HI_TEXT, source: 'officer_report' } },
    expected: 'priority 43.7, Moderate, conflict = true (verify on ground)',
  },
  {
    id: 'T7', title: 'Many vehicles, no people',
    request: { vision: { detections: Array.from({ length: 5 }, () => ({ class_name: 'vehicle' as const, confidence: 0.8 })), source: 'officer_photo' }, nlp: null },
    expected: 'S_v 0.472, priority 47.2, Moderate (must not reach High)',
  },
];

const describeRequest = (r: FusionRequest) => {
  const parts: string[] = [];
  if (r.vision) {
    const dets = r.vision.detections.length
      ? r.vision.detections.map((d) => `${d.class_name} ${d.confidence}`).join(', ')
      : 'no detections';
    parts.push(`${SOURCE_LABELS[r.vision.source]}: ${dets}`);
  }
  if (r.nlp) {
    const p = r.nlp.probabilities;
    parts.push(`${SOURCE_LABELS[r.nlp.source]}: Low ${p.Low} / Medium ${p.Medium} / High ${p.High}`);
  }
  return parts.join('  +  ');
};

// ── Result panel ────────────────────────────────────────────────────────────

function FusionResultCard({ fusion, expected }: { fusion: FusionResult; expected?: string }) {
  return (
    <Card className="border-console-line shadow-md">
      <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
        <CardTitle className="flex items-center justify-between text-base">
          <span>Priority decision</span>
          <DataOriginBadge origin={fusion.data_origin} />
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4 space-y-5">
        <div className="flex flex-wrap items-center gap-4">
          <p className="text-5xl font-bold">{fusion.priority_score.toFixed(1)}</p>
          <div className="space-y-2">
            <RiskBadge band={fusion.risk_band} size="lg" />
            <EvidenceMeter value={fusion.evidence_strength} />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {fusion.conflict && <ConflictFlag withLabel />}
            {fusion.single_source && <SingleSourceChip />}
          </div>
        </div>
        {expected && (
          <p className="text-xs rounded-lg bg-console-bg border border-console-line px-3 py-2">
            <span className="font-semibold">Expected (spec §8.4):</span> {expected}
          </p>
        )}
        <div className="rounded-xl border border-console-line p-4">
          <p className="text-xs uppercase tracking-wide text-slate-500 mb-1 flex items-center gap-1"><Siren size={12} /> Suggested action</p>
          <p className="font-semibold">{fusion.suggested_action}</p>
          <p className="text-xs text-slate-500 mt-1">Dominant class: {CLASS_LABELS[fusion.dominant_class]}</p>
        </div>
        <FusionBreakdown fusion={fusion} />
      </CardContent>
    </Card>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────

export default function FusionDashboard() {
  const [image, setImage]             = useState<File | null>(null);
  const [preview, setPreview]         = useState<string | null>(null);
  const [text, setText]               = useState('');
  const [imageSource, setImageSource] = useState<VisionSource>('officer_photo');
  const [textSource, setTextSource]   = useState<TextSource>('citizen_sms');
  const [loading, setLoading]         = useState(false);
  const [analysis, setAnalysis]       = useState<IncidentAnalysis | null>(null);
  const [testCase, setTestCase]       = useState<TestCase | null>(null);
  const [testResult, setTestResult]   = useState<FusionResult | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const pickImage = (f: File | undefined) => {
    if (!f) return;
    if (!['image/jpeg', 'image/png'].includes(f.type)) {
      toast.error('Only JPG and PNG images are supported.');
      return;
    }
    setImage(f);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const clearImage = () => {
    setImage(null);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const runIncident = async () => {
    if (!image && text.trim().length < 3) {
      toast.warning('Provide an image, a text (3+ characters), or both.');
      return;
    }
    setLoading(true);
    setAnalysis(null);
    setTestResult(null);
    try {
      const res = await analyzeIncident({ image, text, imageSource, textSource });
      setAnalysis(res.data);
    } catch {
      // Interceptor shows the toast (503 if a model is not loaded)
    } finally {
      setLoading(false);
    }
  };

  const runTestCase = async (tc: TestCase) => {
    setLoading(true);
    setAnalysis(null);
    setTestResult(null);
    try {
      const res = await analyzeFusion(tc.request);
      setTestResult(res.data);
    } catch {
      // Interceptor shows the toast
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-console-soft border border-console-line">
            <GitMerge className="text-accent" size={22} />
          </div>
          <h2 className="console-title text-[28px]">Incident Analyzer (Fusion)</h2>
        </div>
        <p className="text-slate-500 ml-12">
          Image and/or text → YOLOv8n + fine-tuned DistilBERT → confidence × source-reliability weighted fusion.
          Weather is not an input.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Inputs */}
        <div className="space-y-4">
          <Card className="border-console-line shadow-md">
            <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
              <CardTitle className="text-base">Incident inputs (both optional)</CardTitle>
            </CardHeader>
            <CardContent className="pt-4 space-y-4">
              <div>
                <p className="text-xs font-medium text-slate-500 mb-1">Image</p>
                {preview ? (
                  <div className="relative">
                    <img src={preview} alt="Incident" className="w-full h-48 object-cover rounded-xl" />
                    <button onClick={clearImage} className="absolute top-2 right-2 p-1.5 rounded-full bg-slate-900/70 text-white hover:bg-red-600">
                      <X size={14} />
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => inputRef.current?.click()}
                    className="w-full py-8 rounded-xl border-2 border-dashed border-slate-300 dark:border-slate-700 hover:border-blue-400 text-sm text-slate-500 flex flex-col items-center gap-2"
                  >
                    <Upload size={20} /> Choose a JPG or PNG
                  </button>
                )}
                <input ref={inputRef} type="file" accept="image/jpeg,image/png" className="hidden" onChange={(e) => pickImage(e.target.files?.[0])} />
                <select
                  value={imageSource}
                  onChange={(e) => setImageSource(e.target.value as VisionSource)}
                  className="mt-2 w-full rounded-xl border border-slate-200 dark:border-slate-700 bg-console-bg px-3 py-2 text-sm"
                >
                  {VISION_SOURCES.map((s) => <option key={s} value={s}>Image source: {SOURCE_LABELS[s]}</option>)}
                </select>
              </div>

              <div>
                <p className="text-xs font-medium text-slate-500 mb-1">Text</p>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={4}
                  maxLength={2000}
                  placeholder="Report text, SMS or transcript…"
                  className="w-full px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-console-bg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
                <select
                  value={textSource}
                  onChange={(e) => setTextSource(e.target.value as TextSource)}
                  className="mt-2 w-full rounded-xl border border-slate-200 dark:border-slate-700 bg-console-bg px-3 py-2 text-sm"
                >
                  {TEXT_SOURCES.map((s) => <option key={s} value={s}>Text source: {SOURCE_LABELS[s]}</option>)}
                </select>
              </div>

              <button
                onClick={runIncident}
                disabled={loading}
                className="w-full py-3 rounded-xl font-semibold text-sm bg-accent hover:bg-accent-dark text-white shadow-md shadow-blue-500/20 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {loading ? <><Loader2 size={16} className="animate-spin" /> Analyzing…</> : <><Zap size={16} /> Analyze incident</>}
              </button>
            </CardContent>
          </Card>

          {/* Test cases */}
          <Card className="border-console-line shadow-sm">
            <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
              <CardTitle className="text-base flex items-center gap-2"><FlaskConical size={16} /> Load test case (T1–T7)</CardTitle>
            </CardHeader>
            <CardContent className="pt-4 space-y-3">
              <p className="text-xs text-slate-500">
                The fixed fusion test cases from the specification. They send detections and probabilities straight to
                /api/fusion/analyze, so the rule can be demonstrated without images.
              </p>
              <select
                value={testCase?.id ?? ''}
                onChange={(e) => {
                  const tc = TEST_CASES.find((t) => t.id === e.target.value) ?? null;
                  setTestCase(tc);
                  setTestResult(null);
                }}
                className="w-full rounded-xl border border-slate-200 dark:border-slate-700 bg-console-bg px-3 py-2 text-sm"
              >
                <option value="">Choose a test case…</option>
                {TEST_CASES.map((t) => <option key={t.id} value={t.id}>{t.id} · {t.title}</option>)}
              </select>
              {testCase && (
                <>
                  <p className="text-xs font-mono rounded-lg bg-console-bg border border-console-line p-2">
                    {describeRequest(testCase.request)}
                  </p>
                  <button
                    onClick={() => runTestCase(testCase)}
                    disabled={loading}
                    className="w-full py-2.5 rounded-xl font-semibold text-sm bg-slate-800 hover:bg-slate-700 text-white disabled:opacity-50"
                  >
                    Run {testCase.id}
                  </button>
                </>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Results */}
        <div className="space-y-4">
          {!analysis && !testResult && !loading && (
            <Card className="border-console-line min-h-[300px] flex items-center justify-center">
              <CardContent className="text-center py-12 text-slate-500">
                Analyze an incident or run a test case to see the fused decision.
              </CardContent>
            </Card>
          )}
          {loading && (
            <Card className="border-console-line min-h-[300px] flex items-center justify-center">
              <CardContent className="flex flex-col items-center py-12">
                <Loader2 size={36} className="text-emerald-500 animate-spin mb-3" />
                <p className="text-slate-500">Running models…</p>
              </CardContent>
            </Card>
          )}

          {testResult && testCase && !loading && <FusionResultCard fusion={testResult} expected={testCase.expected} />}

          {analysis && !loading && (
            <>
              <FusionResultCard fusion={analysis.fusion} />
              {analysis.vision && (
                <Card className="border-console-line shadow-sm overflow-hidden">
                  <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
                    <CardTitle className="text-sm flex items-center justify-between">
                      <span>Vision: {analysis.vision.detections.length} detections · S_v {analysis.vision.vision_score.toFixed(3)}</span>
                      <DataOriginBadge origin={analysis.vision.data_origin} />
                    </CardTitle>
                  </CardHeader>
                  {analysis.vision.annotated_image_url && (
                    <CardContent className="p-0">
                      <img src={serverUrl(analysis.vision.annotated_image_url)} alt="Annotated" className="w-full object-contain max-h-80" />
                    </CardContent>
                  )}
                </Card>
              )}
              {analysis.vision && <LlmSecondOpinion file={image} yoloCounts={analysis.vision.counts} />}
              {analysis.nlp && (
                <Card className="border-console-line shadow-sm">
                  <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
                    <CardTitle className="text-sm flex items-center justify-between">
                      <span>Text: {analysis.nlp.urgency} urgency · S_n {analysis.nlp.nlp_score.toFixed(3)}</span>
                      <DataOriginBadge origin={analysis.nlp.data_origin} />
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-4">
                    <UrgencyBars probabilities={analysis.nlp.probabilities} />
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
