import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import {
  fetchVisionStatus, predictVision, serverUrl, VISION_SOURCES,
  type ModelStatus, type VisionResult, type VisionSource,
} from '../services/api';
import { toast } from 'sonner';
import { Upload, Image as ImageIcon, Loader2, Eye, Clock, Cpu, X, Crosshair } from 'lucide-react';
import DataOriginBadge from '../components/DataOriginBadge';
import LlmSecondOpinion from '../components/LlmSecondOpinion';
import RiskBadge from '../components/RiskBadge';
import { CLASS_LABELS, SOURCE_LABELS } from '../components/labels';

// ── Helpers ─────────────────────────────────────────────────────────────────

const CLASS_DOT: Record<string, string> = {
  person_in_water: 'bg-red-600',
  person_in_vehicle: 'bg-orange-600',
  person: 'bg-yellow-600',
  vehicle: 'bg-blue-600',
};

const ACCEPTED = ['image/jpeg', 'image/jpg', 'image/png'];

// ── Main Component ───────────────────────────────────────────────────────────

export default function VisionAI() {
  const [preview, setPreview]       = useState<string | null>(null);
  const [file, setFile]             = useState<File | null>(null);
  const [source, setSource]         = useState<VisionSource>('officer_photo');
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading]   = useState(false);
  const [uploadPct, setUploadPct]   = useState(0);
  const [result, setResult]         = useState<VisionResult | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus]         = useState<ModelStatus | null>(null);

  // Label the page with the model the backend actually serves (D-37)
  useEffect(() => { fetchVisionStatus().then((r) => setStatus(r.data)).catch(() => {}); }, []);
  const variant = status ? (status.fine_tuned ? 'fine-tuned' : 'COCO-pretrained') : null;

  // Handle file selection (drag or click)
  const handleFile = (f: File) => {
    if (!ACCEPTED.includes(f.type)) {
      toast.error('Only JPG and PNG images are supported.');
      return;
    }
    setFile(f);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFile(dropped);
  }, []);

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = () => setIsDragging(false);

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadPct(0);
    try {
      const res = await predictVision(file, source, setUploadPct);
      setResult(res.data);
      toast.success('Inference complete');
    } catch {
      // Interceptor already shows toast (503 if the model is not loaded)
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-console-soft border border-console-line">
            <Eye className="text-accent" size={22} />
          </div>
          <h2 className="console-title text-[28px]">Vision (YOLOv8n{variant ? `, ${variant}` : ''})</h2>
        </div>
        <p className="text-slate-500 ml-12">
          {status && !status.fine_tuned
            ? 'COCO-pretrained YOLOv8n (not fine-tuned) detects people and vehicles; COCO person → person, car/motorcycle/bus/truck → vehicle. It cannot tell person in water or person in vehicle apart.'
            : 'Fine-tuned YOLOv8n detects people and vehicles in flood imagery: person in water, person in vehicle, person, vehicle.'}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Panel */}
        <div className="space-y-4">
          {/* Dropzone */}
          <div
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => !preview && inputRef.current?.click()}
            className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 overflow-hidden cursor-pointer
              ${isDragging
                ? 'border-blue-500 bg-blue-500/5 scale-[1.01]'
                : preview
                ? 'border-slate-300 dark:border-slate-700 cursor-default'
                : 'border-slate-300 dark:border-slate-700 hover:border-blue-400'
              }`}
          >
            {preview ? (
              <div className="relative">
                <img src={preview} alt="Preview" className="w-full h-72 object-cover" />
                <button
                  onClick={(e) => { e.stopPropagation(); clearFile(); }}
                  className="absolute top-3 right-3 p-1.5 rounded-full bg-slate-900/70 text-white hover:bg-red-600 transition-colors"
                >
                  <X size={16} />
                </button>
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-4">
                  <p className="text-white text-sm font-medium truncate">{file?.name}</p>
                  <p className="text-slate-300 text-xs">{((file?.size ?? 0) / 1024).toFixed(0)} KB</p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
                <div className="w-16 h-16 rounded-2xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-4">
                  <Upload className="text-slate-400" size={28} />
                </div>
                <p className="text-base font-semibold text-slate-700 dark:text-slate-200 mb-1">
                  Drag & drop a flood image
                </p>
                <p className="text-sm text-slate-500 mb-4">or click to browse</p>
                <span className="px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-xs text-slate-500 font-mono">
                  JPG • PNG • Max 15 MB
                </span>
              </div>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
          />

          {/* Source picker */}
          <label className="block">
            <span className="text-xs font-medium text-slate-500">Image source (sets its reliability in fusion)</span>
            <select
              value={source}
              onChange={(e) => setSource(e.target.value as VisionSource)}
              className="mt-1 w-full rounded-xl border border-slate-200 dark:border-slate-700 bg-console-bg px-3 py-2 text-sm"
            >
              {VISION_SOURCES.map((s) => <option key={s} value={s}>{SOURCE_LABELS[s]}</option>)}
            </select>
          </label>

          {/* Upload progress */}
          {uploading && (
            <div className="space-y-2">
              <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-1.5">
                <div
                  className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${uploadPct}%` }}
                />
              </div>
              <p className="text-xs text-slate-500 text-center">
                {uploadPct < 100 ? `Uploading… ${uploadPct}%` : 'Running inference…'}
              </p>
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            id="vision-upload-btn"
            className="w-full py-3 px-6 rounded-xl font-semibold text-sm transition-all duration-200
              bg-accent hover:bg-accent-dark text-white shadow-md shadow-blue-500/20
              disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none
              flex items-center justify-center gap-2"
          >
            {uploading
              ? <><Loader2 size={18} className="animate-spin" /> Running Inference…</>
              : <><ImageIcon size={18} /> Analyze Image</>
            }
          </button>

          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Cpu size={12} />
            <span>{status ? `${status.name} · ${status.checkpoint}` : 'YOLOv8n'} · detection threshold 0.25</span>
          </div>
        </div>

        {/* Results Panel */}
        <div className="space-y-4">
          {!result && !uploading && (
            <Card className="border-console-line h-full flex items-center justify-center min-h-[300px]">
              <CardContent className="flex flex-col items-center text-center py-12">
                <div className="w-16 h-16 rounded-2xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-4">
                  <Crosshair className="text-slate-300 dark:text-slate-600" size={28} />
                </div>
                <p className="text-slate-500 font-medium">Upload an image to see detections</p>
                <p className="text-slate-400 text-sm mt-1">Results will appear here</p>
              </CardContent>
            </Card>
          )}

          {uploading && (
            <Card className="border-console-line h-full flex items-center justify-center min-h-[300px]">
              <CardContent className="flex flex-col items-center py-12">
                <Loader2 size={40} className="text-blue-500 animate-spin mb-4" />
                <p className="text-slate-600 dark:text-slate-300 font-semibold">Running YOLOv8n…</p>
              </CardContent>
            </Card>
          )}

          {result && !uploading && (
            <>
              <Card className="border-console-line shadow-md">
                <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
                  <CardTitle className="flex items-center justify-between text-base">
                    <span>Vision score</span>
                    <DataOriginBadge origin={result.data_origin} />
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-4 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-3xl font-bold">{result.vision_score.toFixed(3)}</p>
                      <p className="text-xs text-slate-500">S_v = 1 − Π(1 − w_c · conf) over detections</p>
                    </div>
                    <div className="text-right space-y-1">
                      <RiskBadge band={result.severity_band} size="lg" />
                      <p className="text-xs text-slate-500">vision-only band</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 pt-2">
                    <div className="p-3 rounded-xl bg-console-bg border border-console-line">
                      <div className="flex items-center gap-1.5 text-slate-500 text-xs mb-1">
                        <Clock size={12} /> Inference time
                      </div>
                      <p className="font-bold text-sm">{result.inference_ms.toFixed(0)} ms</p>
                    </div>
                    <div className="p-3 rounded-xl bg-console-bg border border-console-line">
                      <div className="flex items-center gap-1.5 text-slate-500 text-xs mb-1">
                        <Cpu size={12} /> Model
                      </div>
                      <p className="font-bold text-sm truncate" title={result.model.checkpoint}>{result.model.name}</p>
                    </div>
                  </div>
                  <p className="text-xs text-slate-500">
                    Confidence used by fusion: {result.vision_confidence.toFixed(3)}
                    {result.detections.length === 0 && ' (no detections: fixed value 0.50)'} · Source: {SOURCE_LABELS[result.source]}
                  </p>
                </CardContent>
              </Card>

              <Card className="border-console-line shadow-sm">
                <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
                  <CardTitle className="text-sm">Detections ({result.detections.length})</CardTitle>
                </CardHeader>
                <CardContent className="pt-4">
                  {result.detections.length === 0 ? (
                    <p className="text-sm text-slate-500">No objects detected at confidence ≥ 0.25.</p>
                  ) : (
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-xs text-slate-500">
                          <th className="pb-2">Class</th>
                          <th className="pb-2">Confidence</th>
                          <th className="pb-2">Box (x1, y1, x2, y2)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.detections.map((d, i) => (
                          <tr key={i} className="border-t border-slate-100 dark:border-slate-800/60">
                            <td className="py-1.5 flex items-center gap-2">
                              <span className={`w-2 h-2 rounded-full ${CLASS_DOT[d.class_name]}`} />
                              {CLASS_LABELS[d.class_name]}
                            </td>
                            <td className="py-1.5 font-mono">{(d.confidence * 100).toFixed(1)}%</td>
                            <td className="py-1.5 font-mono text-xs text-slate-500">
                              {d.bbox.x1}, {d.bbox.y1}, {d.bbox.x2}, {d.bbox.y2}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </CardContent>
              </Card>

              <LlmSecondOpinion file={file} yoloCounts={result.counts} />

              {result.annotated_image_url && (
                <Card className="border-console-line shadow-sm overflow-hidden">
                  <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
                    <CardTitle className="text-sm">Annotated output</CardTitle>
                  </CardHeader>
                  <CardContent className="p-0">
                    <img src={serverUrl(result.annotated_image_url)} alt="Annotated" className="w-full object-contain max-h-96" />
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
