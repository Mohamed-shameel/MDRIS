/**
 * MDRIS API Service Layer
 * ========================
 * Central Axios client plus the types of docs/MDRIS_CONTEXT.md section 9.4.
 * Every backend payload carries `data_origin`, shown by <DataOriginBadge>.
 */

import axios, { type AxiosProgressEvent } from 'axios';
import { toast } from 'sonner';

export const SERVER_URL = 'http://localhost:8000';
const API_URL = `${SERVER_URL}/api`;

/** Absolute URL for a backend-relative path such as /uploads/x.jpg or /figures/vision/y.png */
export const serverUrl = (path: string | null | undefined) => (path ? `${SERVER_URL}${path}` : '');

// ── Axios instance ─────────────────────────────────────────────────────────

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60s — model inference can take time on first call
});

// ── Response interceptor ────────────────────────────────────────────────────

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Network errors (server not running, etc.)
    if (!error.response) {
      toast.error('Cannot reach MDRIS server. Is the backend running?', {
        id: 'network-error',
      });
      return Promise.reject(error);
    }

    const status = error.response?.status;
    const detail = error.response?.data?.detail ?? 'Unknown error';
    const message = typeof detail === 'string' ? detail : JSON.stringify(detail);

    if (status === 422) {
      toast.error(`Validation error: ${message}`);
    } else if (status === 503) {
      toast.error(`Model unavailable: ${message}`, { id: `503-${message}` });
    } else if (status === 502) {
      toast.error(message);
    } else if (status === 500) {
      toast.error('Server error. Please try again.');
    }

    return Promise.reject(error);
  }
);

// ── Shared types ────────────────────────────────────────────────────────────

export type DataOrigin = 'model' | 'simulated' | 'static_demo' | 'live' | 'external_llm';
export type RiskBand = 'Critical' | 'High' | 'Moderate' | 'Low';
export type UrgencyLabel = 'Low' | 'Medium' | 'High';
export type EvidenceLevel = 'Strong' | 'Fair' | 'Weak';
export type VisionClass = 'person_in_water' | 'person_in_vehicle' | 'person' | 'vehicle';
export type DominantClass = VisionClass | 'text_only';

export type VisionSource = 'officer_photo' | 'drone' | 'cctv' | 'citizen_upload';
export type TextSource = 'officer_report' | 'control_room_call' | 'citizen_sms' | 'social_media';

export const VISION_SOURCES: VisionSource[] = ['officer_photo', 'drone', 'cctv', 'citizen_upload'];
export const TEXT_SOURCES: TextSource[] = ['officer_report', 'control_room_call', 'citizen_sms', 'social_media'];

// ── Vision ──────────────────────────────────────────────────────────────────

export interface Detection {
  class_name: VisionClass;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

export interface VisionResult {
  data_origin: DataOrigin;
  model: { name: string; checkpoint: string; trained_at: string | null; fine_tuned: boolean };
  source: VisionSource;
  detections: Detection[];
  counts: Record<VisionClass, number>;
  vision_score: number;
  vision_confidence: number;
  severity_band: RiskBand;
  annotated_image_url: string | null;
  inference_ms: number;
}

export interface ModelStatus {
  data_origin: DataOrigin;
  name: string;
  loaded: boolean;
  error: string | null;
  checkpoint: string;
  device: string | null;
  trained_at: string | null;
  fine_tuned: boolean;
  headline_metric: { name: string; value: number } | null;
}

export const predictVision = (
  file: File,
  source: VisionSource,
  onProgress?: (percent: number) => void
) => {
  const form = new FormData();
  form.append('file', file);
  form.append('source', source);

  return apiClient.post<VisionResult>('/vision/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
    onUploadProgress: (evt: AxiosProgressEvent) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded / evt.total) * 100));
      }
    },
  });
};

export const fetchVisionStatus = () => apiClient.get<ModelStatus>('/vision/status');

/** LLM second opinion (D-36): counts + descriptions only, never used by fusion. */
export interface LlmVisionResult {
  data_origin: 'external_llm';
  model: { provider: string; name: string };
  used_in_fusion: false;
  flood_visible: boolean;
  objects: { class_name: VisionClass; description: string }[];
  counts: Record<VisionClass, number>;
  summary: string;
  latency_ms: number;
}

export const llmVisionCheck = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return apiClient.post<LlmVisionResult>('/vision/llm-check', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000,
  });
};

// ── NLP ─────────────────────────────────────────────────────────────────────

export interface NlpResult {
  data_origin: DataOrigin;
  model: { name: string; path: string; trained_at: string | null };
  source: TextSource;
  urgency: UrgencyLabel;
  probabilities: Record<UrgencyLabel, number>;
  confidence: number;
  nlp_score: number;
  inference_ms: number;
}

export interface NlpLabels {
  data_origin: DataOrigin;
  labels: UrgencyLabel[];
  urgency_values: Record<UrgencyLabel, number>;
  humaid_mapping: Array<{ humaid_label: string; urgency: UrgencyLabel }>;
}

export const analyzeNLP = (text: string, source: TextSource) =>
  apiClient.post<NlpResult>('/nlp/analyze', { text, source });

export const fetchNLPLabels = () => apiClient.get<NlpLabels>('/nlp/labels');

// ── Fusion ──────────────────────────────────────────────────────────────────

export interface FusionBreakdownData {
  S_v: number | null;
  conf_v: number | null;
  reliability_v: number | null;
  eff_v: number;
  S_n: number | null;
  conf_n: number | null;
  reliability_n: number | null;
  eff_n: number;
  weights: { vision: number; nlp: number };
}

export interface FusionResult {
  data_origin: DataOrigin;
  priority_score: number;
  risk_band: RiskBand;
  evidence_strength: number;
  evidence_level: EvidenceLevel;
  conflict: boolean;
  single_source: boolean;
  dominant_class: DominantClass;
  suggested_action: string;
  breakdown: FusionBreakdownData;
}

export interface FusionRequest {
  vision: { detections: Array<{ class_name: VisionClass; confidence: number }>; source: VisionSource } | null;
  nlp: { probabilities: Record<UrgencyLabel, number>; source: TextSource } | null;
}

export const analyzeFusion = (request: FusionRequest) =>
  apiClient.post<FusionResult>('/fusion/analyze', request);

export interface IncidentAnalysis {
  data_origin: DataOrigin;
  vision: VisionResult | null;
  nlp: NlpResult | null;
  fusion: FusionResult;
}

export const analyzeIncident = (params: {
  image: File | null;
  text: string;
  imageSource: VisionSource;
  textSource: TextSource;
}) => {
  const form = new FormData();
  if (params.image) form.append('image', params.image);
  if (params.text.trim()) form.append('text', params.text.trim());
  form.append('image_source', params.imageSource);
  form.append('text_source', params.textSource);
  return apiClient.post<IncidentAnalysis>('/incident/analyze', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
};

// ── Simulated incidents ─────────────────────────────────────────────────────

export interface Incident {
  id: string;
  simulated: true;
  created_at: string;
  location: { name: string; district: string; lat: number; lon: number };
  image: { url: string; source: VisionSource; dataset_file: string } | null;
  text: { content: string; source: TextSource; humaid_label: string; event: string } | null;
  vision: VisionResult | null;
  nlp: NlpResult | null;
  fusion: FusionResult;
}

export interface IncidentList {
  data_origin: DataOrigin;
  generated_at: string;
  count: number;
  incidents: Incident[];
}

export interface DashboardPayload {
  data_origin: DataOrigin;
  generated_at: string;
  kpis: {
    total_incidents: number;
    critical: number;
    high: number;
    conflicts: number;
    mean_evidence_strength: number;
  };
  band_counts: Record<RiskBand, number>;
}

export interface MapPayload {
  data_origin: DataOrigin;
  features: Array<{
    id: string;
    lat: number;
    lon: number;
    location: string;
    risk_band: RiskBand;
    priority_score: number;
    evidence_strength: number;
    conflict: boolean;
    suggested_action: string;
  }>;
}

export const fetchIncidents = () => apiClient.get<IncidentList>('/incidents');
export const fetchIncident = (id: string) =>
  apiClient.get<Incident & { data_origin: DataOrigin }>(`/incidents/${id}`);
export const fetchDashboard = () => apiClient.get<DashboardPayload>('/dashboard');
export const fetchMap = () => apiClient.get<MapPayload>('/map');
export const fetchAlerts = () => apiClient.get<IncidentList>('/alerts');

// ── Static demo / evaluation / status ──────────────────────────────────────

export interface ResourceItem {
  id: string;
  type: string;
  total: number;
  available: number;
  assigned_area: string;
  status: string;
}

export const fetchResources = () =>
  apiClient.get<{ data_origin: DataOrigin; note: string; resources: ResourceItem[] }>('/resources');

export interface PerClassDetection { precision: number | null; recall: number | null; AP50: number | null; instances?: number }
export interface PerClassText { precision: number; recall: number; f1: number; support: number }

export interface VisionMetrics {
  model_name: string;
  trained_at: string;
  hardware: string;
  train_time_minutes: number;
  dataset: { name: string; url: string; licence: string; splits: Record<string, number> };
  classes: string[];
  hyperparameters: Record<string, number | string>;
  metrics: { test: { mAP50: number; mAP50_95: number; per_class: Record<string, PerClassDetection> } };
  baselines: Record<string, unknown>;
  figures: string[];
  notes?: string[];
}

export interface TextBaseline {
  name: string;
  accuracy: number;
  macro_f1: number;
  weighted_f1?: number;
  per_class_f1: Record<string, number>;
  model?: string;
}

export interface NlpMetrics {
  model_name: string;
  trained_at: string;
  hardware: string;
  train_time_minutes: number;
  dataset: { name: string; url: string; licence: string; splits: Record<string, number> };
  classes: string[];
  hyperparameters: Record<string, number | string>;
  metrics: {
    test: {
      accuracy: number;
      macro_f1: number;
      weighted_f1: number;
      per_class: Record<string, PerClassText>;
      confusion_matrix: number[][];
      per_event_macro_f1: Record<string, number>;
    };
  };
  baselines: Record<string, TextBaseline>;
  figures: string[];
  notes?: string[];
}

export interface AblationRow {
  variant: string;
  description: string;
  band_changes: number;
  kendall_tau: number | null;
  kendall_tau_top10: number | null;
  mean_score_shift: number;
}

export interface SensitivityRow {
  setting: string;
  kendall_tau_top10: number | null;
  kendall_tau_all: number | null;
  band_changes: number;
}

export interface FusionAnalysis {
  generated_at: string;
  n_incidents: number;
  ablation: AblationRow[];
  sensitivity: { vision_weight: SensitivityRow[]; reliability_scale: SensitivityRow[] };
  notes?: string[];
}

export interface ModelEvaluationBlock<T> {
  available: boolean;
  reason: string | null;
  metrics: T | null;
  figure_urls: Record<string, string>;
}

export interface EvaluationPayload {
  data_origin: DataOrigin;
  vision: ModelEvaluationBlock<VisionMetrics>;
  nlp: ModelEvaluationBlock<NlpMetrics>;
  fusion: { available: boolean; reason: string | null; analysis: FusionAnalysis | null };
}

export const fetchEvaluation = () => apiClient.get<EvaluationPayload>('/evaluation');

export interface SystemStatusPayload {
  data_origin: DataOrigin;
  models: { vision: ModelStatus; nlp: ModelStatus };
  device: string;
  versions: Record<string, string>;
  incidents: { path: string; exists: boolean; generated_at: string | null; count: number; age_minutes: number | null };
  weather: { configured: boolean };
}

export const fetchSystemStatus = () => apiClient.get<SystemStatusPayload>('/system-status');

// ── Weather (context only) ──────────────────────────────────────────────────

export interface WeatherResult {
  data_origin: DataOrigin;
  configured: true;
  location_name: string;
  latitude: number;
  longitude: number;
  temperature_c: number;
  feels_like_c: number;
  humidity_pct: number;
  wind_speed_kmh: number;
  rainfall_mm: number;
  weather_description: string;
  risk_level: string;
  risk_score: number;
}

export interface WeatherNotConfigured {
  data_origin: DataOrigin;
  configured: false;
  detail: string;
}

export const fetchWeather = (lat: number, lon: number) =>
  apiClient.get<WeatherResult | WeatherNotConfigured>('/weather/', { params: { lat, lon } });

export const fetchWeatherPresets = () =>
  apiClient.get<{ data_origin: DataOrigin; presets: Record<string, { lat: number; lon: number; name: string }> }>(
    '/weather/presets'
  );

// ── Default export ──────────────────────────────────────────────────────────

export default apiClient;
