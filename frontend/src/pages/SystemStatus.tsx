import { useEffect, useState, type ReactNode } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Cpu, Database, CloudRain, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';
import { fetchSystemStatus, type ModelStatus, type SystemStatusPayload } from '../services/api';
import DataOriginBadge from '../components/DataOriginBadge';
import LoadError, { errorDetail } from '../components/LoadError';

const Row = ({ label, value }: { label: string; value: ReactNode }) => (
  <div className="flex justify-between gap-4 py-1.5 border-b border-slate-100 dark:border-slate-800/60 text-sm">
    <span className="text-slate-500">{label}</span>
    <span className="font-mono text-xs text-right break-all">{value}</span>
  </div>
);

function Loaded({ ok }: { ok: boolean }) {
  return ok ? (
    <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-semibold"><CheckCircle2 size={14} /> loaded</span>
  ) : (
    <span className="inline-flex items-center gap-1 text-red-600 dark:text-red-400 font-semibold"><XCircle size={14} /> not loaded</span>
  );
}

function ModelCard({ title, status }: { title: string; status: ModelStatus }) {
  return (
    <Card className="border-console-line shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-base">
          <span className="flex items-center gap-2"><Cpu size={16} /> {title}</span>
          <Loaded ok={status.loaded} />
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Row label="Model" value={status.name} />
        <Row label="Checkpoint" value={status.checkpoint} />
        <Row label="Device" value={status.device ?? '—'} />
        <Row label="Trained at" value={status.trained_at ? new Date(status.trained_at).toLocaleString() : '—'} />
        <Row label="Headline test metric" value={status.headline_metric ? `${status.headline_metric.name} = ${status.headline_metric.value.toFixed(3)}` : '—'} />
        {status.error && <p className="mt-2 text-xs text-red-500 font-mono break-all">{status.error}</p>}
      </CardContent>
    </Card>
  );
}

export default function SystemStatus() {
  const [data, setData] = useState<SystemStatusPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setError(null);
    fetchSystemStatus().then((res) => setData(res.data)).catch((err) => setError(errorDetail(err)));
  };
  useEffect(load, []);

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="console-title text-[28px]">System Status</h2>
          <p className="text-slate-500">Read live from the running backend. Nothing here is estimated.</p>
        </div>
        <div className="flex items-center gap-3">
          {data && <DataOriginBadge origin={data.data_origin} />}
          <button onClick={load} className="p-2 rounded-full bg-slate-100 dark:bg-slate-900 hover:bg-slate-200 dark:hover:bg-slate-800" aria-label="Refresh">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {error && <LoadError title="Backend unreachable" detail={error} />}
      {data && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ModelCard title="Vision (YOLOv8n)" status={data.models.vision} />
            <ModelCard title="Urgency NLP (DistilBERT)" status={data.models.nlp} />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="border-console-line shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-base"><Database size={16} /> Simulated incidents</CardTitle>
              </CardHeader>
              <CardContent>
                <Row label="File" value={data.incidents.path} />
                <Row label="Exists" value={data.incidents.exists ? 'yes' : 'no'} />
                <Row label="Generated at" value={data.incidents.generated_at ? new Date(data.incidents.generated_at).toLocaleString() : '—'} />
                <Row label="Incidents" value={data.incidents.count} />
                <Row label="Age" value={data.incidents.age_minutes !== null ? `${data.incidents.age_minutes} min` : '—'} />
              </CardContent>
            </Card>
            <Card className="border-console-line shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-base"><CloudRain size={16} /> Runtime</CardTitle>
              </CardHeader>
              <CardContent>
                <Row label="Inference device" value={data.device} />
                <Row label="Weather API configured" value={data.weather.configured ? 'yes' : 'no'} />
                {Object.entries(data.versions).map(([k, ver]) => <Row key={k} label={k} value={ver} />)}
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
