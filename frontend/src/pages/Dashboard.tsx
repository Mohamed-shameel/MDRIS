import { useEffect, useState, type ElementType } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Activity, AlertTriangle, CloudRain, Flame, Gauge, ListOrdered } from 'lucide-react';
import {
  fetchDashboard, fetchIncidents, fetchWeather,
  type DashboardPayload, type Incident, type IncidentList, type WeatherNotConfigured, type WeatherResult,
} from '../services/api';
import SimulationBanner from '../components/SimulationBanner';
import DataOriginBadge from '../components/DataOriginBadge';
import RiskBadge, { RISK_COLORS, RISK_ORDER } from '../components/RiskBadge';
import EvidenceMeter from '../components/EvidenceMeter';
import ConflictFlag from '../components/ConflictFlag';
import SingleSourceChip from '../components/SingleSourceChip';
import IncidentDrawer from '../components/IncidentDrawer';
import LoadError, { errorDetail } from '../components/LoadError';
import { CLASS_LABELS } from '../components/labels';

const CHENNAI = { lat: 13.0827, lon: 80.2707 };

function StatCard({ title, value, icon: Icon, color, note }: { title: string; value: string | number; icon: ElementType; color: string; note: string }) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <Icon size={20} className={`mt-0.5 ${color}`} />
          <div>
            <p className="console-label">{title}</p>
            <p className="font-display font-bold text-[34px] leading-tight text-navy-900">{value}</p>
          </div>
        </div>
        <p className="mt-2 rounded-md bg-console-bg px-3 py-1.5 text-[12px] text-console-label">{note}</p>
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const [dash, setDash]           = useState<DashboardPayload | null>(null);
  const [list, setList]           = useState<IncidentList | null>(null);
  const [error, setError]         = useState<string | null>(null);
  const [weather, setWeather]     = useState<WeatherResult | WeatherNotConfigured | null>(null);
  const [weatherErr, setWeatherErr] = useState<string | null>(null);
  const [selected, setSelected]   = useState<Incident | null>(null);

  useEffect(() => {
    Promise.all([fetchDashboard(), fetchIncidents()])
      .then(([d, l]) => { setDash(d.data); setList(l.data); })
      .catch((err) => setError(errorDetail(err)));
    fetchWeather(CHENNAI.lat, CHENNAI.lon)
      .then((res) => setWeather(res.data))
      .catch((err) => setWeatherErr(errorDetail(err)));
  }, []);

  return (
    <div className="space-y-6">
      <SimulationBanner />
      <div className="flex items-center justify-between">
        <div>
          <h2 className="console-title text-[28px]">Command Dashboard</h2>
          <p className="text-slate-500">Which incident needs a team first: fused priority from YOLO and the text model.</p>
        </div>
        {dash && <DataOriginBadge origin={dash.data_origin} />}
      </div>

      {error && <LoadError title="Simulated incidents unavailable" detail={error} />}

      {dash && (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
          <StatCard title="Total incidents" value={dash.kpis.total_incidents} icon={Activity} color="text-accent" note="simulated set" />
          <StatCard title="Critical" value={dash.kpis.critical} icon={Flame} color="text-red-600" note="score ≥ 75" />
          <StatCard title="High" value={dash.kpis.high} icon={AlertTriangle} color="text-orange-600" note="score 55–75" />
          <StatCard title="Conflicts to verify" value={dash.kpis.conflicts} icon={AlertTriangle} color="text-amber-500" note="image and text disagree" />
          <StatCard title="Mean evidence" value={dash.kpis.mean_evidence_strength.toFixed(2)} icon={Gauge} color="text-indiagreen" note="0–1, confidence × reliability" />
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Priority queue */}
        <Card className="xl:col-span-3 border-console-line shadow-md">
          <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
            <CardTitle className="flex items-center justify-between text-base">
              <span className="flex items-center gap-2"><ListOrdered size={18} /> Priority queue</span>
              {list && <DataOriginBadge origin={list.data_origin} />}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 overflow-x-auto">
            {!list && !error && <div className="h-64 animate-pulse bg-slate-100 dark:bg-slate-900" />}
            {list && (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-slate-500 border-b border-slate-100 dark:border-slate-800">
                    <th className="px-4 py-2">#</th>
                    <th className="px-2 py-2">ID</th>
                    <th className="px-2 py-2">Location</th>
                    <th className="px-2 py-2">Band</th>
                    <th className="px-2 py-2">Score</th>
                    <th className="px-2 py-2">Evidence</th>
                    <th className="px-2 py-2">Dominant</th>
                    <th className="px-2 py-2">Action</th>
                    <th className="px-2 py-2">Flags</th>
                  </tr>
                </thead>
                <tbody>
                  {list.incidents.map((inc, i) => (
                    <tr
                      key={inc.id}
                      onClick={() => setSelected(inc)}
                      className="border-b border-slate-100 dark:border-slate-800/60 hover:bg-slate-50 dark:hover:bg-slate-900/60 cursor-pointer"
                    >
                      <td className="px-4 py-2 font-mono text-slate-500">{i + 1}</td>
                      <td className="px-2 py-2 font-mono text-xs">{inc.id}</td>
                      <td className="px-2 py-2 whitespace-nowrap">{inc.location.name}</td>
                      <td className="px-2 py-2"><RiskBadge band={inc.fusion.risk_band} /></td>
                      <td className="px-2 py-2 font-bold">{inc.fusion.priority_score.toFixed(1)}</td>
                      <td className="px-2 py-2"><EvidenceMeter value={inc.fusion.evidence_strength} showValue={false} /></td>
                      <td className="px-2 py-2 whitespace-nowrap text-xs">{CLASS_LABELS[inc.fusion.dominant_class]}</td>
                      <td className="px-2 py-2 max-w-[260px] truncate text-xs text-slate-600 dark:text-slate-300" title={inc.fusion.suggested_action}>
                        {inc.fusion.suggested_action}
                      </td>
                      <td className="px-2 py-2">
                        <div className="flex items-center gap-1.5">
                          {inc.fusion.conflict && <ConflictFlag />}
                          {inc.fusion.single_source && <SingleSourceChip />}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>

        {/* Side column */}
        <div className="space-y-6">
          <Card className="border-console-line shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2"><CloudRain size={16} /> Chennai weather</span>
                {weather && <DataOriginBadge origin={weather.data_origin} />}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p className="text-[11px] font-semibold text-cyan-600 dark:text-cyan-400">Context only: not used in priority score</p>
              {weatherErr && <p className="text-xs text-red-500">{weatherErr}</p>}
              {weather && !weather.configured && <p className="text-slate-500">Weather API key not configured</p>}
              {weather && weather.configured && (
                <div className="space-y-1">
                  <p className="font-semibold">{weather.weather_description}</p>
                  <p className="text-slate-500">{weather.temperature_c.toFixed(1)} °C · humidity {weather.humidity_pct.toFixed(0)}%</p>
                  <p className="text-slate-500">Rain (1 h): {weather.rainfall_mm.toFixed(1)} mm · wind {weather.wind_speed_kmh.toFixed(0)} km/h</p>
                </div>
              )}
            </CardContent>
          </Card>

          {dash && (
            <Card className="border-console-line shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center justify-between text-sm">
                  <span>Risk bands</span>
                  <DataOriginBadge origin={dash.data_origin} />
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {RISK_ORDER.map((band) => {
                  const n = dash.band_counts[band] ?? 0;
                  const pct = dash.kpis.total_incidents ? (100 * n) / dash.kpis.total_incidents : 0;
                  return (
                    <div key={band} className="flex items-center gap-2 text-xs">
                      <span className="w-16 font-semibold" style={{ color: RISK_COLORS[band] }}>{band}</span>
                      <div className="flex-1 h-2 rounded-full bg-slate-100 dark:bg-slate-800">
                        <div className="h-2 rounded-full" style={{ width: `${pct}%`, backgroundColor: RISK_COLORS[band] }} />
                      </div>
                      <span className="w-6 text-right font-mono">{n}</span>
                    </div>
                  );
                })}
                <p className="text-[11px] text-slate-400 pt-1">Generated {new Date(dash.generated_at).toLocaleString()}</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      <IncidentDrawer incident={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
