import { useEffect, useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { fetchIncident, fetchMap, type Incident, type MapPayload, type RiskBand } from '../services/api';
import SimulationBanner from '../components/SimulationBanner';
import DataOriginBadge from '../components/DataOriginBadge';
import { RISK_COLORS, RISK_ORDER } from '../components/RiskBadge';
import IncidentDrawer from '../components/IncidentDrawer';
import LoadError, { errorDetail } from '../components/LoadError';

const CHENNAI: [number, number] = [13.0827, 80.2707];

export default function DisasterMap() {
  const [data, setData]           = useState<MapPayload | null>(null);
  const [error, setError]         = useState<string | null>(null);
  const [bands, setBands]         = useState<Record<RiskBand, boolean>>({ Critical: true, High: true, Moderate: true, Low: true });
  const [conflictsOnly, setConflictsOnly] = useState(false);
  const [selected, setSelected]   = useState<Incident | null>(null);

  useEffect(() => {
    fetchMap().then((res) => setData(res.data)).catch((err) => setError(errorDetail(err)));
  }, []);

  const features = useMemo(
    () => (data?.features ?? []).filter((f) => bands[f.risk_band] && (!conflictsOnly || f.conflict)),
    [data, bands, conflictsOnly]
  );

  const openIncident = (id: string) => {
    fetchIncident(id).then((res) => setSelected(res.data)).catch(() => {});
  };

  return (
    <div className="space-y-4">
      <SimulationBanner />
      <Card className="flex flex-col border-console-line shadow-lg">
        <CardHeader className="py-4 border-b border-slate-100 dark:border-slate-800">
          <CardTitle className="flex flex-wrap items-center justify-between gap-3">
            <span>Incident Map</span>
            <div className="flex flex-wrap items-center gap-3 text-xs font-normal">
              {RISK_ORDER.map((b) => (
                <label key={b} className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={bands[b]}
                    onChange={(e) => setBands({ ...bands, [b]: e.target.checked })}
                  />
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: RISK_COLORS[b] }} />
                  {b}
                </label>
              ))}
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input type="checkbox" checked={conflictsOnly} onChange={(e) => setConflictsOnly(e.target.checked)} />
                Conflicts only
              </label>
              {data && <DataOriginBadge origin={data.data_origin} />}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0 relative z-0 h-[calc(100vh-17rem)] min-h-[420px]">
          {error && <div className="p-4"><LoadError title="Simulated incidents unavailable" detail={error} /></div>}
          {!error && (
            <MapContainer center={CHENNAI} zoom={11} style={{ height: '100%', width: '100%' }}>
              <TileLayer
                url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              {features.map((f) => (
                <CircleMarker
                  key={f.id}
                  center={[f.lat, f.lon]}
                  radius={6 + f.priority_score / 10}
                  pathOptions={{
                    color: RISK_COLORS[f.risk_band],
                    weight: f.conflict ? 3 : 1.5,
                    dashArray: f.conflict ? '4 3' : undefined,
                    fillColor: RISK_COLORS[f.risk_band],
                    fillOpacity: 0.3 + 0.6 * f.evidence_strength,
                  }}
                >
                  <Popup>
                    <div className="space-y-1 text-xs">
                      <p className="font-bold text-sm">{f.id} · {f.location}</p>
                      <p><span className="font-semibold" style={{ color: RISK_COLORS[f.risk_band] }}>{f.risk_band}</span> · score {f.priority_score.toFixed(1)} · evidence {f.evidence_strength.toFixed(2)}</p>
                      {f.conflict && <p className="text-amber-600 font-semibold">Image and text disagree: verify on ground</p>}
                      <p>{f.suggested_action}</p>
                      <p className="text-amber-700">Simulated incident</p>
                      <button
                        onClick={() => openIncident(f.id)}
                        className="mt-1 rounded-md bg-blue-600 px-3 py-1 text-white hover:bg-blue-700"
                      >
                        Open
                      </button>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>
          )}
          {/* Legend */}
          <div className="absolute top-4 right-4 z-[500] rounded-xl border border-slate-200 dark:border-slate-700 bg-white/95 dark:bg-slate-900/95 p-3 text-xs shadow-lg space-y-1">
            {RISK_ORDER.map((b) => (
              <div key={b} className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full" style={{ backgroundColor: RISK_COLORS[b] }} /> {b}
              </div>
            ))}
            <p className="pt-1 text-slate-500">Size = priority · faded = weak evidence</p>
            <p className="text-slate-500">Dashed outline = conflict</p>
          </div>
        </CardContent>
      </Card>
      <IncidentDrawer incident={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
