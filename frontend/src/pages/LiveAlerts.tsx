import { useEffect, useState } from 'react';
import { Card, CardContent } from '../components/ui/card';
import { Bell, Clock, MapPin } from 'lucide-react';
import { fetchAlerts, type Incident, type IncidentList } from '../services/api';
import SimulationBanner from '../components/SimulationBanner';
import DataOriginBadge from '../components/DataOriginBadge';
import RiskBadge, { RISK_COLORS } from '../components/RiskBadge';
import EvidenceMeter from '../components/EvidenceMeter';
import ConflictFlag from '../components/ConflictFlag';
import SingleSourceChip from '../components/SingleSourceChip';
import IncidentDrawer from '../components/IncidentDrawer';
import LoadError, { errorDetail } from '../components/LoadError';

export default function LiveAlerts() {
  const [data, setData]         = useState<IncidentList | null>(null);
  const [error, setError]       = useState<string | null>(null);
  const [selected, setSelected] = useState<Incident | null>(null);

  useEffect(() => {
    fetchAlerts().then((res) => setData(res.data)).catch((err) => setError(errorDetail(err)));
  }, []);

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <SimulationBanner />
      <div className="flex items-center justify-between">
        <div>
          <h2 className="console-title text-[28px] flex items-center gap-3"><Bell /> Alerts</h2>
          <p className="text-slate-500">Simulated incidents in the High or Critical band, newest first.</p>
        </div>
        {data && <DataOriginBadge origin={data.data_origin} />}
      </div>

      {error && <LoadError title="Simulated incidents unavailable" detail={error} />}
      {data && data.incidents.length === 0 && (
        <p className="text-slate-500">No High or Critical incidents in the current simulated set.</p>
      )}

      <div className="space-y-3">
        {data?.incidents.map((inc) => (
          <Card
            key={inc.id}
            onClick={() => setSelected(inc)}
            className="cursor-pointer border-console-line shadow-sm hover:shadow-md transition-shadow border-l-4"
            style={{ borderLeftColor: RISK_COLORS[inc.fusion.risk_band] }}
          >
            <CardContent className="p-4 flex flex-col md:flex-row md:items-center gap-4">
              <div className="flex items-center gap-3 md:w-56 shrink-0">
                <RiskBadge band={inc.fusion.risk_band} />
                <span className="text-xl font-bold">{inc.fusion.priority_score.toFixed(1)}</span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-semibold">{inc.fusion.suggested_action}</p>
                <p className="text-xs text-slate-500 flex flex-wrap items-center gap-3 mt-1">
                  <span className="font-mono">{inc.id}</span>
                  <span className="flex items-center gap-1"><MapPin size={12} /> {inc.location.name}</span>
                  <span className="flex items-center gap-1"><Clock size={12} /> {new Date(inc.created_at).toLocaleTimeString()}</span>
                </p>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                <EvidenceMeter value={inc.fusion.evidence_strength} />
                {inc.fusion.conflict && <ConflictFlag />}
                {inc.fusion.single_source && <SingleSourceChip />}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      <IncidentDrawer incident={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
