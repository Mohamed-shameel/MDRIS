import type { RiskBand } from '../services/api';

export const RISK_COLORS: Record<RiskBand, string> = {
  Critical: '#DC2626',
  High: '#EA580C',
  Moderate: '#CA8A04',
  Low: '#16A34A',
};

export const RISK_ORDER: RiskBand[] = ['Critical', 'High', 'Moderate', 'Low'];

/** Risk band as text plus colour (never colour alone). */
export default function RiskBadge({ band, size = 'sm' }: { band: RiskBand; size?: 'sm' | 'lg' }) {
  const color = RISK_COLORS[band];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-bold ${size === 'lg' ? 'px-3 py-1 text-sm' : 'px-2 py-0.5 text-xs'}`}
      style={{ color, borderColor: `${color}66`, backgroundColor: `${color}1A` }}
    >
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      {band}
    </span>
  );
}
