import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { fetchWeather, fetchWeatherPresets, type WeatherNotConfigured, type WeatherResult } from '../services/api';
import DataOriginBadge from '../components/DataOriginBadge';
import { toast } from 'sonner';
import {
  Cloud, Loader2, AlertTriangle, Thermometer, Wind,
  Droplets, CloudRain, MapPin, RefreshCw
} from 'lucide-react';

// ── Preset location type ─────────────────────────────────────────────────────
interface Preset {
  lat: number;
  lon: number;
  name: string;
}

// ── Risk styling ─────────────────────────────────────────────────────────────
const RISK_STYLES: Record<string, { card: string; badge: string; bar: string }> = {
  Critical: {
    card:  'border-red-300 dark:border-red-800 bg-red-50/40 dark:bg-red-900/10',
    badge: 'bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800',
    bar:   'bg-red-500',
  },
  High: {
    card:  'border-orange-300 dark:border-orange-800 bg-orange-50/40 dark:bg-orange-900/10',
    badge: 'bg-orange-100 dark:bg-orange-900/50 text-orange-700 dark:text-orange-400 border-orange-200 dark:border-orange-800',
    bar:   'bg-orange-500',
  },
  Moderate: {
    card:  'border-yellow-300 dark:border-yellow-800 bg-yellow-50/40 dark:bg-yellow-900/10',
    badge: 'bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-400 border-yellow-200 dark:border-yellow-800',
    bar:   'bg-yellow-500',
  },
  Low: {
    card:  'border-blue-300 dark:border-blue-800 bg-blue-50/40 dark:bg-blue-900/10',
    badge: 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800',
    bar:   'bg-blue-500',
  },
  None: {
    card:  'border-emerald-300 dark:border-emerald-800 bg-emerald-50/40 dark:bg-emerald-900/10',
    badge: 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800',
    bar:   'bg-emerald-500',
  },
  Unknown: {
    card:  'border-slate-200 dark:border-slate-700',
    badge: 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-700',
    bar:   'bg-slate-400',
  },
};

const getRisk = (r: string) => RISK_STYLES[r] ?? RISK_STYLES.Unknown;

// ── Metric Card ──────────────────────────────────────────────────────────────
function MetricCard({
  icon: Icon, iconColor, label, value, unit, sub,
}: {
  icon: React.ElementType; iconColor: string;
  label: string; value: string | number; unit: string; sub?: string;
}) {
  return (
    <Card className="border-console-line shadow-sm">
      <CardContent className="pt-5 pb-4">
        <div className="flex items-start justify-between mb-3">
          <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">{label}</p>
          <div className={`p-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 ${iconColor}`}>
            <Icon size={14} />
          </div>
        </div>
        <p className="text-3xl font-bold">
          {value}<span className="text-base font-normal text-slate-500 ml-1">{unit}</span>
        </p>
        {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
      </CardContent>
    </Card>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────
export default function WeatherIntel() {
  const [lat, setLat]         = useState('13.0827');
  const [lon, setLon]         = useState('80.2707');
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState<WeatherResult | WeatherNotConfigured | null>(null);
  const [presets, setPresets] = useState<Record<string, Preset>>({});

  // Load presets on mount
  useEffect(() => {
    fetchWeatherPresets()
      .then(r => setPresets(r.data.presets))
      .catch(() => {});
  }, []);

  const handleFetch = async () => {
    const latNum = parseFloat(lat);
    const lonNum = parseFloat(lon);
    if (isNaN(latNum) || isNaN(lonNum)) {
      toast.error('Please enter valid latitude and longitude.');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const res = await fetchWeather(latNum, lonNum);
      setResult(res.data);
      if (!res.data.configured) {
        toast.warning(res.data.detail);
      } else {
        toast.success(`Weather data loaded for ${res.data.location_name}`);
      }
    } catch {
      // Interceptor handles toast
    } finally {
      setLoading(false);
    }
  };

  const handlePreset = (key: string) => {
    const p = presets[key];
    if (p) {
      setLat(String(p.lat));
      setLon(String(p.lon));
    }
  };

  const riskStyle = result && result.configured ? getRisk(result.risk_level) : getRisk('Unknown');

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-console-soft border border-console-line">
            <Cloud className="text-accent" size={22} />
          </div>
          <h2 className="console-title text-[28px]">Weather (context only)</h2>
        </div>
        <p className="text-slate-500 ml-12">
          Current conditions from OpenWeatherMap, shown for situational awareness.
        </p>
        <div className="ml-12 mt-3 inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold text-cyan-700 dark:text-cyan-300">
          Context only: not used in priority score
        </div>
      </div>

      {/* Location Input Card */}
      <Card className="border-console-line shadow-md">
        <CardHeader className="pb-3 border-b border-slate-100 dark:border-slate-800/60">
          <CardTitle className="flex items-center gap-2 text-base">
            <MapPin size={18} className="text-cyan-500" />
            Location
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-4 space-y-4">
          {/* Coordinates */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-medium text-slate-500 mb-1 block">Latitude</label>
              <input
                id="weather-lat-input"
                type="number"
                value={lat}
                onChange={(e) => setLat(e.target.value)}
                step="0.0001"
                className="w-full px-3 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700
                  bg-console-bg text-sm
                  focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-slate-500 mb-1 block">Longitude</label>
              <input
                id="weather-lon-input"
                type="number"
                value={lon}
                onChange={(e) => setLon(e.target.value)}
                step="0.0001"
                className="w-full px-3 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700
                  bg-console-bg text-sm
                  focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all"
              />
            </div>
          </div>

          {/* Preset buttons */}
          {Object.keys(presets).length > 0 && (
            <div>
              <p className="text-xs text-slate-400 mb-2 font-medium">Quick select:</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(presets).map(([key, p]) => (
                  <button
                    key={key}
                    onClick={() => handlePreset(key)}
                    className="text-xs px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-700
                      text-slate-600 dark:text-slate-300 hover:border-blue-400 hover:text-cyan-600
                      dark:hover:text-cyan-400 transition-colors"
                  >
                    {p.name}
                  </button>
                ))}
              </div>
            </div>
          )}

          <button
            id="weather-fetch-btn"
            onClick={handleFetch}
            disabled={loading}
            className="w-full py-3 rounded-xl font-semibold text-sm
              bg-accent hover:bg-accent-dark text-white shadow-md shadow-blue-500/20
              disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none
              flex items-center justify-center gap-2 transition-all duration-200"
          >
            {loading
              ? <><Loader2 size={16} className="animate-spin" /> Fetching…</>
              : <><RefreshCw size={16} /> Fetch Weather Data</>
            }
          </button>
        </CardContent>
      </Card>

      {/* Not configured */}
      {result && !result.configured && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20">
          <AlertTriangle className="text-amber-500 mt-0.5 shrink-0" size={16} />
          <div>
            <p className="text-sm font-semibold text-amber-700 dark:text-amber-400 mb-0.5">Weather API key not configured</p>
            <p className="text-sm text-amber-600 dark:text-amber-500">{result.detail}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && result.configured && (
        <div className="space-y-4">

          {/* Risk Level Banner */}
          <Card className={`shadow-md border ${riskStyle.card}`}>
            <CardContent className="pt-5 pb-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">
                    {result.location_name} · {result.latitude.toFixed(2)}°N, {result.longitude.toFixed(2)}°E
                  </p>
                  <p className="text-lg font-semibold text-slate-700 dark:text-slate-200 capitalize">
                    {result.weather_description}
                  </p>
                </div>
                <div className="text-right">
                  <div className="flex justify-end mb-1"><DataOriginBadge origin={result.data_origin} /></div>
                  <p className="text-xs text-slate-500 mb-1">Weather indicator (display only)</p>
                  <span className={`px-3 py-1 rounded-full border text-sm font-bold ${riskStyle.badge}`}>
                    {result.risk_level}
                  </span>
                  <div className="w-24 ml-auto mt-2 bg-slate-200 dark:bg-slate-800 rounded-full h-1.5">
                    <div
                      className={`${riskStyle.bar} h-1.5 rounded-full transition-all duration-700`}
                      style={{ width: `${result.risk_score * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              icon={Thermometer} iconColor="text-orange-500"
              label="Temperature" value={result.temperature_c.toFixed(1)} unit="°C"
              sub={`Feels like ${result.feels_like_c.toFixed(1)}°C`}
            />
            <MetricCard
              icon={Droplets} iconColor="text-blue-500"
              label="Humidity" value={result.humidity_pct.toFixed(0)} unit="%"
            />
            <MetricCard
              icon={Wind} iconColor="text-cyan-500"
              label="Wind Speed" value={result.wind_speed_kmh.toFixed(1)} unit="km/h"
            />
            <MetricCard
              icon={CloudRain} iconColor="text-indigo-500"
              label="Rainfall" value={result.rainfall_mm.toFixed(1)} unit="mm"
              sub="Last 1 hour"
            />
          </div>
        </div>
      )}
    </div>
  );
}
