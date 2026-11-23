import { useEffect, useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { Home, Map, Bell, Package, BarChart3, Settings, Eye, Brain, Cloud, GitMerge, Waves } from 'lucide-react';
import { fetchSystemStatus, type SystemStatusPayload } from '../services/api';

const NAV = [
  {
    section: 'Operations',
    items: [
      { name: 'Command Dashboard', icon: Home, path: '/dashboard' },
      { name: 'Incident Map', icon: Map, path: '/dashboard/map' },
      { name: 'Alerts', icon: Bell, path: '/dashboard/alerts' },
      { name: 'Resources', icon: Package, path: '/dashboard/resources' },
    ],
  },
  {
    section: 'Assurance',
    items: [
      { name: 'Model Evaluation', icon: BarChart3, path: '/dashboard/analytics' },
      { name: 'System Status', icon: Settings, path: '/dashboard/settings' },
    ],
  },
  {
    section: 'AI Modules',
    items: [
      { name: 'AI Vision', icon: Eye, path: '/dashboard/vision' },
      { name: 'Urgency NLP (DistilBERT)', icon: Brain, path: '/dashboard/nlp' },
      { name: 'Incident Analyzer (Fusion)', icon: GitMerge, path: '/dashboard/fusion' },
      { name: 'Weather (context only)', icon: Cloud, path: '/dashboard/weather' },
    ],
  },
];

/** Line-art water waves (decoration for the header and the sidebar foot). */
export const WaveArt = ({ className = '' }: { className?: string }) => (
  <svg viewBox="0 0 220 90" fill="none" className={className} aria-hidden="true">
    {[0, 1, 2, 3].map((i) => (
      <path
        key={i}
        d={`M0 ${30 + i * 14} C 27 ${18 + i * 14}, 55 ${42 + i * 14}, 82 ${30 + i * 14} S 137 ${18 + i * 14}, 165 ${30 + i * 14} S 205 ${42 + i * 14}, 220 ${30 + i * 14}`}
        stroke="#8fb3ea"
        strokeWidth={1.4}
        opacity={1 - i * 0.18}
      />
    ))}
  </svg>
);

/** Faint saffron / white / green diagonal stripes (header decoration). */
const TricolourStripes = () => (
  <svg viewBox="0 0 260 80" className="absolute right-40 top-0 h-full w-[260px] pointer-events-none" aria-hidden="true">
    <polygon points="70,0 110,0 60,80 20,80" fill="#f59a3c" opacity="0.28" />
    <polygon points="110,0 150,0 100,80 60,80" fill="#ffffff" opacity="0.7" />
    <polygon points="150,0 190,0 140,80 100,80" fill="#1f9d4c" opacity="0.25" />
  </svg>
);

/** Real model load state from /api/system-status (no invented health figures). */
function useModelStatus() {
  const [status, setStatus] = useState<SystemStatusPayload | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const location = useLocation();
  useEffect(() => {
    fetchSystemStatus()
      .then((res) => { setStatus(res.data); setUnreachable(false); })
      .catch(() => setUnreachable(true));
  }, [location.pathname]);
  const loaded = status ? [status.models.vision.loaded, status.models.nlp.loaded].filter(Boolean).length : 0;
  return { status, unreachable, loaded };
}

const Sidebar = () => {
  const location = useLocation();
  return (
    <aside className="h-full w-64 shrink-0 bg-console-side border-r border-console-line flex flex-col overflow-y-auto">
      <nav className="flex-1 px-3 pt-5 space-y-5">
        {NAV.map((group) => (
          <div key={group.section}>
            <p className="console-label px-3 mb-2">{group.section}</p>
            <div className="space-y-1">
              {group.items.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.name}
                    to={item.path}
                    className={`flex items-center gap-3 px-3 py-2 rounded-lg text-[14px] font-medium transition-colors ${
                      isActive
                        ? 'bg-console-active text-accent-dark'
                        : 'text-navy-900 hover:bg-console-soft'
                    }`}
                  >
                    <Icon size={18} className={isActive ? 'text-accent' : 'text-navy-800'} />
                    <span>{item.name}</span>
                  </Link>
                );
              })}
            </div>
            <div className="mt-4 mx-3 border-b border-console-line" />
          </div>
        ))}
      </nav>
      <div className="px-6 pb-6 pt-4">
        <p className="font-display font-extrabold text-[22px] leading-[1.05] text-navy-900 uppercase">
          Right team<br />Right place<br />First
        </p>
        <div className="flex gap-2 mt-2">
          <span className="h-1 w-8 rounded bg-saffron" />
          <span className="h-1 w-8 rounded bg-white border border-console-line" />
          <span className="h-1 w-8 rounded bg-indiagreen" />
        </div>
        <WaveArt className="w-40 mt-3" />
      </div>
    </aside>
  );
};

export default function DashboardLayout() {
  const { status, unreachable, loaded } = useModelStatus();
  const ok = !!status && loaded === 2;

  return (
    <div className="flex flex-col h-screen bg-console-bg">
      {/* Header */}
      <header className="relative h-20 shrink-0 bg-white border-b border-console-line flex items-center gap-8 px-6 overflow-hidden">
        <TricolourStripes />
        <WaveArt className="absolute right-44 top-1 h-20 w-56 opacity-70 pointer-events-none" />
        <Link to="/" className="relative flex items-center gap-3">
          <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-accent to-navy-800 flex items-center justify-center shadow-md shadow-blue-900/20">
            <Waves size={24} className="text-white" />
          </div>
          <div className="leading-tight">
            <p className="font-display font-bold text-[21px] uppercase text-navy-900 tracking-tight">MDRIS Flood Response Console</p>
            <p className="text-[12px] font-medium text-navy-800">Decision support · Chennai demo (simulated incidents)</p>
          </div>
        </Link>
        <div className="relative hidden xl:flex items-center gap-5 font-mono text-[12px]">
          <span className="text-console-label">SCOPE <span className="text-navy-900 font-semibold">FLOOD</span></span>
          <span className="text-console-label">REGION <span className="text-navy-900 font-semibold">CHENNAI</span></span>
          <span className="text-console-label">
            MODELS{' '}
            <span className={`font-semibold ${ok ? 'text-navy-900' : 'text-red-600'}`}>
              {unreachable ? 'backend unreachable' : !status ? '…' : `${loaded}/2 loaded`}
            </span>
          </span>
        </div>
        <div className="relative ml-auto flex items-center gap-3">
          <Link
            to="/dashboard/settings"
            title="Live model load state"
            className={`px-4 py-2 rounded-full border text-[13px] font-semibold uppercase tracking-wide ${
              ok ? 'border-blue-200 bg-console-soft text-accent-dark' : 'border-amber-300 bg-amber-50 text-amber-700'
            }`}
          >
            {ok ? 'Models online' : unreachable ? 'Offline' : 'Models: check'}
          </Link>
          <Link
            to="/dashboard/alerts"
            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-console-line bg-white text-navy-900 text-[14px] font-semibold hover:bg-console-soft"
          >
            <Bell size={18} /> Alerts
          </Link>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-auto px-8 py-6">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
