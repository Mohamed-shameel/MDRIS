import { Link } from 'react-router-dom';
import { ArrowRight, Eye, Brain, GitMerge, FlaskConical, Waves } from 'lucide-react';
import { WaveArt } from '../layouts/DashboardLayout';

const POINTS = [
  { icon: Eye, title: 'YOLOv8n (COCO-pretrained)', text: 'people and vehicles in flood imagery' },
  { icon: Brain, title: 'Fine-tuned DistilBERT', text: 'Low / Medium / High text urgency' },
  { icon: GitMerge, title: 'Documented fusion', text: 'confidence × source reliability' },
  { icon: FlaskConical, title: 'Honest demo data', text: 'simulated incidents, labelled everywhere' },
];

export default function LandingPage() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-b from-[#eef4fc] via-console-bg to-[#e6eef9] flex flex-col">
      {/* Decorative stripes and water line art */}
      <svg viewBox="0 0 600 600" overflow="visible" className="absolute -right-24 top-24 w-[640px] opacity-60 pointer-events-none" aria-hidden="true">
        <path d="M0 520 C 200 380, 380 240, 600 60" stroke="#f59a3c" strokeWidth="46" fill="none" opacity="0.28" />
        <path d="M0 580 C 200 440, 380 300, 600 120" stroke="#ffffff" strokeWidth="46" fill="none" opacity="0.9" />
        <path d="M0 640 C 200 500, 380 360, 600 180" stroke="#1f9d4c" strokeWidth="46" fill="none" opacity="0.22" />
      </svg>
      <WaveArt className="absolute left-6 bottom-6 w-[420px] opacity-70 pointer-events-none" />
      <WaveArt className="absolute right-10 bottom-2 w-[360px] opacity-50 pointer-events-none" />

      {/* Top bar */}
      <div className="relative flex items-start justify-between px-10 pt-8">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent to-navy-800 flex items-center justify-center shadow-md shadow-blue-900/20">
            <Waves size={26} className="text-white" />
          </div>
          <div className="leading-tight">
            <p className="font-display font-bold text-[26px] uppercase text-navy-900 tracking-tight">MDRIS Flood Response Console</p>
            <p className="text-[14px] font-medium text-navy-800">Decision support · Chennai demo</p>
          </div>
        </div>
        <div className="text-right hidden md:block">
          <p className="font-display text-[15px] tracking-[0.35em] text-navy-800 uppercase">Right team · Right place</p>
          <div className="flex justify-end gap-2 mt-2">
            <span className="h-1 w-10 rounded bg-saffron" />
            <span className="h-1 w-10 rounded bg-white border border-console-line" />
            <span className="h-1 w-10 rounded bg-indiagreen" />
          </div>
        </div>
      </div>

      {/* Card */}
      <div className="relative flex-1 flex items-center justify-center px-6 py-10">
        <div className="w-full max-w-xl rounded-3xl bg-white/95 border border-console-line shadow-[0_20px_60px_rgba(15,40,100,0.12)] px-10 py-10">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent to-navy-800 flex items-center justify-center shadow-lg shadow-blue-900/25">
              <Waves size={34} className="text-white" />
            </div>
            <div>
              <h1 className="font-display font-bold text-[30px] leading-tight text-navy-900">Flood Response Decision Support</h1>
              <p className="console-label mt-1">Priority queue · Evidence strength · Conflict flags</p>
            </div>
          </div>

          <p className="text-[15px] text-slate-600 mb-6">
            Which incident needs a team first? MDRIS fuses a YOLOv8n image detector and a fine-tuned text model into a
            ranked, explainable priority. Weather is shown for context only and never enters the score.
          </p>

          <Link
            to="/dashboard"
            className="flex items-center justify-center gap-2 w-full py-3.5 rounded-xl bg-accent hover:bg-accent-dark text-white font-semibold text-[16px] shadow-md shadow-blue-500/25 transition-colors"
          >
            Open the console <ArrowRight size={18} />
          </Link>

          <div className="my-7 border-t border-console-line" />
          <p className="text-center text-[13px] text-console-label mb-4">What is inside</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {POINTS.map(({ icon: Icon, title, text }) => (
              <div key={title} className="flex items-start gap-3 rounded-xl border border-console-line bg-console-side px-4 py-3">
                <Icon size={20} className="mt-0.5 text-accent shrink-0" />
                <div>
                  <p className="text-[14px] font-semibold text-navy-900">{title}</p>
                  <p className="text-[12px] text-console-label">{text}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 border-t border-console-line pt-5 text-center">
            <p className="text-[12px] tracking-[0.15em] text-navy-800">Applied Deep Learning course project</p>
            <div className="flex justify-center gap-2 my-2">
              <span className="h-1 w-8 rounded bg-saffron" />
              <span className="h-1 w-8 rounded bg-indiagreen" />
            </div>
            <p className="text-[12px] tracking-[0.3em] text-console-label uppercase">Flood scope only · simulated incidents</p>
          </div>
        </div>
      </div>
    </div>
  );
}
