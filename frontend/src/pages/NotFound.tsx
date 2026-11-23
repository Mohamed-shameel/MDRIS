import { Link } from 'react-router-dom';
import { AlertCircle } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="h-full w-full flex flex-col items-center justify-center p-8 text-center space-y-6 relative overflow-hidden">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-red-500/5 rounded-full blur-[100px] pointer-events-none"></div>
      
      <div className="relative z-10 p-6 bg-slate-900/50 rounded-3xl border border-slate-800 shadow-xl">
        <AlertCircle size={64} className="text-red-500 mx-auto mb-4" />
        <h1 className="text-6xl font-extrabold tracking-tight text-white mb-2">404</h1>
        <h2 className="text-2xl font-bold text-slate-300 mb-6">Sector Not Found</h2>
        <p className="text-slate-500 max-w-md mx-auto mb-8">
          The operation sector you are looking for does not exist or has been relocated. Return to the command center to resume monitoring.
        </p>
        <Link 
          to="/dashboard" 
          className="inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-full text-white bg-accent hover:bg-accent-dark transition-colors shadow-lg shadow-blue-900/20"
        >
          Return to Dashboard
        </Link>
      </div>
    </div>
  );
}
