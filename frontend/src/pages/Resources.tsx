import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { fetchResources, type DataOrigin, type ResourceItem } from '../services/api';
import DataOriginBadge from '../components/DataOriginBadge';
import LoadError, { errorDetail } from '../components/LoadError';

export default function Resources() {
  const [data, setData] = useState<{ data_origin: DataOrigin; note: string; resources: ResourceItem[] } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchResources().then((res) => setData(res.data)).catch((err) => setError(errorDetail(err)));
  }, []);

  if (error) return <LoadError title="Resources unavailable" detail={error} />;
  if (!data) {
    return <div className="animate-pulse flex-1 h-full w-full bg-slate-100 dark:bg-slate-900 rounded-xl" />;
  }

  const getStatusBadge = (status: string) => {
    const base = "px-2.5 py-1 rounded-full text-xs font-semibold";
    switch (status.toLowerCase()) {
      case 'active':
      case 'available':
        return `${base} bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400`;
      case 'deploying':
      case 'distributing':
        return `${base} bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400`;
      case 'standby':
        return `${base} bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400`;
      default:
        return `${base} bg-slate-100 text-slate-700`;
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="console-title text-[28px]">Resources</h2>
          <p className="text-slate-500">Illustrative relief inventory. {data.note}</p>
        </div>
        <DataOriginBadge origin={data.data_origin} />
      </div>

      <Card className="border-console-line shadow-md overflow-hidden">
        <CardHeader className="border-b border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50">
          <CardTitle className="flex items-center justify-between">
            <span>Inventory & Deployment Status</span>
            <DataOriginBadge origin={data.data_origin} />
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0 overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-console-bg border-b border-console-line">
                <th className="p-4 text-sm font-semibold text-slate-600 dark:text-slate-400">Resource ID</th>
                <th className="p-4 text-sm font-semibold text-slate-600 dark:text-slate-400">Type</th>
                <th className="p-4 text-sm font-semibold text-slate-600 dark:text-slate-400">Total</th>
                <th className="p-4 text-sm font-semibold text-slate-600 dark:text-slate-400">Available</th>
                <th className="p-4 text-sm font-semibold text-slate-600 dark:text-slate-400">Assigned Area</th>
                <th className="p-4 text-sm font-semibold text-slate-600 dark:text-slate-400">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
              {data.resources.map((res) => (
                <tr key={res.id} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/30 transition-colors">
                  <td className="p-4 text-sm font-mono text-slate-500">{res.id}</td>
                  <td className="p-4 text-sm font-semibold">{res.type}</td>
                  <td className="p-4 text-sm">{res.total.toLocaleString()}</td>
                  <td className="p-4 text-sm font-medium text-slate-900 dark:text-slate-100">{res.available.toLocaleString()}</td>
                  <td className="p-4 text-sm text-slate-600 dark:text-slate-400">{res.assigned_area}</td>
                  <td className="p-4">
                    <span className={getStatusBadge(res.status)}>{res.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}
