import { Routes, Route } from 'react-router-dom';
import { Toaster } from 'sonner';
import LandingPage from './pages/LandingPage';
import DashboardLayout from './layouts/DashboardLayout';
import Dashboard from './pages/Dashboard';
import DisasterMap from './pages/DisasterMap';
import LiveAlerts from './pages/LiveAlerts';
import Resources from './pages/Resources';
import Analytics from './pages/Analytics';
import SystemStatus from './pages/SystemStatus';
import NotFound from './pages/NotFound';
import VisionAI from './pages/VisionAI';
import NlpAI from './pages/NlpAI';
import WeatherIntel from './pages/WeatherIntel';
import FusionDashboard from './pages/FusionDashboard';

function App() {
  return (
    <>
      <Toaster position="top-right" theme="light" richColors />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<DashboardLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="map" element={<DisasterMap />} />
          <Route path="alerts" element={<LiveAlerts />} />
          <Route path="resources" element={<Resources />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="settings" element={<SystemStatus />} />
          {/* AI Module Routes */}
          <Route path="vision" element={<VisionAI />} />
          <Route path="nlp" element={<NlpAI />} />
          <Route path="weather" element={<WeatherIntel />} />
          <Route path="fusion" element={<FusionDashboard />} />
        </Route>
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}

export default App;
