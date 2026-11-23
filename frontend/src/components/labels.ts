import type { DominantClass, TextSource, UrgencyLabel, VisionSource } from '../services/api';

export const SOURCE_LABELS: Record<VisionSource | TextSource, string> = {
  officer_photo: 'Officer photo',
  drone: 'Drone',
  cctv: 'CCTV',
  citizen_upload: 'Citizen upload',
  officer_report: 'Officer report',
  control_room_call: 'Control-room call',
  citizen_sms: 'Citizen SMS',
  social_media: 'Social media',
};

export const CLASS_LABELS: Record<DominantClass, string> = {
  person_in_water: 'Person in water',
  person_in_vehicle: 'Person in vehicle',
  person: 'Person',
  vehicle: 'Vehicle',
  text_only: 'Text only',
};

export const URGENCY_COLORS: Record<UrgencyLabel, string> = {
  Low: '#16A34A',
  Medium: '#CA8A04',
  High: '#DC2626',
};
