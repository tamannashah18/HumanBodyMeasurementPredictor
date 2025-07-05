export interface Measurements {
  [key: string]: number;
  ankle?: number;
  armLength?: number;
  bicep?: number;
  calf?: number;
  chest?: number;
  forearm?: number;
  height?: number;
  hip?: number;
  legLength?: number;
  shoulderBreadth?: number;
  shoulderToCrotch?: number;
  thigh?: number;
  waist?: number;
  wrist?: number;
}

export const MEASUREMENT_LABELS = {
  ankle: 'Ankle',
  armLength: 'Arm Length',
  bicep: 'Bicep',
  calf: 'Calf',
  chest: 'Chest',
  forearm: 'Forearm',
  height: 'Height',
  hip: 'Hip',
  legLength: 'Leg Length',
  shoulderBreadth: 'Shoulder Breadth',
  shoulderToCrotch: 'Shoulder to Crotch',
  thigh: 'Thigh',
  waist: 'Waist',
  wrist: 'Wrist'
};