import React from 'react';
import { Measurements, MEASUREMENT_LABELS } from '../types/measurements';

interface MeasurementResultsProps {
  measurements: Measurements;
}

export const MeasurementResults: React.FC<MeasurementResultsProps> = ({ measurements }) => {
  const measurementEntries = Object.entries(measurements).filter(([_, value]) => value !== undefined);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {measurementEntries.map(([key, value]) => (
          <div key={key} className="measurement-card">
            <div className="measurement-value">
              {value.toFixed(1)} cm
            </div>
            <div className="measurement-label">
              {MEASUREMENT_LABELS[key as keyof typeof MEASUREMENT_LABELS] || key}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <h3 className="font-semibold text-blue-800 mb-2">Summary</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-blue-600 font-medium">Height:</span> {measurements.height?.toFixed(1)} cm
          </div>
          <div>
            <span className="text-blue-600 font-medium">Chest:</span> {measurements.chest?.toFixed(1)} cm
          </div>
          <div>
            <span className="text-blue-600 font-medium">Waist:</span> {measurements.waist?.toFixed(1)} cm
          </div>
          <div>
            <span className="text-blue-600 font-medium">Hip:</span> {measurements.hip?.toFixed(1)} cm
          </div>
        </div>
      </div>

      <div className="text-xs text-gray-500 mt-4">
        * Measurements are estimates based on AI analysis. For precise measurements, please consult a professional.
      </div>
    </div>
  );
};