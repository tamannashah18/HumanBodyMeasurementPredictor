import React, { useState } from 'react';
import { createAndTrainModel } from '../utils/modelUtils';
import * as tf from '@tensorflow/tfjs';

interface ModelTrainerProps {
  onModelTrained: (model: tf.LayersModel) => void;
}

export const ModelTrainer: React.FC<ModelTrainerProps> = ({ onModelTrained }) => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingLoss, setTrainingLoss] = useState<number | null>(null);

  const handleTrainModel = async () => {
    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingLoss(null);

    try {
      const model = await createAndTrainModel((epoch, logs) => {
        setTrainingProgress(((epoch + 1) / 50) * 100);
        if (logs?.loss) {
          setTrainingLoss(logs.loss as number);
        }
      });

      onModelTrained(model);
      alert('Model training completed successfully!');
    } catch (error) {
      console.error('Training failed:', error);
      alert('Model training failed. Please try again.');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <h3 className="text-lg font-semibold text-gray-800 mb-3">Model Training</h3>
      
      {!isTraining ? (
        <div className="space-y-3">
          <p className="text-sm text-gray-600">
            Train a new model with synthetic data. This will create a neural network 
            that can predict body measurements from images.
          </p>
          <button
            onClick={handleTrainModel}
            className="w-full bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors"
          >
            Start Training
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex justify-between text-sm">
            <span>Training Progress</span>
            <span>{trainingProgress.toFixed(1)}%</span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-purple-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${trainingProgress}%` }}
            />
          </div>

          {trainingLoss !== null && (
            <div className="text-sm text-gray-600">
              Current Loss: {trainingLoss.toFixed(4)}
            </div>
          )}

          <div className="text-sm text-gray-500">
            Training in progress... This may take a few minutes.
          </div>
        </div>
      )}
    </div>
  );
};