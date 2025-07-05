import React, { useState, useRef, useEffect } from 'react';
import { WebcamCapture } from './components/WebcamCapture';
import { MeasurementResults } from './components/MeasurementResults';
import { ModelTrainer } from './components/ModelTrainer';
import { loadModel, createAndTrainModel } from './utils/modelUtils';
import { Measurements } from './types/measurements';
import * as tf from '@tensorflow/tfjs';

function App() {
  const [measurements, setMeasurements] = useState<Measurements | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [isModelReady, setIsModelReady] = useState(false);
  const [showTraining, setShowTraining] = useState(false);

  useEffect(() => {
    initializeModel();
  }, []);

  const initializeModel = async () => {
    setIsLoading(true);
    try {
      // Try to load existing model first
      let loadedModel = await loadModel();
      
      if (!loadedModel) {
        console.log('No existing model found, creating new one...');
        loadedModel = await createAndTrainModel();
      }
      
      setModel(loadedModel);
      setIsModelReady(true);
    } catch (error) {
      console.error('Error initializing model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageCapture = async (imageData: ImageData) => {
    if (!model) {
      alert('Model not ready yet. Please wait...');
      return;
    }

    setIsLoading(true);
    try {
      // Preprocess the image
      const tensor = tf.browser.fromPixels(imageData)
        .resizeNearestNeighbor([128, 128])
        .expandDims(0)
        .div(255.0);

      // Make prediction
      const prediction = model.predict(tensor) as tf.Tensor;
      const results = await prediction.data();
      
      // Convert to measurements object
      const measurementLabels = [
        'ankle', 'armLength', 'bicep', 'calf', 'chest', 'forearm', 
        'height', 'hip', 'legLength', 'shoulderBreadth', 
        'shoulderToCrotch', 'thigh', 'waist', 'wrist'
      ];

      const measurementResults: Measurements = {};
      measurementLabels.forEach((label, index) => {
        measurementResults[label] = Math.round(results[index] * 100) / 100;
      });

      setMeasurements(measurementResults);
      
      // Cleanup tensors
      tensor.dispose();
      prediction.dispose();
    } catch (error) {
      console.error('Error making prediction:', error);
      alert('Error processing image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelTrained = (newModel: tf.LayersModel) => {
    setModel(newModel);
    setIsModelReady(true);
    setShowTraining(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            AI Body Measurement Predictor
          </h1>
          <p className="text-lg text-gray-600">
            Capture your photo and get instant body measurements using AI
          </p>
        </header>

        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column - Camera and Controls */}
            <div className="space-y-6">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                  Capture Your Image
                </h2>
                
                {isLoading && (
                  <div className="text-center py-8">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    <p className="mt-2 text-gray-600">
                      {!isModelReady ? 'Initializing AI model...' : 'Processing image...'}
                    </p>
                  </div>
                )}

                {!isLoading && (
                  <WebcamCapture 
                    onImageCapture={handleImageCapture}
                    disabled={!isModelReady}
                  />
                )}

                <div className="mt-4 flex gap-2">
                  <button
                    onClick={() => setShowTraining(!showTraining)}
                    className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    {showTraining ? 'Hide' : 'Show'} Model Training
                  </button>
                  
                  <button
                    onClick={initializeModel}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    disabled={isLoading}
                  >
                    Retrain Model
                  </button>
                </div>

                {showTraining && (
                  <div className="mt-4">
                    <ModelTrainer onModelTrained={handleModelTrained} />
                  </div>
                )}
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">
                  Instructions
                </h3>
                <ul className="space-y-2 text-gray-600">
                  <li className="flex items-start">
                    <span className="text-blue-600 mr-2">1.</span>
                    Stand 6-8 feet away from your camera
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-600 mr-2">2.</span>
                    Ensure good lighting and clear background
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-600 mr-2">3.</span>
                    Stand straight with arms slightly away from body
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-600 mr-2">4.</span>
                    Click "Capture Image" when ready
                  </li>
                </ul>
              </div>
            </div>

            {/* Right Column - Results */}
            <div>
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                  Your Measurements
                </h2>
                
                {measurements ? (
                  <MeasurementResults measurements={measurements} />
                ) : (
                  <div className="text-center py-12">
                    <div className="text-gray-400 mb-4">
                      <svg className="w-16 h-16 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <p className="text-gray-500">
                      Capture an image to see your body measurements
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <footer className="text-center mt-12 text-gray-500">
          <p>AI-powered body measurement prediction using TensorFlow.js</p>
        </footer>
      </div>
    </div>
  );
}

export default App;