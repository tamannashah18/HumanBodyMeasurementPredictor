import * as tf from '@tensorflow/tfjs';

// Model architecture inspired by the prototxt file but adapted for our use case
export const createModel = (): tf.LayersModel => {
  const model = tf.sequential({
    layers: [
      // Input layer - 128x128 RGB images
      tf.layers.conv2d({
        inputShape: [128, 128, 4], // RGBA
        filters: 64,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv1_1'
      }),
      tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv1_2'
      }),
      tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 2,
        name: 'pool1'
      }),

      // Second block
      tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv2_1'
      }),
      tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv2_2'
      }),
      tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 2,
        name: 'pool2'
      }),

      // Third block
      tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv3_1'
      }),
      tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv3_2'
      }),
      tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv3_3'
      }),
      tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 2,
        name: 'pool3'
      }),

      // Fourth block
      tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv4_1'
      }),
      tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        name: 'conv4_2'
      }),
      tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 2,
        name: 'pool4'
      }),

      // Global average pooling instead of flatten to reduce parameters
      tf.layers.globalAveragePooling2d({ name: 'global_avg_pool' }),

      // Dense layers for regression
      tf.layers.dense({
        units: 512,
        activation: 'relu',
        name: 'dense1'
      }),
      tf.layers.dropout({ rate: 0.5, name: 'dropout1' }),
      
      tf.layers.dense({
        units: 256,
        activation: 'relu',
        name: 'dense2'
      }),
      tf.layers.dropout({ rate: 0.3, name: 'dropout2' }),

      // Output layer - 14 measurements
      tf.layers.dense({
        units: 14,
        activation: 'linear', // Linear activation for regression
        name: 'measurements_output'
      })
    ]
  });

  return model;
};

// Generate synthetic training data
export const generateSyntheticData = (numSamples: number = 1000) => {
  // Create random images (simulating silhouettes)
  const images = tf.randomNormal([numSamples, 128, 128, 4]);
  
  // Generate realistic body measurements based on anthropometric data
  const measurements = tf.tidy(() => {
    // Base measurements for an average person (in cm)
    const baseMeasurements = [
      22,   // ankle
      60,   // arm length
      30,   // bicep
      35,   // calf
      95,   // chest
      25,   // forearm
      170,  // height
      95,   // hip
      90,   // leg length
      45,   // shoulder breadth
      75,   // shoulder to crotch
      55,   // thigh
      80,   // waist
      16    // wrist
    ];

    // Add some realistic variation
    const variations = tf.randomNormal([numSamples, 14], 0, 0.1);
    const baseArray = tf.tensor2d(Array(numSamples).fill(baseMeasurements));
    
    return baseArray.add(baseArray.mul(variations));
  });

  return { images, measurements };
};

// Train the model
export const createAndTrainModel = async (
  onProgress?: (epoch: number, logs?: tf.Logs) => void
): Promise<tf.LayersModel> => {
  console.log('Creating model...');
  const model = createModel();

  // Compile the model
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mae']
  });

  console.log('Generating synthetic training data...');
  const { images, measurements } = generateSyntheticData(1000);

  console.log('Starting training...');
  await model.fit(images, measurements, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs?.loss?.toFixed(4)}`);
        if (onProgress) {
          onProgress(epoch, logs);
        }
      }
    }
  });

  // Save the model
  await model.save('localstorage://body-measurement-model');
  console.log('Model saved successfully');

  // Cleanup
  images.dispose();
  measurements.dispose();

  return model;
};

// Load existing model
export const loadModel = async (): Promise<tf.LayersModel | null> => {
  try {
    console.log('Attempting to load existing model...');
    const model = await tf.loadLayersModel('localstorage://body-measurement-model');
    console.log('Model loaded successfully');
    return model;
  } catch (error) {
    console.log('No existing model found');
    return null;
  }
};