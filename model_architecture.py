import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_body_measurement_model(input_shape=(128, 128, 4)):
    """
    Create a CNN model for body measurement prediction
    Based on the architecture from pose_deploy_linevec.prototxt but adapted for regression
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_4'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Fourth convolutional block
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_3_CPM'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4_4_CPM'),
        
        # Global average pooling to reduce parameters
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # Dense layers for measurement prediction
        layers.Dense(512, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout1'),
        
        layers.Dense(256, activation='relu', name='dense2'),
        layers.Dropout(0.3, name='dropout2'),
        
        layers.Dense(128, activation='relu', name='dense3'),
        layers.Dropout(0.2, name='dropout3'),
        
        # Output layer - 14 body measurements
        layers.Dense(14, activation='linear', name='measurements_output')
    ])
    
    return model

def compile_model(model):
    """Compile the model with appropriate optimizer and loss function"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    return model

def get_measurement_labels():
    """Return the labels for the 14 body measurements"""
    return [
        'ankle', 'arm_length', 'bicep', 'calf', 'chest', 'forearm',
        'height', 'hip', 'leg_length', 'shoulder_breadth', 
        'shoulder_to_crotch', 'thigh', 'waist', 'wrist'
    ]

if __name__ == "__main__":
    # Test model creation
    model = create_body_measurement_model()
    model = compile_model(model)
    model.summary()
    print(f"\nModel will predict {len(get_measurement_labels())} measurements:")
    for i, label in enumerate(get_measurement_labels()):
        print(f"{i+1}. {label}")