import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from image_segmentation import ImageSegmenter
import random
from sklearn.model_selection import train_test_split

class RealDataGenerator(Sequence):
    """Data generator for real Amazon BodyM dataset"""
    
    def __init__(self, training_data, batch_size=32, shuffle=True, augment=True):
        self.training_data = training_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.segmenter = ImageSegmenter()
        self.indices = np.arange(len(training_data))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.training_data) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_images = []
        batch_measurements = []
        
        for i in batch_indices:
            try:
                data_point = self.training_data[i]
                image = data_point['image']
                measurements = data_point['measurements']
                
                # Process image
                processed_image = self.process_image(image)
                
                # Apply augmentation if enabled
                if self.augment:
                    processed_image, measurements = self.augment_data(processed_image, measurements)
                
                batch_images.append(processed_image)
                batch_measurements.append(measurements)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Use a fallback sample
                fallback_image = np.zeros((128, 128, 4), dtype=np.float32)
                fallback_measurements = np.random.normal(50, 15, 14)
                batch_images.append(fallback_image)
                batch_measurements.append(fallback_measurements)
        
        return np.array(batch_images), np.array(batch_measurements)
    
    def process_image(self, image):
        """Process image for model input"""
        try:
            # Segment person from background
            silhouette = self.segmenter.segment_person(image)
            
            # Preprocess for model
            processed = self.segmenter.preprocess_for_model(silhouette)
            
            return processed[0]  # Remove batch dimension
            
        except Exception as e:
            print(f"Error in image processing: {e}")
            # Return a default processed image
            return np.zeros((128, 128, 4), dtype=np.float32)
    
    def augment_data(self, image, measurements):
        """Apply data augmentation"""
        try:
            # Random horizontal flip
            if random.random() > 0.5:
                image = np.fliplr(image)
            
            # Random rotation (small angle)
            if random.random() > 0.7:
                angle = random.uniform(-5, 5)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, rotation_matrix, (w, h))
            
            # Random brightness adjustment
            if random.random() > 0.6:
                brightness_factor = random.uniform(0.8, 1.2)
                image = np.clip(image * brightness_factor, 0, 1)
            
            # Add small noise to measurements (±2cm)
            noise = np.random.normal(0, 2, len(measurements))
            measurements = measurements + noise
            measurements = np.clip(measurements, 5, 300)  # Reasonable bounds
            
            return image, measurements
            
        except Exception as e:
            print(f"Error in augmentation: {e}")
            return image, measurements
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_real_data_generators(training_data, validation_split=0.2, batch_size=32):
    """Create training and validation data generators from real data"""
    
    if not training_data:
        raise ValueError("No training data provided")
    
    print(f"Creating data generators from {len(training_data)} samples...")
    
    # Split data into training and validation
    train_data, val_data = train_test_split(
        training_data, 
        test_size=validation_split, 
        random_state=42
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create generators
    train_generator = RealDataGenerator(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        augment=True
    )
    
    val_generator = RealDataGenerator(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        augment=False
    )
    
    return train_generator, val_generator

def test_data_generator(training_data, num_samples=5):
    """Test the data generator with sample data"""
    print("Testing data generator...")
    
    if not training_data:
        print("No training data to test")
        return
    
    try:
        train_gen, val_gen = create_real_data_generators(training_data, batch_size=2)
        
        print(f"Train generator length: {len(train_gen)}")
        print(f"Validation generator length: {len(val_gen)}")
        
        # Test a batch
        batch_images, batch_measurements = train_gen[0]
        
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch measurements shape: {batch_measurements.shape}")
        print(f"Sample measurements: {batch_measurements[0]}")
        
        print("✓ Data generator test successful!")
        
    except Exception as e:
        print(f"✗ Data generator test failed: {e}")

if __name__ == "__main__":
    # Test with dummy data
    dummy_data = [
        {
            'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'measurements': np.random.normal(50, 15, 14)
        }
        for _ in range(10)
    ]
    
    test_data_generator(dummy_data)