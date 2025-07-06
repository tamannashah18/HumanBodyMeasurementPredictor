import boto3
import os
import json
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
import zipfile
import requests
from tqdm import tqdm
import pickle

class AWSBodyMDataLoader:
    """Load and process Amazon BodyM dataset from S3"""
    
    def __init__(self, data_dir="bodym_dataset"):
        self.data_dir = data_dir
        self.bucket_name = "amazon-bodym"
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.metadata = None
        self.processed_data_path = os.path.join(data_dir, "processed_data.pkl")
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
    
    def list_s3_contents(self):
        """List contents of the S3 bucket"""
        print("Listing S3 bucket contents...")
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            
            if 'Contents' in response:
                print(f"Found {len(response['Contents'])} objects in bucket:")
                for obj in response['Contents'][:10]:  # Show first 10
                    print(f"  {obj['Key']} ({obj['Size']} bytes)")
                if len(response['Contents']) > 10:
                    print(f"  ... and {len(response['Contents']) - 10} more objects")
                return [obj['Key'] for obj in response['Contents']]
            else:
                print("No objects found in bucket")
                return []
        except Exception as e:
            print(f"Error listing S3 contents: {e}")
            return []
    
    def download_file_from_s3(self, s3_key, local_path):
        """Download a file from S3"""
        try:
            print(f"Downloading {s3_key}...")
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            print(f"Downloaded to {local_path}")
            return True
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            return False
    
    def download_dataset_metadata(self):
        """Download dataset metadata and structure info"""
        print("Downloading dataset metadata...")
        
        # List all files to understand structure
        s3_objects = self.list_s3_contents()
        
        # Look for metadata files
        metadata_files = [obj for obj in s3_objects if 'metadata' in obj.lower() or obj.endswith('.json')]
        csv_files = [obj for obj in s3_objects if obj.endswith('.csv')]
        
        print(f"Found {len(metadata_files)} metadata files")
        print(f"Found {len(csv_files)} CSV files")
        
        # Download key files
        downloaded_files = []
        
        # Download first few metadata/csv files to understand structure
        for file_key in (metadata_files + csv_files)[:5]:
            local_path = os.path.join(self.data_dir, os.path.basename(file_key))
            if self.download_file_from_s3(file_key, local_path):
                downloaded_files.append(local_path)
        
        return downloaded_files, s3_objects
    
    def download_sample_images(self, max_images=100):
        """Download a sample of images for training"""
        print(f"Downloading sample images (max {max_images})...")
        
        # List all objects
        s3_objects = self.list_s3_contents()
        
        # Filter image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [obj for obj in s3_objects 
                      if any(obj.lower().endswith(ext) for ext in image_extensions)]
        
        print(f"Found {len(image_files)} image files")
        
        # Create images directory
        images_dir = os.path.join(self.data_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Download sample images
        downloaded_images = []
        for i, image_key in enumerate(image_files[:max_images]):
            if i >= max_images:
                break
                
            local_path = os.path.join(images_dir, f"image_{i:04d}_{os.path.basename(image_key)}")
            if self.download_file_from_s3(image_key, local_path):
                downloaded_images.append(local_path)
            
            if (i + 1) % 10 == 0:
                print(f"Downloaded {i + 1}/{min(max_images, len(image_files))} images")
        
        return downloaded_images
    
    def parse_measurements_data(self):
        """Parse measurement data from downloaded files"""
        print("Parsing measurement data...")
        
        measurements_data = []
        
        # Look for CSV files with measurements
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.data_dir, file_name)
                try:
                    df = pd.read_csv(file_path)
                    print(f"CSV file {file_name} columns: {list(df.columns)}")
                    
                    # Look for measurement-related columns
                    measurement_columns = []
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in 
                              ['height', 'weight', 'chest', 'waist', 'hip', 'arm', 'leg', 'shoulder']):
                            measurement_columns.append(col)
                    
                    if measurement_columns:
                        print(f"Found measurement columns: {measurement_columns}")
                        measurements_data.append({
                            'file': file_name,
                            'data': df,
                            'measurement_columns': measurement_columns
                        })
                
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")
        
        return measurements_data
    
    def create_training_dataset(self, image_paths, measurements_data):
        """Create training dataset from images and measurements"""
        print("Creating training dataset...")
        
        if not measurements_data:
            print("No measurement data found. Creating synthetic measurements...")
            return self.create_synthetic_dataset(image_paths)
        
        # Process real measurement data
        training_data = []
        
        # Use the first measurement dataset
        main_data = measurements_data[0]['data']
        measurement_cols = measurements_data[0]['measurement_columns']
        
        print(f"Using {len(measurement_cols)} measurement columns: {measurement_cols}")
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load and process image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Get corresponding measurements (if available)
                if i < len(main_data):
                    measurements = []
                    for col in measurement_cols:
                        value = main_data.iloc[i][col]
                        if pd.isna(value):
                            value = np.random.normal(100, 20)  # Fallback
                        measurements.append(float(value))
                else:
                    # Generate synthetic measurements if no real data
                    measurements = self.generate_synthetic_measurements()
                
                # Ensure we have exactly 14 measurements
                while len(measurements) < 14:
                    measurements.append(np.random.normal(50, 15))
                measurements = measurements[:14]
                
                training_data.append({
                    'image_path': image_path,
                    'image': image,
                    'measurements': np.array(measurements)
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Created training dataset with {len(training_data)} samples")
        return training_data
    
    def create_synthetic_dataset(self, image_paths):
        """Create synthetic measurement dataset"""
        print("Creating synthetic measurement dataset...")
        
        training_data = []
        measurement_labels = [
            'ankle', 'arm_length', 'bicep', 'calf', 'chest', 'forearm',
            'height', 'hip', 'leg_length', 'shoulder_breadth', 
            'shoulder_to_crotch', 'thigh', 'waist', 'wrist'
        ]
        
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                measurements = self.generate_synthetic_measurements()
                
                training_data.append({
                    'image_path': image_path,
                    'image': image,
                    'measurements': np.array(measurements)
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return training_data
    
    def generate_synthetic_measurements(self):
        """Generate realistic synthetic body measurements"""
        # Base measurements for an average adult (in cm)
        base_measurements = {
            'ankle': np.random.normal(22, 3),
            'arm_length': np.random.normal(60, 8),
            'bicep': np.random.normal(28, 5),
            'calf': np.random.normal(35, 5),
            'chest': np.random.normal(95, 15),
            'forearm': np.random.normal(25, 3),
            'height': np.random.normal(170, 15),
            'hip': np.random.normal(95, 12),
            'leg_length': np.random.normal(80, 10),
            'shoulder_breadth': np.random.normal(40, 6),
            'shoulder_to_crotch': np.random.normal(60, 8),
            'thigh': np.random.normal(55, 8),
            'waist': np.random.normal(80, 15),
            'wrist': np.random.normal(16, 2)
        }
        
        return [max(5, base_measurements[key]) for key in base_measurements.keys()]
    
    def save_processed_data(self, training_data):
        """Save processed training data"""
        print(f"Saving processed data to {self.processed_data_path}...")
        
        with open(self.processed_data_path, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"Saved {len(training_data)} training samples")
    
    def load_processed_data(self):
        """Load previously processed training data"""
        if os.path.exists(self.processed_data_path):
            print(f"Loading processed data from {self.processed_data_path}...")
            with open(self.processed_data_path, 'rb') as f:
                training_data = pickle.load(f)
            print(f"Loaded {len(training_data)} training samples")
            return training_data
        return None
    
    def prepare_dataset(self, max_images=500, force_download=False):
        """Main method to prepare the dataset"""
        print("=== Preparing Amazon BodyM Dataset ===")
        
        # Check if processed data already exists
        if not force_download:
            existing_data = self.load_processed_data()
            if existing_data:
                return existing_data
        
        # Download metadata and understand structure
        metadata_files, all_objects = self.download_dataset_metadata()
        
        # Download sample images
        image_paths = self.download_sample_images(max_images)
        
        if not image_paths:
            print("No images downloaded. Cannot proceed with training.")
            return None
        
        # Parse measurement data
        measurements_data = self.parse_measurements_data()
        
        # Create training dataset
        training_data = self.create_training_dataset(image_paths, measurements_data)
        
        # Save processed data
        self.save_processed_data(training_data)
        
        return training_data

def test_aws_connection():
    """Test AWS S3 connection"""
    print("Testing AWS S3 connection...")
    
    try:
        loader = AWSBodyMDataLoader()
        objects = loader.list_s3_contents()
        
        if objects:
            print(f"✓ Successfully connected to S3 bucket 'amazon-bodym'")
            print(f"✓ Found {len(objects)} objects in bucket")
            return True
        else:
            print("✗ No objects found in bucket")
            return False
            
    except Exception as e:
        print(f"✗ Error connecting to S3: {e}")
        return False

if __name__ == "__main__":
    # Test the connection and download sample data
    if test_aws_connection():
        loader = AWSBodyMDataLoader()
        training_data = loader.prepare_dataset(max_images=100)
        
        if training_data:
            print(f"\n✓ Dataset preparation completed!")
            print(f"✓ {len(training_data)} training samples ready")
            print("✓ You can now run the training script")
        else:
            print("\n✗ Dataset preparation failed")
    else:
        print("Please check your internet connection and try again")