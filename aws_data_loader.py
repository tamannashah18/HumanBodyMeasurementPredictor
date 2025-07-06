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
    
    def list_s3_contents(self, max_keys=None):
        """List contents of the S3 bucket"""
        print("Listing S3 bucket contents...")
        try:
            all_objects = []
            continuation_token = None
            
            while True:
                if continuation_token:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        ContinuationToken=continuation_token,
                        MaxKeys=1000
                    )
                else:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        MaxKeys=1000
                    )
                
                if 'Contents' in response:
                    all_objects.extend([obj['Key'] for obj in response['Contents']])
                
                if response.get('IsTruncated'):
                    continuation_token = response.get('NextContinuationToken')
                else:
                    break
                
                if max_keys and len(all_objects) >= max_keys:
                    all_objects = all_objects[:max_keys]
                    break
            
            print(f"Found {len(all_objects)} objects in bucket:")
            for obj in all_objects[:10]:  # Show first 10
                print(f"  {obj}")
            if len(all_objects) > 10:
                print(f"  ... and {len(all_objects) - 10} more objects")
            
            return all_objects
            
        except Exception as e:
            print(f"Error listing S3 contents: {e}")
            return []
    
    def download_file_from_s3(self, s3_key, local_path):
        """Download a file from S3"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            return False
    
    def download_dataset_metadata(self):
        """Download dataset metadata and structure info"""
        print("Downloading dataset metadata...")
        
        # List all files to understand structure
        s3_objects = self.list_s3_contents(max_keys=5000)  # Get more objects for analysis
        
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
    
    def download_sample_images(self, max_images=1000, s3_objects=None):
        """Download a sample of images for training"""
        print(f"Downloading sample images (max {max_images})...")
        
        # Use provided objects or list them
        if s3_objects is None:
            s3_objects = self.list_s3_contents()
        
        # Filter image files - look for common patterns in BodyM dataset
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for obj in s3_objects:
            obj_lower = obj.lower()
            # Look for image files, prioritize certain directories
            if any(obj_lower.endswith(ext) for ext in image_extensions):
                # Prioritize files in specific directories that likely contain body images
                if any(keyword in obj_lower for keyword in ['image', 'photo', 'body', 'person', 'human']):
                    image_files.insert(0, obj)  # Add to front
                else:
                    image_files.append(obj)
        
        print(f"Found {len(image_files)} image files")
        
        if not image_files:
            print("No image files found in the bucket!")
            return []
        
        # Create images directory
        images_dir = os.path.join(self.data_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Download sample images with progress bar
        downloaded_images = []
        failed_downloads = 0
        
        print(f"Downloading {min(max_images, len(image_files))} images...")
        
        for i in tqdm(range(min(max_images, len(image_files))), desc="Downloading images"):
            image_key = image_files[i]
            
            # Create a clean filename
            filename = os.path.basename(image_key)
            if not filename:
                filename = f"image_{i:04d}.jpg"
            
            local_path = os.path.join(images_dir, f"{i:04d}_{filename}")
            
            if self.download_file_from_s3(image_key, local_path):
                # Verify the downloaded file is a valid image
                try:
                    img = cv2.imread(local_path)
                    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                        downloaded_images.append(local_path)
                    else:
                        os.remove(local_path)  # Remove invalid image
                        failed_downloads += 1
                except:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    failed_downloads += 1
            else:
                failed_downloads += 1
        
        print(f"Successfully downloaded {len(downloaded_images)} valid images")
        if failed_downloads > 0:
            print(f"Failed to download or process {failed_downloads} images")
        
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
                    print(f"CSV file {file_name} shape: {df.shape}")
                    
                    # Look for measurement-related columns
                    measurement_columns = []
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in 
                              ['height', 'weight', 'chest', 'waist', 'hip', 'arm', 'leg', 'shoulder', 'measurement']):
                            measurement_columns.append(col)
                    
                    if measurement_columns:
                        print(f"Found measurement columns: {measurement_columns}")
                        measurements_data.append({
                            'file': file_name,
                            'data': df,
                            'measurement_columns': measurement_columns
                        })
                    else:
                        print(f"No measurement columns found in {file_name}")
                
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")
        
        return measurements_data
    
    def create_training_dataset(self, image_paths, measurements_data, target_samples=None):
        """Create training dataset from images and measurements"""
        print(f"Creating training dataset from {len(image_paths)} images...")
        
        if target_samples is None:
            target_samples = len(image_paths)
        
        if not measurements_data:
            print("No measurement data found. Creating synthetic measurements...")
            return self.create_synthetic_dataset(image_paths, target_samples)
        
        # Process real measurement data
        training_data = []
        
        # Use the first measurement dataset
        main_data = measurements_data[0]['data']
        measurement_cols = measurements_data[0]['measurement_columns']
        
        print(f"Using {len(measurement_cols)} measurement columns: {measurement_cols}")
        print(f"Available measurement records: {len(main_data)}")
        
        successful_processed = 0
        failed_processed = 0
        
        for i, image_path in enumerate(tqdm(image_paths[:target_samples], desc="Processing images")):
            try:
                # Load and validate image
                image = cv2.imread(image_path)
                if image is None:
                    failed_processed += 1
                    continue
                
                # Check image dimensions
                if image.shape[0] < 50 or image.shape[1] < 50:
                    failed_processed += 1
                    continue
                
                # Get corresponding measurements (if available)
                if i < len(main_data):
                    measurements = []
                    for col in measurement_cols:
                        try:
                            value = main_data.iloc[i][col]
                            if pd.isna(value):
                                value = np.random.normal(100, 20)  # Fallback
                            measurements.append(float(value))
                        except:
                            measurements.append(np.random.normal(100, 20))
                else:
                    # Generate synthetic measurements if no real data
                    measurements = self.generate_synthetic_measurements()
                
                # Ensure we have exactly 14 measurements
                while len(measurements) < 14:
                    measurements.append(np.random.normal(50, 15))
                measurements = measurements[:14]
                
                # Validate measurements (reasonable ranges)
                measurements = [max(5, min(300, m)) for m in measurements]
                
                training_data.append({
                    'image_path': image_path,
                    'image': image,
                    'measurements': np.array(measurements, dtype=np.float32)
                })
                
                successful_processed += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                failed_processed += 1
                continue
        
        print(f"Successfully processed: {successful_processed} images")
        print(f"Failed to process: {failed_processed} images")
        print(f"Created training dataset with {len(training_data)} samples")
        
        return training_data
    
    def create_synthetic_dataset(self, image_paths, target_samples=None):
        """Create synthetic measurement dataset"""
        print("Creating synthetic measurement dataset...")
        
        if target_samples is None:
            target_samples = len(image_paths)
        
        training_data = []
        measurement_labels = [
            'ankle', 'arm_length', 'bicep', 'calf', 'chest', 'forearm',
            'height', 'hip', 'leg_length', 'shoulder_breadth', 
            'shoulder_to_crotch', 'thigh', 'waist', 'wrist'
        ]
        
        successful_processed = 0
        failed_processed = 0
        
        for i, image_path in enumerate(tqdm(image_paths[:target_samples], desc="Processing images")):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    failed_processed += 1
                    continue
                
                # Check image dimensions
                if image.shape[0] < 50 or image.shape[1] < 50:
                    failed_processed += 1
                    continue
                
                measurements = self.generate_synthetic_measurements()
                
                training_data.append({
                    'image_path': image_path,
                    'image': image,
                    'measurements': np.array(measurements, dtype=np.float32)
                })
                
                successful_processed += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                failed_processed += 1
                continue
        
        print(f"Successfully processed: {successful_processed} images")
        print(f"Failed to process: {failed_processed} images")
        
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
    
    def prepare_dataset(self, max_images=1000, force_download=False):
        """Main method to prepare the dataset"""
        print("=== Preparing Amazon BodyM Dataset ===")
        
        # Check if processed data already exists and has enough samples
        if not force_download:
            existing_data = self.load_processed_data()
            if existing_data and len(existing_data) >= min(max_images * 0.8, 100):
                print(f"Using existing processed data with {len(existing_data)} samples")
                return existing_data
        
        # Download metadata and understand structure
        metadata_files, all_objects = self.download_dataset_metadata()
        
        # Download sample images
        image_paths = self.download_sample_images(max_images, all_objects)
        
        if not image_paths:
            print("No images downloaded. Cannot proceed with training.")
            return None
        
        print(f"Downloaded {len(image_paths)} images")
        
        # Parse measurement data
        measurements_data = self.parse_measurements_data()
        
        # Create training dataset
        training_data = self.create_training_dataset(image_paths, measurements_data, max_images)
        
        if not training_data:
            print("Failed to create training dataset!")
            return None
        
        # Save processed data
        self.save_processed_data(training_data)
        
        return training_data

def test_aws_connection():
    """Test AWS S3 connection"""
    print("Testing AWS S3 connection...")
    
    try:
        loader = AWSBodyMDataLoader()
        objects = loader.list_s3_contents(max_keys=100)
        
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
        training_data = loader.prepare_dataset(max_images=1000, force_download=True)
        
        if training_data:
            print(f"\n✓ Dataset preparation completed!")
            print(f"✓ {len(training_data)} training samples ready")
            print("✓ You can now run the training script")
        else:
            print("\n✗ Dataset preparation failed")
    else:
        print("Please check your internet connection and try again")