import boto3
import os
import numpy as np
import cv2
from botocore import UNSIGNED
from botocore.config import Config

class AWSBodyMFullDownloader:
    """
    Download all images and metadata from the Amazon BodyM S3 bucket.
    """

    def __init__(self, data_dir="bodym_full_dataset"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.metadata_dir = os.path.join(data_dir, "metadata")
        self.bucket_name = "amazon-bodym"
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def list_all_s3_objects(self):
        """
        List all objects in the S3 bucket, handling pagination.
        """
        print("Listing all S3 objects...")
        keys = []
        continuation_token = None
        while True:
            if continuation_token:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    ContinuationToken=continuation_token
                )
            else:
                response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            contents = response.get('Contents', [])
            for obj in contents:
                keys.append(obj['Key'])
            if response.get('IsTruncated'):
                continuation_token = response.get('NextContinuationToken')
            else:
                break
        print(f"Total objects found: {len(keys)}")
        return keys

    def download_all_metadata(self, s3_keys):
        """
        Download all metadata and CSV files from the S3 bucket.
        """
        print("Downloading all metadata and CSV files...")
        metadata_files = [k for k in s3_keys if 'metadata' in k.lower() or k.endswith('.json') or k.endswith('.csv')]
        for key in metadata_files:
            local_path = os.path.join(self.metadata_dir, os.path.basename(key))
            if not os.path.exists(local_path):
                try:
                    self.s3_client.download_file(self.bucket_name, key, local_path)
                    print(f"Downloaded metadata: {key}")
                except Exception as e:
                    print(f"Error downloading {key}: {e}")

    def download_all_images(self, s3_keys):
        """
        Download all image files from the S3 bucket.
        """
        print("Downloading all image files...")
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [k for k in s3_keys if k.lower().endswith(image_exts)]
        for i, key in enumerate(image_files):
            local_path = os.path.join(self.images_dir, f"image_{i:05d}_{os.path.basename(key)}")
            if not os.path.exists(local_path):
                try:
                    self.s3_client.download_file(self.bucket_name, key, local_path)
                    if (i + 1) % 50 == 0:
                        print(f"Downloaded {i + 1}/{len(image_files)} images")
                except Exception as e:
                    print(f"Error downloading {key}: {e}")

    def preprocess_images(self):
        """
        Example: Preprocess all images (resize, normalize, etc.).
        """
        print("Preprocessing images...")
        for fname in os.listdir(self.images_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(self.images_dir, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    # Example: Resize to 224x224 and normalize
                    img_resized = cv2.resize(img, (224, 224))
                    img_normalized = img_resized / 255.0
                    # Save preprocessed image as .npy
                    np.save(img_path.replace('.jpg', '.npy').replace('.jpeg', '.npy')
                            .replace('.png', '.npy').replace('.bmp', '.npy'), img_normalized)
        print("Preprocessing complete.")

    def run(self):
        """
        Run the full download and preprocessing pipeline.
        """
        s3_keys = self.list_all_s3_objects()
        self.download_all_metadata(s3_keys)
        self.download_all_images(s3_keys)
        self.preprocess_images()
        print("All files downloaded and preprocessed.")

if __name__ == "__main__":
    downloader = AWSBodyMFullDownloader()
    downloader.run()
