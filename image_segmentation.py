import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import requests
import os

class ImageSegmenter:
    """Handle image segmentation using MediaPipe"""
    
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation_model = None
        self._initialize_segmentation()
    
    def _initialize_segmentation(self):
        """Initialize MediaPipe selfie segmentation"""
        try:
            self.segmentation_model = self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1  # 0 for general, 1 for landscape
            )
            print("MediaPipe segmentation model initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            print("Falling back to simple background removal")
    
    def segment_person(self, image):
        """
        Segment person from background and create silhouette
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            silhouette: White silhouette on black background (RGBA)
        """
        if self.segmentation_model is None:
            return self._simple_segmentation(image)
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.segmentation_model.process(rgb_image)
            
            # Create mask
            mask = results.segmentation_mask
            
            # Threshold the mask
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Create RGBA silhouette
            silhouette = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            
            # Set white pixels where person is detected
            silhouette[:, :, 0] = binary_mask  # R
            silhouette[:, :, 1] = binary_mask  # G
            silhouette[:, :, 2] = binary_mask  # B
            silhouette[:, :, 3] = binary_mask  # A
            
            return silhouette
            
        except Exception as e:
            print(f"Error in MediaPipe segmentation: {e}")
            return self._simple_segmentation(image)
    
    def _simple_segmentation(self, image):
        """
        Simple background removal using color-based segmentation
        Fallback method when MediaPipe is not available
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin-like colors (rough approximation)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Create mask for clothing (darker colors)
        lower_clothing = np.array([0, 0, 0], dtype=np.uint8)
        upper_clothing = np.array([180, 255, 100], dtype=np.uint8)
        clothing_mask = cv2.inRange(hsv, lower_clothing, upper_clothing)
        
        # Combine masks
        person_mask = cv2.bitwise_or(skin_mask, clothing_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
        
        # Create RGBA silhouette
        silhouette = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        silhouette[:, :, 0] = person_mask  # R
        silhouette[:, :, 1] = person_mask  # G
        silhouette[:, :, 2] = person_mask  # B
        silhouette[:, :, 3] = person_mask  # A
        
        return silhouette
    
    def preprocess_for_model(self, silhouette, target_size=(128, 128)):
        """
        Preprocess silhouette image for model input
        
        Args:
            silhouette: RGBA silhouette image
            target_size: Target size for resizing
            
        Returns:
            preprocessed: Normalized image ready for model
        """
        # Resize image
        resized = cv2.resize(silhouette, target_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        preprocessed = np.expand_dims(normalized, axis=0)
        
        return preprocessed

def download_deeplab_model():
    """Download DeepLab model if not already present"""
    model_path = "deeplabv3.tflite"
    
    if not os.path.exists(model_path):
        print("Downloading DeepLab model...")
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite"
        
        try:
            response = requests.get(url)
            with open(model_path, "wb") as file:
                file.write(response.content)
            print("DeepLab model downloaded successfully")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    
    return True

if __name__ == "__main__":
    # Test image segmentation
    segmenter = ImageSegmenter()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 's' to capture and segment image, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Webcam', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Segment the current frame
            silhouette = segmenter.segment_person(frame)
            
            # Show the silhouette
            cv2.imshow('Silhouette', silhouette)
            
            # Save the silhouette
            cv2.imwrite('test_silhouette.png', silhouette)
            print("Silhouette saved as 'test_silhouette.png'")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()