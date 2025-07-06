import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from tabulate import tabulate
from image_segmentation import ImageSegmenter
from model_architecture import get_measurement_labels
import os

class BodyMeasurementPredictor:
    """Real-time body measurement prediction from webcam"""
    
    def __init__(self, model_path="final_model.h5"):
        self.model_path = model_path
        self.model = None
        self.segmenter = ImageSegmenter()
        self.measurement_labels = get_measurement_labels()
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            print(f"Error: Model file '{self.model_path}' not found!")
            print("Please run 'train_model.py' first to train the model.")
            return False
        
        try:
            print("Loading trained model...")
            self.model = load_model(self.model_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def capture_image_with_countdown(self, cap, countdown_seconds=3):
        """Capture image with countdown"""
        print(f"Starting {countdown_seconds}-second countdown...")
        
        for i in range(countdown_seconds, 0, -1):
            ret, frame = cap.read()
            if not ret:
                return None
            
            # Add countdown text to frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(i)
            text_size = cv2.getTextSize(text, font, 3, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            cv2.putText(frame, text, (text_x, text_y), font, 3, (0, 255, 0), 3)
            cv2.imshow('Webcam - Get Ready!', frame)
            cv2.waitKey(1000)  # Wait 1 second
        
        # Capture final image
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "CAPTURED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Webcam - Get Ready!', frame)
            cv2.waitKey(1000)
        
        return frame if ret else None
    
    def predict_measurements(self, image):
        """Predict body measurements from image"""
        if self.model is None:
            print("Model not loaded!")
            return None
        
        try:
            # Segment the person from the image
            print("Segmenting person from background...")
            silhouette = self.segmenter.segment_person(image)
            
            # Preprocess for model
            print("Preprocessing image for model...")
            preprocessed = self.segmenter.preprocess_for_model(silhouette)
            
            # Make prediction
            print("Making prediction...")
            predictions = self.model.predict(preprocessed, verbose=0)
            
            # Convert to dictionary
            measurements = {}
            for i, label in enumerate(self.measurement_labels):
                measurements[label] = float(predictions[0][i])
            
            return measurements, silhouette
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None
    
    def display_results(self, measurements):
        """Display measurement results in a formatted table"""
        if measurements is None:
            print("No measurements to display.")
            return
        
        # Prepare data for table
        table_data = []
        for label, value in measurements.items():
            # Convert label to readable format
            readable_label = label.replace('_', ' ').title()
            table_data.append([readable_label, f"{value:.1f} cm"])
        
        # Display table
        print("\n" + "="*50)
        print("YOUR BODY MEASUREMENTS")
        print("="*50)
        print(tabulate(table_data, headers=["Measurement", "Value"], tablefmt="grid"))
        print("="*50)
        
        # Display key measurements summary
        key_measurements = ['height', 'chest', 'waist', 'hip']
        print("\nKEY MEASUREMENTS SUMMARY:")
        for key in key_measurements:
            if key in measurements:
                readable_key = key.replace('_', ' ').title()
                print(f"  {readable_key}: {measurements[key]:.1f} cm")
        
        print("\n* Measurements are AI estimates. For precise measurements, consult a professional.")
    
    def save_results(self, measurements, silhouette, image, filename_prefix="measurement"):
        """Save results to files"""
        if measurements is None:
            return
        
        timestamp = int(time.time())
        
        # Save original image
        original_filename = f"{filename_prefix}_{timestamp}_original.jpg"
        cv2.imwrite(original_filename, image)
        
        # Save silhouette
        silhouette_filename = f"{filename_prefix}_{timestamp}_silhouette.png"
        cv2.imwrite(silhouette_filename, silhouette)
        
        # Save measurements to text file
        measurements_filename = f"{filename_prefix}_{timestamp}_results.txt"
        with open(measurements_filename, 'w') as f:
            f.write("BODY MEASUREMENT RESULTS\n")
            f.write("="*30 + "\n\n")
            
            for label, value in measurements.items():
                readable_label = label.replace('_', ' ').title()
                f.write(f"{readable_label}: {value:.1f} cm\n")
            
            f.write(f"\nTimestamp: {time.ctime(timestamp)}\n")
            f.write("Note: Measurements are AI estimates.\n")
        
        print(f"\nResults saved:")
        print(f"  Original image: {original_filename}")
        print(f"  Silhouette: {silhouette_filename}")
        print(f"  Measurements: {measurements_filename}")
    
    def run_interactive_session(self):
        """Run interactive webcam session"""
        if self.model is None:
            return
        
        print("\n=== BODY MEASUREMENT PREDICTOR ===")
        print("Instructions:")
        print("1. Position yourself 6-8 feet from the camera")
        print("2. Ensure good lighting and clear background")
        print("3. Stand straight with arms slightly away from body")
        print("4. Press 'c' to capture and analyze")
        print("5. Press 'q' to quit")
        print("\nStarting webcam...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from webcam!")
                break
            
            # Add instructions to frame
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Body Measurement Predictor', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Capture image with countdown
                captured_image = self.capture_image_with_countdown(cap)
                
                if captured_image is not None:
                    # Predict measurements
                    measurements, silhouette = self.predict_measurements(captured_image)
                    
                    if measurements is not None:
                        # Display results
                        self.display_results(measurements)
                        
                        # Show silhouette
                        cv2.imshow('Your Silhouette', silhouette)
                        
                        # Ask if user wants to save results
                        save_choice = input("\nSave results to files? (y/n): ").lower()
                        if save_choice == 'y':
                            self.save_results(measurements, silhouette, captured_image)
                        
                        print("\nPress any key to continue or 'q' to quit...")
                        cv2.waitKey(0)
                        cv2.destroyWindow('Your Silhouette')
                    else:
                        print("Failed to predict measurements. Please try again.")
                else:
                    print("Failed to capture image. Please try again.")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Session ended.")

def main():
    """Main function"""
    predictor = BodyMeasurementPredictor()
    
    if predictor.model is not None:
        predictor.run_interactive_session()
    else:
        print("\nTo get started:")
        print("1. Run 'python train_model.py' to train the model")
        print("2. Then run this script again to make predictions")

if __name__ == "__main__":
    main()