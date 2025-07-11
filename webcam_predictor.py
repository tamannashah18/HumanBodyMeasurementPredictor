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
    
    def __init__(self, model_path=None):
        # Try different model paths in order of preference
        possible_models = [
            "bodym_real_model.h5",  # AWS-trained model (preferred)
            "final_model.h5",       # Original synthetic model
            model_path              # User-specified path
        ]
        
        self.model_path = None
        for path in possible_models:
            if path and os.path.exists(path):
                self.model_path = path
                break
        
        if not self.model_path:
            print("No trained model found!")
            print("Available options:")
            print("1. Train with real data: python train_real_model.py")
            print("2. Train with synthetic data: python train_model.py")
            return
        
        self.model = None
        self.segmenter = ImageSegmenter()
        self.measurement_labels = get_measurement_labels()
        self._load_model()
    
    def _load_model(self):
        """Load the trained model with compatibility handling"""
        try:
            print(f"Loading model from {self.model_path}...")
            
            # Try loading with custom objects to handle compatibility issues
            try:
                self.model = load_model(self.model_path, compile=False)
                # Recompile the model to ensure compatibility
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='mean_squared_error',
                    metrics=['mean_absolute_error', 'mean_squared_error']
                )
                print("✓ Model loaded and recompiled successfully!")
            except Exception as e1:
                print(f"Standard loading failed: {e1}")
                print("Trying alternative loading method...")
                
                # Try loading weights only and rebuild model
                from model_architecture import create_body_measurement_model, compile_model
                print("Rebuilding model architecture...")
                self.model = create_body_measurement_model()
                self.model = compile_model(self.model)
                
                # Try to load weights
                try:
                    self.model.load_weights(self.model_path)
                    print("✓ Model weights loaded successfully!")
                except Exception as e2:
                    print(f"Weight loading also failed: {e2}")
                    raise e2
            
            # Display model info
            if "real" in self.model_path.lower() or "bodym" in self.model_path.lower():
                print("📊 Using AWS BodyM dataset trained model")
            else:
                print("🔬 Using synthetic dataset trained model")
                
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("1. The model might be corrupted or incompatible")
            print("2. Try retraining the model:")
            print("   - For real data: python train_real_model.py")
            print("   - For synthetic data: python train_model.py")
            print("3. Check TensorFlow version compatibility")
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
            cv2.putText(frame, "Get Ready!", (50, 50), font, 1, (255, 255, 255), 2)
            cv2.imshow('Body Measurement Predictor', frame)
            cv2.waitKey(1000)  # Wait 1 second
        
        # Capture final image
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "CAPTURED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Body Measurement Predictor', frame)
            cv2.waitKey(1000)
        
        return frame if ret else None
    
    def predict_measurements(self, image):
        """Predict body measurements from image"""
        if self.model is None:
            print("Model not loaded!")
            return None
        
        try:
            # Segment the person from the image
            print("🔍 Segmenting person from background...")
            silhouette = self.segmenter.segment_person(image)
            
            # Preprocess for model
            print("⚙️  Preprocessing image for model...")
            preprocessed = self.segmenter.preprocess_for_model(silhouette)
            
            # Make prediction
            print("🧠 Making AI prediction...")
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
        print("\n" + "="*60)
        print("🎯 YOUR BODY MEASUREMENTS")
        print("="*60)
        print(tabulate(table_data, headers=["Measurement", "Value"], tablefmt="grid"))
        print("="*60)
        
        # Display key measurements summary
        key_measurements = ['height', 'chest', 'waist', 'hip']
        print("\n📏 KEY MEASUREMENTS SUMMARY:")
        for key in key_measurements:
            if key in measurements:
                readable_key = key.replace('_', ' ').title()
                print(f"  {readable_key}: {measurements[key]:.1f} cm")
        
        # Model info
        model_type = "AWS BodyM" if ("real" in self.model_path.lower() or "bodym" in self.model_path.lower()) else "Synthetic"
        print(f"\n🤖 Model: {model_type} dataset trained")
        print("📝 Note: Measurements are AI estimates. For precise measurements, consult a professional.")
    
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
            f.write("="*40 + "\n\n")
            
            model_type = "AWS BodyM" if ("real" in self.model_path.lower() or "bodym" in self.model_path.lower()) else "Synthetic"
            f.write(f"Model: {model_type} dataset trained\n")
            f.write(f"Model file: {self.model_path}\n\n")
            
            for label, value in measurements.items():
                readable_label = label.replace('_', ' ').title()
                f.write(f"{readable_label}: {value:.1f} cm\n")
            
            f.write(f"\nTimestamp: {time.ctime(timestamp)}\n")
            f.write("Note: Measurements are AI estimates.\n")
        
        print(f"\n💾 Results saved:")
        print(f"  📷 Original image: {original_filename}")
        print(f"  👤 Silhouette: {silhouette_filename}")
        print(f"  📊 Measurements: {measurements_filename}")
    
    def run_interactive_session(self):
        """Run interactive webcam session"""
        if self.model is None:
            return
        
        print("\n" + "="*60)
        print("🎯 BODY MEASUREMENT PREDICTOR")
        print("="*60)
        print("📋 Instructions:")
        print("1. Position yourself 6-8 feet from the camera")
        print("2. Ensure good lighting and clear background")
        print("3. Stand straight with arms slightly away from body")
        print("4. Press 'c' to capture and analyze")
        print("5. Press 'q' to quit")
        print(f"\n🤖 Model: {os.path.basename(self.model_path)}")
        print("🎥 Starting webcam...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam!")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read from webcam!")
                break
            
            # Add instructions to frame
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add model info
            model_info = f"Model: {os.path.basename(self.model_path)}"
            cv2.putText(frame, model_info, 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
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
                        save_choice = input("\n💾 Save results to files? (y/n): ").lower()
                        if save_choice == 'y':
                            self.save_results(measurements, silhouette, captured_image)
                        
                        print("\n⌨️  Press any key to continue or 'q' to quit...")
                        cv2.waitKey(0)
                        cv2.destroyWindow('Your Silhouette')
                    else:
                        print("❌ Failed to predict measurements. Please try again.")
                else:
                    print("❌ Failed to capture image. Please try again.")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Session ended.")

def main():
    """Main function"""
    print("🚀 Starting Body Measurement Predictor...")
    
    # Check for available models
    models = []
    if os.path.exists("bodym_real_model.h5"):
        models.append(("bodym_real_model.h5", "AWS BodyM dataset (recommended)"))
    if os.path.exists("final_model.h5"):
        models.append(("final_model.h5", "Synthetic dataset"))
    
    if not models:
        print("❌ No trained models found!")
        print("\n📚 To get started:")
        print("1. Train with real data: python train_real_model.py")
        print("2. Train with synthetic data: python train_model.py")
        return
    
    # Let user choose model if multiple available
    if len(models) > 1:
        print("\n🎯 Available models:")
        for i, (path, desc) in enumerate(models):
            print(f"{i+1}. {path} - {desc}")
        
        try:
            choice = int(input(f"\nSelect model (1-{len(models)}): ")) - 1
            if 0 <= choice < len(models):
                model_path = models[choice][0]
            else:
                model_path = models[0][0]  # Default to first
        except ValueError:
            model_path = models[0][0]  # Default to first
    else:
        model_path = models[0][0]
    
    # Create predictor and run
    predictor = BodyMeasurementPredictor(model_path)
    
    if predictor.model is not None:
        predictor.run_interactive_session()
    else:
        print("\n📚 To get started:")
        print("1. Run 'python train_real_model.py' to train with real data")
        print("2. Run 'python train_model.py' to train with synthetic data")

if __name__ == "__main__":
    main()