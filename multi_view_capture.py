import cv2
import numpy as np
import os
import time
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

class MultiViewCapture:
    """Capture multiple views (front, left, right, back) for 3D body reconstruction"""
    def __init__(self, output_dir="captures"):
        self.output_dir = output_dir
        self.views = ["front", "left", "right", "back"]
        self.current_view_index = 0
        self.captured_images = {}
        self.session_id = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def create_session(self):
        """Create a new capture session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{timestamp}"
        session_dir = os.path.join(self.output_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        print(f"📸 New capture session created: {self.session_id}")
        return session_dir
    
    def get_capture_instructions(self, view: str) -> str:
        """Get instructions for each view"""
        instructions = {
            "front": "Stand facing the camera with arms slightly away from body",
            "left": "Turn 90° left, show your left side profile",
            "right": "Turn 90° right, show your right side profile", 
            "back": "Turn around, show your back to the camera"
        }
        return instructions.get(view, "Position yourself as instructed")
    
    def capture_view_with_countdown(self, cap, view: str, countdown_seconds=5):
        """Capture a specific view with countdown and instructions"""
        print(f"\n📸 Capturing {view.upper()} view")
        print(f"📋 Instructions: {self.get_capture_instructions(view)}")
        
        # Countdown with live preview
        for i in range(countdown_seconds, 0, -1):
            ret, frame = cap.read()
            if not ret:
                return None
            
            # Add overlay text
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # View name
            cv2.putText(frame, f"{view.upper()} VIEW", (50, 50), 
                       font, 1.2, (0, 255, 255), 2)
            
            # Instructions
            instruction = self.get_capture_instructions(view)
            y_offset = 100
            for line in self._wrap_text(instruction, 50):
                cv2.putText(frame, line, (50, y_offset), 
                           font, 0.6, (255, 255, 255), 1)
                y_offset += 25
            
            # Countdown
            countdown_text = str(i)
            text_size = cv2.getTextSize(countdown_text, font, 3, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            cv2.putText(frame, countdown_text, (text_x, text_y), 
                       font, 3, (0, 255, 0), 3)
            
            # Progress indicator
            progress = (len(self.captured_images) / len(self.views)) * 100
            cv2.putText(frame, f"Progress: {progress:.0f}%", 
                       (frame.shape[1] - 200, 50), font, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Multi-View Body Capture', frame)
            cv2.waitKey(1000)
        
        # Capture final image
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "CAPTURED!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Multi-View Body Capture', frame)
            cv2.waitKey(1500)
            
            return frame
        
        return None
    
    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        """Wrap text for display"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def validate_image_quality(self, image: np.ndarray) -> Tuple[bool, str]:
        """Validate image quality for 3D reconstruction"""
        if image is None:
            return False, "No image captured"
        
        # Check image dimensions
        height, width = image.shape[:2]
        if height < 480 or width < 640:
            return False, "Image resolution too low (minimum 640x480)"
        
        # Check brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 50:
            return False, "Image too dark"
        if mean_brightness > 200:
            return False, "Image too bright"
        
        # Check blur (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 100:
            return False, "Image too blurry"
        
        return True, "Image quality acceptable"
    
    def save_captured_image(self, image: np.ndarray, view: str) -> str:
        """Save captured image with metadata"""
        if self.session_id is None:
            raise ValueError("No active session. Call create_session() first.")
        
        session_dir = os.path.join(self.output_dir, self.session_id)
        filename = f"{view}_view.jpg"
        filepath = os.path.join(session_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, image)
        
        # Save metadata
        metadata = {
            "view": view,
            "timestamp": datetime.now().isoformat(),
            "image_shape": image.shape,
            "filename": filename
        }
        
        metadata_file = os.path.join(session_dir, f"{view}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved {view} view: {filepath}")
        return filepath
    
    def run_capture_session(self):
        """Run complete multi-view capture session"""
        print("🎯 3D BODY MEASUREMENT - MULTI-VIEW CAPTURE")
        print("=" * 60)
        print("📋 You will capture 4 views: Front, Left, Right, Back")
        print("📏 Stand 6-8 feet from camera with good lighting")
        print("🎽 Wear form-fitting clothes for best results")
        print("🏠 Use a plain background")
        print("=" * 60)
        
        # Create session
        session_dir = self.create_session()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera!")
            return None
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            for view in self.views:
                print(f"\n🔄 Preparing to capture {view.upper()} view...")
                input("Press Enter when ready...")
                
                # Capture image
                captured_image = self.capture_view_with_countdown(cap, view)
                
                if captured_image is not None:
                    # Validate image quality
                    is_valid, message = self.validate_image_quality(captured_image)
                    
                    if is_valid:
                        # Save image
                        filepath = self.save_captured_image(captured_image, view)
                        self.captured_images[view] = filepath
                        print(f"✓ {view.upper()} view captured successfully!")
                    else:
                        print(f"❌ Image quality issue: {message}")
                        retry = input("Retry this view? (y/n): ").lower()
                        if retry == 'y':
                            # Retry current view
                            continue
                        else:
                            print("⚠️ Skipping this view...")
                else:
                    print(f"❌ Failed to capture {view} view")
                    retry = input("Retry this view? (y/n): ").lower()
                    if retry == 'y':
                        continue
            
            # Save session summary
            self.save_session_summary(session_dir)
            
            print(f"\n🎉 Capture session completed!")
            print(f"📁 Images saved in: {session_dir}")
            print(f"📸 Captured views: {list(self.captured_images.keys())}")
            
            return session_dir
            
        except KeyboardInterrupt:
            print("\n⚠️ Capture session interrupted by user")
            return None
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def save_session_summary(self, session_dir: str):
        """Save session summary with all captured images"""
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "captured_views": list(self.captured_images.keys()),
            "total_images": len(self.captured_images),
            "image_paths": self.captured_images,
            "status": "completed" if len(self.captured_images) == 4 else "partial"
        }
        
        summary_file = os.path.join(session_dir, "session_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Session summary saved: {summary_file}")
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load a previous capture session"""
        session_dir = os.path.join(self.output_dir, session_id)
        summary_file = os.path.join(session_dir, "session_summary.json")
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                return json.load(f)
        return None
    
    def list_sessions(self) -> List[str]:
        """List all available capture sessions"""
        sessions = []
        if os.path.exists(self.output_dir):
            for item in os.listdir(self.output_dir):
                if item.startswith("session_"):
                    sessions.append(item)
        return sorted(sessions)

def main():
    """Main function for multi-view capture"""
    capture = MultiViewCapture()
    
    print("🎯 3D Body Measurement - Multi-View Capture System")
    print("1. Start new capture session")
    print("2. List previous sessions")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        session_dir = capture.run_capture_session()
        if session_dir:
            print(f"\n✅ Session completed successfully!")
            print(f"📁 Files saved in: {session_dir}")
            print("🔄 Next step: Run 3D reconstruction with these images")
    
    elif choice == "2":
        sessions = capture.list_sessions()
        if sessions:
            print("\n📁 Previous sessions:")
            for i, session in enumerate(sessions, 1):
                print(f"{i}. {session}")
        else:
            print("📭 No previous sessions found")
    
    elif choice == "3":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()