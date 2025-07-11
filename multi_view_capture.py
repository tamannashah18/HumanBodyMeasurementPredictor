import cv2
import numpy as np
import os
import time
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import mediapipe as mp

class MultiViewCapture:
    """Capture multiple views (front, left, right, back) for 3D body reconstruction"""
    def __init__(self, output_dir="captures"):
        self.output_dir = output_dir
        self.views = ["front", "left", "right", "back"]
        self.current_view_index = 0
        self.captured_images = {}
        self.session_id = None
        
        # Initialize MediaPipe pose detection for auto-detection
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Pose detection parameters
        self.pose_stable_frames = 0
        self.required_stable_frames = 15  # ~0.5 seconds at 30fps
        self.pose_detected = False
        
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
    
    def detect_pose_quality(self, frame: np.ndarray, view: str) -> Tuple[bool, str, float]:
        """
        Detect if person is in correct pose for the given view
        Returns: (pose_detected, feedback_message, confidence)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return False, "No person detected - step into frame", 0.0
        
        landmarks = results.pose_landmarks.landmark
        
        # Get key landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Calculate visibility scores
        key_landmarks = [nose, left_shoulder, right_shoulder, left_hip, right_hip]
        avg_visibility = sum(lm.visibility for lm in key_landmarks) / len(key_landmarks)
        
        if avg_visibility < 0.7:
            return False, "Stand more clearly in view", avg_visibility
        
        # Check pose based on view
        if view == "front":
            return self._check_front_pose(landmarks)
        elif view == "left":
            return self._check_side_pose(landmarks, "left")
        elif view == "right":
            return self._check_side_pose(landmarks, "right")
        elif view == "back":
            return self._check_back_pose(landmarks)
        
        return False, "Unknown view", 0.0
    
    def _check_front_pose(self, landmarks) -> Tuple[bool, str, float]:
        """Check if person is facing front correctly"""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Check if shoulders are roughly level (front-facing)
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_diff > 0.05:
            return False, "Keep shoulders level", 0.5
        
        # Check if both shoulders are visible (front view)
        if left_shoulder.visibility < 0.7 or right_shoulder.visibility < 0.7:
            return False, "Face the camera directly", 0.5
        
        # Check if arms are away from body
        torso_width = abs(left_shoulder.x - right_shoulder.x)
        left_arm_distance = abs(left_wrist.x - left_shoulder.x)
        right_arm_distance = abs(right_wrist.x - right_shoulder.x)
        
        if left_arm_distance < torso_width * 0.3 or right_arm_distance < torso_width * 0.3:
            return False, "Move arms away from body", 0.7
        
        # Check if person is upright
        if abs(nose.y - (left_shoulder.y + right_shoulder.y) / 2) < 0.1:
            return False, "Stand up straight", 0.6
        
        return True, "Perfect! Hold this pose", 0.9
    
    def _check_side_pose(self, landmarks, side: str) -> Tuple[bool, str, float]:
        """Check if person is in correct side profile"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        
        if side == "left":
            # For left side view, left shoulder should be more visible
            if left_shoulder.visibility < 0.8:
                return False, "Turn more to show left side", 0.4
            if right_shoulder.visibility > 0.5:
                return False, "Turn more left - hide right shoulder", 0.5
        else:  # right side
            # For right side view, right shoulder should be more visible
            if right_shoulder.visibility < 0.8:
                return False, "Turn more to show right side", 0.4
            if left_shoulder.visibility > 0.5:
                return False, "Turn more right - hide left shoulder", 0.5
        
        # Check if nose is in profile (not facing camera)
        if nose.visibility > 0.9:
            return False, f"Turn more to {side} - show profile", 0.6
        
        return True, "Perfect! Hold this pose", 0.9
    
    def _check_back_pose(self, landmarks) -> Tuple[bool, str, float]:
        """Check if person is showing their back"""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # For back view, nose should not be very visible
        if nose.visibility > 0.7:
            return False, "Turn around to show your back", 0.3
        
        # Shoulders should be visible but nose should not be
        if left_shoulder.visibility < 0.6 or right_shoulder.visibility < 0.6:
            return False, "Stand straighter - show your back", 0.5
        
        return True, "Perfect! Hold this pose", 0.9
    
    def draw_pose_feedback(self, frame: np.ndarray, pose_detected: bool, 
                          feedback: str, confidence: float, view: str) -> np.ndarray:
        """Draw pose detection feedback on frame"""
        overlay = frame.copy()
        
        # Draw pose detection status
        status_color = (0, 255, 0) if pose_detected else (0, 165, 255)
        status_text = "✓ POSE DETECTED" if pose_detected else "⚠ ADJUST POSE"
        
        cv2.putText(overlay, status_text, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Draw feedback message
        cv2.putText(overlay, feedback, (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = 50
        bar_y = 120
        
        # Background bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Confidence bar
        conf_width = int(bar_width * confidence)
        conf_color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                     conf_color, -1)
        
        # Confidence text
        cv2.putText(overlay, f"Confidence: {confidence:.1%}", (bar_x, bar_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # View instructions
        instruction = self.get_capture_instructions(view)
        y_offset = 180
        for line in self._wrap_text(instruction, 50):
            cv2.putText(overlay, line, (50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_offset += 25
        
        return overlay
    
    def capture_view_with_auto_detection(self, cap, view: str, max_wait_time=30):
        """Capture view with automatic pose detection and smart timer"""
        print(f"\n📸 Preparing to capture {view.upper()} view")
        print(f"📋 Instructions: {self.get_capture_instructions(view)}")
        print("🤖 Auto-detecting pose... Position yourself and hold still")
        
        start_time = time.time()
        pose_stable_frames = 0
        countdown_started = False
        countdown_start_time = 0
        countdown_duration = 3  # 3 second countdown after pose detected
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                return None
            
            # Check if we've been waiting too long
            if time.time() - start_time > max_wait_time:
                print(f"⏰ Timeout after {max_wait_time} seconds")
                retry = input("Continue waiting or capture anyway? (w/c): ").lower()
                if retry == 'c':
                    return frame
                else:
                    start_time = time.time()  # Reset timer
            
            # Detect pose quality
            pose_detected, feedback, confidence = self.detect_pose_quality(frame, view)
            
            # Draw feedback overlay
            display_frame = self.draw_pose_feedback(frame, pose_detected, feedback, confidence, view)
            
            if pose_detected and confidence > 0.8:
                pose_stable_frames += 1
                
                if pose_stable_frames >= self.required_stable_frames and not countdown_started:
                    print("✅ Perfect pose detected! Starting countdown...")
                    countdown_started = True
                    countdown_start_time = time.time()
                
                if countdown_started:
                    elapsed = time.time() - countdown_start_time
                    remaining = countdown_duration - elapsed
                    
                    if remaining > 0:
                        # Draw countdown
                        countdown_text = str(int(remaining) + 1)
                        text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 4)[0]
                        text_x = (display_frame.shape[1] - text_size[0]) // 2
                        text_y = (display_frame.shape[0] + text_size[1]) // 2
                        
                        cv2.putText(display_frame, countdown_text, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
                        cv2.putText(display_frame, "Hold still!", (text_x - 100, text_y + 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        # Capture!
                        cv2.putText(display_frame, "CAPTURED!", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        cv2.imshow('Multi-View Body Capture', display_frame)
                        cv2.waitKey(1000)
                        return frame
            else:
                # Reset if pose is lost
                pose_stable_frames = 0
                countdown_started = False
            
            # Add progress indicator
            progress = (len(self.captured_images) / len(self.views)) * 100
            cv2.putText(display_frame, f"Progress: {progress:.0f}%", 
                       (display_frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Add manual capture option
            cv2.putText(display_frame, "Press 'c' to capture manually, 'q' to quit", 
                       (50, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('Multi-View Body Capture', display_frame)
            
            # Handle manual controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                print("📸 Manual capture triggered")
                return frame
            elif key == ord('q'):
                print("❌ Capture cancelled by user")
                return None
        
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
                print("🤖 Position yourself according to instructions...")
                print("⏱️  The system will automatically detect when you're ready!")
                
                # Capture image
                captured_image = self.capture_view_with_auto_detection(cap, view)
                
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