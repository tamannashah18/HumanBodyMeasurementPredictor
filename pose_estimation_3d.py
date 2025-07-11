import cv2
import numpy as np
import mediapipe as mp
import json
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PoseEstimation3D:
    """3D pose estimation and keypoint extraction using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose estimation
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define body measurement landmarks
        self.measurement_landmarks = {
            'head_top': [mp.solutions.pose.PoseLandmark.NOSE],
            'shoulders': [
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
            ],
            'chest': [
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
            ],
            'waist': [
                mp.solutions.pose.PoseLandmark.LEFT_HIP,
                mp.solutions.pose.PoseLandmark.RIGHT_HIP
            ],
            'hips': [
                mp.solutions.pose.PoseLandmark.LEFT_HIP,
                mp.solutions.pose.PoseLandmark.RIGHT_HIP
            ],
            'arms': [
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                mp.solutions.pose.PoseLandmark.RIGHT_WRIST
            ],
            'legs': [
                mp.solutions.pose.PoseLandmark.LEFT_HIP,
                mp.solutions.pose.PoseLandmark.LEFT_KNEE,
                mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
                mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
                mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
            ]
        }
    
    def extract_pose_landmarks(self, image_path: str) -> Optional[Dict]:
        """Extract pose landmarks from image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            print(f"❌ No pose detected in: {image_path}")
            return None
        
        # Extract landmark coordinates
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return {
            'landmarks': landmarks,
            'image_shape': image.shape,
            'pose_detected': True
        }
    
    def process_multi_view_session(self, session_dir: str) -> Dict:
        """Process all views in a capture session"""
        print(f"🔍 Processing multi-view session: {session_dir}")
        
        # Load session summary
        summary_file = os.path.join(session_dir, "session_summary.json")
        if not os.path.exists(summary_file):
            raise FileNotFoundError(f"Session summary not found: {summary_file}")
        
        with open(summary_file, 'r') as f:
            session_data = json.load(f)
        
        # Process each view
        pose_data = {}
        for view, image_path in session_data['image_paths'].items():
            print(f"📸 Processing {view} view...")
            
            full_path = image_path if os.path.isabs(image_path) else os.path.join(session_dir, os.path.basename(image_path))
            landmarks = self.extract_pose_landmarks(full_path)
            
            if landmarks:
                pose_data[view] = landmarks
                print(f"✓ {view} view processed successfully")
            else:
                print(f"❌ Failed to process {view} view")
        
        # Save pose data
        pose_file = os.path.join(session_dir, "pose_landmarks.json")
        with open(pose_file, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        print(f"💾 Pose data saved: {pose_file}")
        return pose_data
    
    def triangulate_3d_points(self, pose_data: Dict) -> Dict:
        """Triangulate 3D points from multiple views"""
        print("🔺 Triangulating 3D points from multiple views...")
        
        # For simplified triangulation, we'll use front and side views
        if 'front' not in pose_data or 'left' not in pose_data:
            print("⚠️ Need at least front and left views for 3D triangulation")
            return {}
        
        front_landmarks = pose_data['front']['landmarks']
        side_landmarks = pose_data['left']['landmarks']
        
        # Simple triangulation assuming known camera positions
        # In a real implementation, you'd use proper camera calibration
        triangulated_points = []
        
        for i, (front_pt, side_pt) in enumerate(zip(front_landmarks, side_landmarks)):
            if front_pt['visibility'] > 0.5 and side_pt['visibility'] > 0.5:
                # Simple triangulation (this is a simplified approach)
                x = side_pt['z']  # Depth from side view
                y = front_pt['y']  # Height from front view
                z = front_pt['x']  # Width from front view
                
                triangulated_points.append({
                    'landmark_id': i,
                    'x': x,
                    'y': y,
                    'z': z,
                    'confidence': min(front_pt['visibility'], side_pt['visibility'])
                })
        
        return {
            'triangulated_points': triangulated_points,
            'method': 'simplified_triangulation',
            'views_used': ['front', 'left']
        }
    
    def visualize_pose_landmarks(self, image_path: str, output_path: str = None):
        """Visualize pose landmarks on image"""
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            # Draw landmarks
            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            if output_path:
                cv2.imwrite(output_path, annotated_image)
                print(f"✓ Annotated image saved: {output_path}")
            
            return True
        
        return False
    
    def create_3d_visualization(self, triangulated_data: Dict, output_path: str = None):
        """Create 3D visualization of triangulated points"""
        if not triangulated_data or 'triangulated_points' not in triangulated_data:
            print("❌ No triangulated data available for visualization")
            return False
        
        points = triangulated_data['triangulated_points']
        
        # Extract coordinates
        x_coords = [p['x'] for p in points]
        y_coords = [p['y'] for p in points]
        z_coords = [p['z'] for p in points]
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        scatter = ax.scatter(x_coords, y_coords, z_coords, 
                           c=[p['confidence'] for p in points], 
                           cmap='viridis', s=50)
        
        # Add colorbar
        plt.colorbar(scatter, label='Confidence')
        
        # Set labels
        ax.set_xlabel('X (Width)')
        ax.set_ylabel('Y (Height)')
        ax.set_zlabel('Z (Depth)')
        ax.set_title('3D Body Pose Reconstruction')
        
        # Set equal aspect ratio
        max_range = max(max(x_coords) - min(x_coords),
                       max(y_coords) - min(y_coords),
                       max(z_coords) - min(z_coords)) / 2.0
        
        mid_x = (max(x_coords) + min(x_coords)) * 0.5
        mid_y = (max(y_coords) + min(y_coords)) * 0.5
        mid_z = (max(z_coords) + min(z_coords)) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ 3D visualization saved: {output_path}")
        
        plt.show()
        return True

def main():
    """Main function for pose estimation"""
    pose_estimator = PoseEstimation3D()
    
    print("🎯 3D Pose Estimation and Reconstruction")
    print("1. Process multi-view session")
    print("2. Visualize single image pose")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        session_dir = input("Enter session directory path: ").strip()
        if os.path.exists(session_dir):
            try:
                # Process multi-view session
                pose_data = pose_estimator.process_multi_view_session(session_dir)
                
                # Triangulate 3D points
                triangulated_data = pose_estimator.triangulate_3d_points(pose_data)
                
                if triangulated_data:
                    # Save triangulated data
                    triangulation_file = os.path.join(session_dir, "triangulated_3d.json")
                    with open(triangulation_file, 'w') as f:
                        json.dump(triangulated_data, f, indent=2)
                    
                    # Create 3D visualization
                    viz_path = os.path.join(session_dir, "3d_visualization.png")
                    pose_estimator.create_3d_visualization(triangulated_data, viz_path)
                    
                    print(f"✅ 3D reconstruction completed!")
                    print(f"📁 Results saved in: {session_dir}")
                
            except Exception as e:
                print(f"❌ Error processing session: {e}")
        else:
            print("❌ Session directory not found")
    
    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            output_path = image_path.replace('.jpg', '_pose.jpg').replace('.png', '_pose.png')
            success = pose_estimator.visualize_pose_landmarks(image_path, output_path)
            if success:
                print(f"✅ Pose visualization saved: {output_path}")
            else:
                print("❌ Failed to detect pose in image")
        else:
            print("❌ Image file not found")
    
    elif choice == "3":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()