import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import cv2
import mediapipe as mp
from dataclasses import dataclass
import math

@dataclass
class BodyMeasurement:
    """Data class for body measurements"""
    name: str
    value: float
    unit: str
    confidence: float
    method: str

class BodyMeasurements3D:
    """Extract precise body measurements from 3D pose data"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        
        # Standard body proportions for validation
        self.body_proportions = {
            'head_to_height': 0.125,  # Head is ~1/8 of total height
            'shoulder_to_height': 0.25,  # Shoulder width is ~1/4 of height
            'arm_span_to_height': 1.0,  # Arm span ≈ height
            'leg_to_height': 0.5,  # Legs are ~1/2 of height
        }
        
        # Measurement definitions
        self.measurements_config = {
            'height': {
                'landmarks': ['nose', 'left_ankle', 'right_ankle'],
                'method': 'vertical_distance',
                'description': 'Total body height'
            },
            'shoulder_width': {
                'landmarks': ['left_shoulder', 'right_shoulder'],
                'method': 'horizontal_distance',
                'description': 'Shoulder to shoulder width'
            },
            'chest_width': {
                'landmarks': ['left_shoulder', 'right_shoulder'],
                'method': 'chest_circumference',
                'description': 'Chest circumference estimate'
            },
            'waist_width': {
                'landmarks': ['left_hip', 'right_hip'],
                'method': 'waist_circumference',
                'description': 'Waist circumference estimate'
            },
            'hip_width': {
                'landmarks': ['left_hip', 'right_hip'],
                'method': 'hip_circumference',
                'description': 'Hip circumference estimate'
            },
            'arm_length': {
                'landmarks': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'method': 'arm_length_calculation',
                'description': 'Left arm length'
            },
            'leg_length': {
                'landmarks': ['left_hip', 'left_knee', 'left_ankle'],
                'method': 'leg_length_calculation',
                'description': 'Left leg length (inseam)'
            },
            'torso_length': {
                'landmarks': ['nose', 'left_hip', 'right_hip'],
                'method': 'torso_calculation',
                'description': 'Torso length'
            }
        }
    
    def load_pose_data(self, session_dir: str) -> Dict:
        """Load pose landmarks from session directory"""
        pose_file = os.path.join(session_dir, "pose_landmarks.json")
        triangulated_file = os.path.join(session_dir, "triangulated_3d.json")
        
        pose_data = {}
        triangulated_data = {}
        
        if os.path.exists(pose_file):
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
        
        if os.path.exists(triangulated_file):
            with open(triangulated_file, 'r') as f:
                triangulated_data = json.load(f)
        
        return {
            'pose_landmarks': pose_data,
            'triangulated_3d': triangulated_data
        }
    
    def get_landmark_3d_position(self, landmark_name: str, pose_data: Dict) -> Optional[Tuple[float, float, float]]:
        """Get 3D position of a landmark"""
        # Map landmark names to MediaPipe indices
        landmark_map = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        if landmark_name not in landmark_map:
            return None
        
        landmark_idx = landmark_map[landmark_name]
        
        # Try to get from triangulated 3D data first
        if 'triangulated_3d' in pose_data and 'triangulated_points' in pose_data['triangulated_3d']:
            for point in pose_data['triangulated_3d']['triangulated_points']:
                if point['landmark_id'] == landmark_idx:
                    return (point['x'], point['y'], point['z'])
        
        # Fallback to front view 2D data with estimated depth
        if 'pose_landmarks' in pose_data and 'front' in pose_data['pose_landmarks']:
            landmarks = pose_data['pose_landmarks']['front']['landmarks']
            if landmark_idx < len(landmarks):
                landmark = landmarks[landmark_idx]
                # Use z from MediaPipe as depth estimate
                return (landmark['x'], landmark['y'], landmark['z'])
        
        return None
    
    def calculate_distance_3d(self, point1: Tuple[float, float, float], 
                             point2: Tuple[float, float, float]) -> float:
        """Calculate 3D Euclidean distance between two points"""
        return math.sqrt(
            (point1[0] - point2[0])**2 + 
            (point1[1] - point2[1])**2 + 
            (point1[2] - point2[2])**2
        )
    
    def estimate_real_world_scale(self, pose_data: Dict) -> float:
        """Estimate scale factor to convert normalized coordinates to real measurements"""
        # Use known body proportions to estimate scale
        # This is a simplified approach - in practice, you'd use reference objects or camera calibration
        
        # Get shoulder width in normalized coordinates
        left_shoulder = self.get_landmark_3d_position('left_shoulder', pose_data)
        right_shoulder = self.get_landmark_3d_position('right_shoulder', pose_data)
        
        if left_shoulder and right_shoulder:
            shoulder_width_normalized = self.calculate_distance_3d(left_shoulder, right_shoulder)
            
            # Assume average shoulder width is 45cm for adults
            estimated_shoulder_width_cm = 45.0
            scale_factor = estimated_shoulder_width_cm / shoulder_width_normalized
            
            return scale_factor
        
        # Default scale factor if shoulder detection fails
        return 100.0  # Assume 1 normalized unit = 100cm
    
    def calculate_height(self, pose_data: Dict, scale_factor: float) -> BodyMeasurement:
        """Calculate total body height"""
        nose = self.get_landmark_3d_position('nose', pose_data)
        left_ankle = self.get_landmark_3d_position('left_ankle', pose_data)
        right_ankle = self.get_landmark_3d_position('right_ankle', pose_data)
        
        if not (nose and left_ankle and right_ankle):
            return BodyMeasurement("height", 0.0, "cm", 0.0, "failed")
        
        # Use average of both ankles for ground reference
        avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # Height is vertical distance from nose to ground
        height_normalized = abs(nose[1] - avg_ankle_y)
        height_cm = height_normalized * scale_factor
        
        # Confidence based on landmark visibility
        confidence = 0.8  # Simplified confidence score
        
        return BodyMeasurement("height", height_cm, "cm", confidence, "3d_landmarks")
    
    def calculate_shoulder_width(self, pose_data: Dict, scale_factor: float) -> BodyMeasurement:
        """Calculate shoulder width"""
        left_shoulder = self.get_landmark_3d_position('left_shoulder', pose_data)
        right_shoulder = self.get_landmark_3d_position('right_shoulder', pose_data)
        
        if not (left_shoulder and right_shoulder):
            return BodyMeasurement("shoulder_width", 0.0, "cm", 0.0, "failed")
        
        width_normalized = self.calculate_distance_3d(left_shoulder, right_shoulder)
        width_cm = width_normalized * scale_factor
        
        return BodyMeasurement("shoulder_width", width_cm, "cm", 0.9, "3d_landmarks")
    
    def calculate_chest_circumference(self, pose_data: Dict, scale_factor: float) -> BodyMeasurement:
        """Estimate chest circumference from shoulder width"""
        shoulder_width = self.calculate_shoulder_width(pose_data, scale_factor)
        
        if shoulder_width.value == 0:
            return BodyMeasurement("chest_circumference", 0.0, "cm", 0.0, "failed")
        
        # Estimate chest circumference as ~2.5 times shoulder width
        chest_circumference = shoulder_width.value * 2.5
        
        return BodyMeasurement("chest_circumference", chest_circumference, "cm", 0.7, "estimated_from_shoulders")
    
    def calculate_waist_circumference(self, pose_data: Dict, scale_factor: float) -> BodyMeasurement:
        """Estimate waist circumference from hip width"""
        left_hip = self.get_landmark_3d_position('left_hip', pose_data)
        right_hip = self.get_landmark_3d_position('right_hip', pose_data)
        
        if not (left_hip and right_hip):
            return BodyMeasurement("waist_circumference", 0.0, "cm", 0.0, "failed")
        
        hip_width_normalized = self.calculate_distance_3d(left_hip, right_hip)
        hip_width_cm = hip_width_normalized * scale_factor
        
        # Estimate waist as ~85% of hip width, then calculate circumference
        waist_width = hip_width_cm * 0.85
        waist_circumference = waist_width * 2.8  # Approximate circumference from width
        
        return BodyMeasurement("waist_circumference", waist_circumference, "cm", 0.6, "estimated_from_hips")
    
    def calculate_arm_length(self, pose_data: Dict, scale_factor: float) -> BodyMeasurement:
        """Calculate arm length"""
        shoulder = self.get_landmark_3d_position('left_shoulder', pose_data)
        elbow = self.get_landmark_3d_position('left_elbow', pose_data)
        wrist = self.get_landmark_3d_position('left_wrist', pose_data)
        
        if not (shoulder and elbow and wrist):
            return BodyMeasurement("arm_length", 0.0, "cm", 0.0, "failed")
        
        # Calculate upper arm + forearm length
        upper_arm = self.calculate_distance_3d(shoulder, elbow)
        forearm = self.calculate_distance_3d(elbow, wrist)
        
        total_arm_length = (upper_arm + forearm) * scale_factor
        
        return BodyMeasurement("arm_length", total_arm_length, "cm", 0.8, "3d_landmarks")
    
    def calculate_leg_length(self, pose_data: Dict, scale_factor: float) -> BodyMeasurement:
        """Calculate leg length (inseam)"""
        hip = self.get_landmark_3d_position('left_hip', pose_data)
        knee = self.get_landmark_3d_position('left_knee', pose_data)
        ankle = self.get_landmark_3d_position('left_ankle', pose_data)
        
        if not (hip and knee and ankle):
            return BodyMeasurement("leg_length", 0.0, "cm", 0.0, "failed")
        
        # Calculate thigh + shin length
        thigh = self.calculate_distance_3d(hip, knee)
        shin = self.calculate_distance_3d(knee, ankle)
        
        total_leg_length = (thigh + shin) * scale_factor
        
        return BodyMeasurement("leg_length", total_leg_length, "cm", 0.8, "3d_landmarks")
    
    def calculate_all_measurements(self, session_dir: str) -> Dict[str, BodyMeasurement]:
        """Calculate all body measurements from session data"""
        print(f"📏 Calculating body measurements from: {session_dir}")
        
        # Load pose data
        pose_data = self.load_pose_data(session_dir)
        
        if not pose_data['pose_landmarks']:
            print("❌ No pose data found")
            return {}
        
        # Estimate scale factor
        scale_factor = self.estimate_real_world_scale(pose_data)
        print(f"🔍 Estimated scale factor: {scale_factor:.2f} cm/unit")
        
        # Calculate measurements
        measurements = {}
        
        measurements['height'] = self.calculate_height(pose_data, scale_factor)
        measurements['shoulder_width'] = self.calculate_shoulder_width(pose_data, scale_factor)
        measurements['chest_circumference'] = self.calculate_chest_circumference(pose_data, scale_factor)
        measurements['waist_circumference'] = self.calculate_waist_circumference(pose_data, scale_factor)
        measurements['arm_length'] = self.calculate_arm_length(pose_data, scale_factor)
        measurements['leg_length'] = self.calculate_leg_length(pose_data, scale_factor)
        
        # Validate measurements using body proportions
        measurements = self.validate_measurements(measurements)
        
        return measurements
    
    def validate_measurements(self, measurements: Dict[str, BodyMeasurement]) -> Dict[str, BodyMeasurement]:
        """Validate measurements using known body proportions"""
        if 'height' not in measurements or measurements['height'].value == 0:
            return measurements
        
        height = measurements['height'].value
        
        # Validate shoulder width (should be ~25% of height)
        if 'shoulder_width' in measurements:
            expected_shoulder = height * 0.25
            actual_shoulder = measurements['shoulder_width'].value
            
            if abs(actual_shoulder - expected_shoulder) / expected_shoulder > 0.5:
                # Adjust confidence if measurement seems unrealistic
                measurements['shoulder_width'].confidence *= 0.5
        
        # Validate arm length (should be ~40% of height)
        if 'arm_length' in measurements:
            expected_arm = height * 0.4
            actual_arm = measurements['arm_length'].value
            
            if abs(actual_arm - expected_arm) / expected_arm > 0.5:
                measurements['arm_length'].confidence *= 0.5
        
        return measurements
    
    def save_measurements(self, measurements: Dict[str, BodyMeasurement], session_dir: str):
        """Save measurements to JSON file"""
        measurements_data = {}
        
        for name, measurement in measurements.items():
            measurements_data[name] = {
                'value': round(measurement.value, 1),
                'unit': measurement.unit,
                'confidence': round(measurement.confidence, 2),
                'method': measurement.method
            }
        
        # Add metadata
        measurements_data['metadata'] = {
            'session_dir': session_dir,
            'calculation_method': '3d_pose_estimation',
            'timestamp': json.dumps(None, default=str),  # Current timestamp
            'total_measurements': len(measurements)
        }
        
        output_file = os.path.join(session_dir, "body_measurements.json")
        with open(output_file, 'w') as f:
            json.dump(measurements_data, f, indent=2)
        
        print(f"💾 Measurements saved: {output_file}")
    
    def print_measurements_report(self, measurements: Dict[str, BodyMeasurement]):
        """Print formatted measurements report"""
        print("\n" + "="*60)
        print("📏 BODY MEASUREMENTS REPORT")
        print("="*60)
        
        for name, measurement in measurements.items():
            confidence_stars = "★" * int(measurement.confidence * 5)
            print(f"{name.replace('_', ' ').title():<20}: {measurement.value:6.1f} {measurement.unit} {confidence_stars}")
        
        print("="*60)
        print("Confidence: ★★★★★ = Very High, ★ = Low")
        print("Note: Measurements are estimates based on pose detection")

def main():
    """Main function for body measurements calculation"""
    calculator = BodyMeasurements3D()
    
    print("📏 3D Body Measurements Calculator")
    session_dir = input("Enter session directory path: ").strip()
    
    if not os.path.exists(session_dir):
        print("❌ Session directory not found")
        return
    
    try:
        # Calculate measurements
        measurements = calculator.calculate_all_measurements(session_dir)
        
        if measurements:
            # Print report
            calculator.print_measurements_report(measurements)
            
            # Save measurements
            calculator.save_measurements(measurements, session_dir)
            
            print(f"\n✅ Measurements calculation completed!")
            print(f"📁 Results saved in: {session_dir}")
        else:
            print("❌ Failed to calculate measurements")
    
    except Exception as e:
        print(f"❌ Error calculating measurements: {e}")

if __name__ == "__main__":
    main()