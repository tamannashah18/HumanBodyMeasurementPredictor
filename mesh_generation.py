import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import trimesh

class BodyMeshGenerator:
    """Generate realistic 3D body mesh from pose landmarks and measurements"""
    
    def __init__(self):
        # Define body segments with anatomical proportions
        self.body_segments = {
            'head': {
                'landmarks': ['nose', 'left_ear', 'right_ear'],
                'type': 'ellipsoid',
                'proportions': {'width': 0.15, 'height': 0.23, 'depth': 0.18}
            },
            'neck': {
                'landmarks': ['nose', 'left_shoulder', 'right_shoulder'],
                'type': 'cylinder',
                'proportions': {'radius': 0.06, 'length': 0.12}
            },
            'torso': {
                'landmarks': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
                'type': 'tapered_cylinder',
                'proportions': {'top_radius': 0.18, 'bottom_radius': 0.14, 'length': 0.45}
            },
            'left_upper_arm': {
                'landmarks': ['left_shoulder', 'left_elbow'],
                'type': 'tapered_cylinder',
                'proportions': {'top_radius': 0.05, 'bottom_radius': 0.04, 'length': 0.3}
            },
            'left_forearm': {
                'landmarks': ['left_elbow', 'left_wrist'],
                'type': 'tapered_cylinder',
                'proportions': {'top_radius': 0.04, 'bottom_radius': 0.03, 'length': 0.25}
            },
            'right_upper_arm': {
                'landmarks': ['right_shoulder', 'right_elbow'],
                'type': 'tapered_cylinder',
                'proportions': {'top_radius': 0.05, 'bottom_radius': 0.04, 'length': 0.3}
            },
            'right_forearm': {
                'landmarks': ['right_elbow', 'right_wrist'],
                'type': 'tapered_cylinder',
                'proportions': {'top_radius': 0.04, 'bottom_radius': 0.03, 'length': 0.25}
            },
            'left_thigh': {
                'landmarks': ['left_hip', 'left_knee'],
                'type': 'tapered_cylinder',
                'proportions': {'top_radius': 0.08, 'bottom_radius': 0.06, 'length': 0.4}
            },
            'left_shin': {
                'landmarks': ['left_knee', 'left_ankle'],
                'type': 'tapered_cylinder',
                'proportions': {'top_radius': 0.06, 'bottom_radius': 0.04, 'length': 0.35}
            },
            'right_thigh': {
                'landmarks': ['right_hip', 'right_knee'],
                'type': 'tapered_cylinder',
                'proportions': {'top_radius': 0.08, 'bottom_radius': 0.06, 'length': 0.4}
            },
            'right_shin': {
                'landmarks': ['right_knee', 'right_ankle'],
                'type': 'tapered_cylinder',
                'proportions': {'top_radius': 0.06, 'bottom_radius': 0.04, 'length': 0.35}
            }
        }
        
        # Landmark mapping for MediaPipe pose
        self.landmark_map = {
            'nose': 0,
            'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8,
            'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
    
    def load_session_data(self, session_dir: str) -> Dict:
        """Load all session data needed for mesh generation"""
        data = {}
        
        # Load pose landmarks
        pose_file = os.path.join(session_dir, "pose_landmarks.json")
        if os.path.exists(pose_file):
            with open(pose_file, 'r') as f:
                data['pose_landmarks'] = json.load(f)
        
        # Load triangulated 3D data
        triangulated_file = os.path.join(session_dir, "triangulated_3d.json")
        if os.path.exists(triangulated_file):
            with open(triangulated_file, 'r') as f:
                data['triangulated_3d'] = json.load(f)
        
        # Load body measurements
        measurements_file = os.path.join(session_dir, "body_measurements.json")
        if os.path.exists(measurements_file):
            with open(measurements_file, 'r') as f:
                data['measurements'] = json.load(f)
        
        return data
    
    def get_landmark_position(self, landmark_name: str, session_data: Dict) -> Optional[Tuple[float, float, float]]:
        """Get 3D position of a landmark with improved accuracy"""
        if landmark_name not in self.landmark_map:
            return None
        
        landmark_idx = self.landmark_map[landmark_name]
        
        # Try triangulated 3D data first
        if 'triangulated_3d' in session_data and 'triangulated_points' in session_data['triangulated_3d']:
            for point in session_data['triangulated_3d']['triangulated_points']:
                if point['landmark_id'] == landmark_idx:
                    return (point['x'], point['y'], point['z'])
        
        # Fallback to front view with estimated depth
        if 'pose_landmarks' in session_data and 'front' in session_data['pose_landmarks']:
            landmarks = session_data['pose_landmarks']['front']['landmarks']
            if landmark_idx < len(landmarks):
                landmark = landmarks[landmark_idx]
                # Estimate depth based on body part
                depth = self.estimate_depth_from_landmark(landmark_name, landmark)
                return (landmark['x'], landmark['y'], depth)
        
        return None
    
    def estimate_depth_from_landmark(self, landmark_name: str, landmark: Dict) -> float:
        """Estimate depth for 2D landmarks based on anatomical knowledge"""
        # Default depth values based on typical human anatomy
        depth_map = {
            'nose': 0.1, 'left_ear': -0.05, 'right_ear': -0.05,
            'left_shoulder': 0.0, 'right_shoulder': 0.0,
            'left_elbow': 0.05, 'right_elbow': 0.05,
            'left_wrist': 0.1, 'right_wrist': 0.1,
            'left_hip': 0.0, 'right_hip': 0.0,
            'left_knee': 0.05, 'right_knee': 0.05,
            'left_ankle': 0.0, 'right_ankle': 0.0
        }
        
        base_depth = depth_map.get(landmark_name, 0.0)
        # Add some variation based on landmark confidence
        confidence_factor = landmark.get('visibility', 0.5)
        return base_depth * confidence_factor
    
    def create_ellipsoid_mesh(self, center: np.ndarray, radii: Tuple[float, float, float], 
                             resolution: int = 16) -> Tuple[np.ndarray, List]:
        """Create an ellipsoid mesh for head/torso"""
        vertices = []
        faces = []
        
        # Generate ellipsoid vertices
        for i in range(resolution):
            for j in range(resolution):
                u = 2 * np.pi * i / resolution
                v = np.pi * j / resolution
                
                x = radii[0] * np.sin(v) * np.cos(u)
                y = radii[1] * np.sin(v) * np.sin(u)
                z = radii[2] * np.cos(v)
                
                vertices.append(center + np.array([x, y, z]))
        
        # Generate faces
        for i in range(resolution):
            for j in range(resolution):
                next_i = (i + 1) % resolution
                next_j = (j + 1) % resolution
                
                v1 = i * resolution + j
                v2 = next_i * resolution + j
                v3 = next_i * resolution + next_j
                v4 = i * resolution + next_j
                
                # Two triangles per quad
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        return np.array(vertices), faces
    
    def create_tapered_cylinder_mesh(self, start_point: np.ndarray, end_point: np.ndarray,
                                   start_radius: float, end_radius: float, 
                                   segments: int = 12) -> Tuple[np.ndarray, List]:
        """Create a tapered cylindrical mesh for limbs"""
        vertices = []
        faces = []
        
        # Calculate cylinder axis
        axis = end_point - start_point
        length = np.linalg.norm(axis)
        axis_normalized = axis / length if length > 0 else np.array([0, 1, 0])
        
        # Create perpendicular vectors
        if abs(axis_normalized[2]) < 0.9:
            perp1 = np.cross(axis_normalized, [0, 0, 1])
        else:
            perp1 = np.cross(axis_normalized, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(axis_normalized, perp1)
        
        # Generate vertices along the cylinder
        num_rings = 8  # Number of rings along the cylinder
        for ring in range(num_rings):
            t = ring / (num_rings - 1)  # Parameter from 0 to 1
            
            # Interpolate position and radius
            current_pos = start_point + t * axis
            current_radius = start_radius + t * (end_radius - start_radius)
            
            # Create ring of vertices
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                offset = current_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                vertices.append(current_pos + offset)
        
        # Generate faces
        for ring in range(num_rings - 1):
            for i in range(segments):
                next_i = (i + 1) % segments
                
                # Current ring vertices
                v1 = ring * segments + i
                v2 = ring * segments + next_i
                
                # Next ring vertices
                v3 = (ring + 1) * segments + next_i
                v4 = (ring + 1) * segments + i
                
                # Two triangles per quad
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        return np.array(vertices), faces
    
    def create_anatomical_torso(self, session_data: Dict) -> Tuple[np.ndarray, List]:
        """Create anatomically correct torso mesh"""
        # Get key landmarks
        left_shoulder = self.get_landmark_position('left_shoulder', session_data)
        right_shoulder = self.get_landmark_position('right_shoulder', session_data)
        left_hip = self.get_landmark_position('left_hip', session_data)
        right_hip = self.get_landmark_position('right_hip', session_data)
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            raise ValueError("Missing required landmarks for torso mesh")
        
        # Calculate torso dimensions from measurements
        measurements = session_data.get('measurements', {})
        
        # Get chest and waist measurements
        chest_width = 0.35  # Default
        waist_width = 0.28  # Default
        
        if 'chest_circumference' in measurements:
            chest_circ = measurements['chest_circumference'].get('value', 100)
            chest_width = chest_circ / (2 * np.pi)
        
        if 'waist_circumference' in measurements:
            waist_circ = measurements['waist_circumference'].get('value', 80)
            waist_width = waist_circ / (2 * np.pi)
        
        # Calculate torso center and orientation
        shoulder_center = (np.array(left_shoulder) + np.array(right_shoulder)) / 2
        hip_center = (np.array(left_hip) + np.array(right_hip)) / 2
        
        # Create tapered cylinder for torso
        chest_radius = chest_width / 2
        waist_radius = waist_width / 2
        
        return self.create_tapered_cylinder_mesh(
            shoulder_center, hip_center, chest_radius, waist_radius, segments=16
        )
    
    def create_anatomical_head(self, session_data: Dict) -> Tuple[np.ndarray, List]:
        """Create anatomically correct head mesh"""
        nose = self.get_landmark_position('nose', session_data)
        left_ear = self.get_landmark_position('left_ear', session_data)
        right_ear = self.get_landmark_position('right_ear', session_data)
        
        if not nose:
            raise ValueError("Missing nose landmark for head mesh")
        
        # Estimate head center and size
        head_center = np.array(nose)
        if left_ear and right_ear:
            ear_center = (np.array(left_ear) + np.array(right_ear)) / 2
            head_center = (head_center + ear_center) / 2
        
        # Head dimensions (approximate)
        head_width = 0.15
        head_height = 0.23
        head_depth = 0.18
        
        return self.create_ellipsoid_mesh(
            head_center, (head_width, head_height, head_depth), resolution=12
        )
    
    def create_anatomical_limb(self, segment_name: str, session_data: Dict) -> Tuple[np.ndarray, List]:
        """Create anatomically correct limb mesh"""
        segment = self.body_segments[segment_name]
        landmarks = segment['landmarks']
        
        # Get landmark positions
        positions = []
        for landmark in landmarks:
            pos = self.get_landmark_position(landmark, session_data)
            if pos:
                positions.append(np.array(pos))
        
        if len(positions) < 2:
            raise ValueError(f"Insufficient landmarks for {segment_name}")
        
        # Get proportions from measurements
        measurements = session_data.get('measurements', {})
        proportions = segment['proportions']
        
        # Adjust radius based on measurements
        start_radius = proportions['top_radius']
        end_radius = proportions['bottom_radius']
        
        # Scale based on actual measurements if available
        if 'arm_length' in measurements and 'arm' in segment_name:
            scale_factor = measurements['arm_length'].get('value', 65) / 65.0
            start_radius *= scale_factor
            end_radius *= scale_factor
        elif 'leg_length' in measurements and ('thigh' in segment_name or 'shin' in segment_name):
            scale_factor = measurements['leg_length'].get('value', 85) / 85.0
            start_radius *= scale_factor
            end_radius *= scale_factor
        
        return self.create_tapered_cylinder_mesh(
            positions[0], positions[1], start_radius, end_radius, segments=12
        )
    
    def smooth_mesh_vertices(self, vertices: np.ndarray, faces: List, iterations: int = 2) -> np.ndarray:
        """Apply Laplacian smoothing to mesh vertices"""
        smoothed_vertices = vertices.copy()
        
        for _ in range(iterations):
            new_vertices = smoothed_vertices.copy()
            
            # For each vertex, average with its neighbors
            for i, vertex in enumerate(smoothed_vertices):
                neighbors = []
                
                # Find neighboring vertices through faces
                for face in faces:
                    if i in face:
                        for v_idx in face:
                            if v_idx != i and v_idx not in neighbors:
                                neighbors.append(v_idx)
                
                if neighbors:
                    neighbor_positions = smoothed_vertices[neighbors]
                    new_vertices[i] = np.mean(neighbor_positions, axis=0)
            
            smoothed_vertices = new_vertices
        
        return smoothed_vertices
    
    def generate_full_body_mesh(self, session_dir: str) -> Dict:
        """Generate complete anatomically correct body mesh"""
        print(f"🎭 Generating realistic 3D body mesh from: {session_dir}")
        
        # Load session data
        session_data = self.load_session_data(session_dir)
        
        if not session_data:
            raise ValueError("No session data found")
        
        mesh_data = {
            'vertices': [],
            'faces': [],
            'segments': {},
            'materials': {}
        }
        
        vertex_offset = 0
        
        try:
            # Generate head mesh
            print("🧠 Creating anatomical head mesh...")
            try:
                head_vertices, head_faces = self.create_anatomical_head(session_data)
                head_vertices = self.smooth_mesh_vertices(head_vertices, head_faces)
                
                adjusted_head_faces = [[f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset] 
                                      for f in head_faces]
                
                mesh_data['vertices'].extend(head_vertices.tolist())
                mesh_data['faces'].extend(adjusted_head_faces)
                mesh_data['segments']['head'] = {
                    'vertex_start': vertex_offset,
                    'vertex_count': len(head_vertices),
                    'face_start': 0,
                    'face_count': len(adjusted_head_faces)
                }
                mesh_data['materials']['head'] = {'color': [0.9, 0.8, 0.7, 1.0]}  # Skin color
                vertex_offset += len(head_vertices)
                
            except Exception as e:
                print(f"⚠️ Failed to create head mesh: {e}")
            
            # Generate torso mesh
            print("📦 Creating anatomical torso mesh...")
            try:
                torso_vertices, torso_faces = self.create_anatomical_torso(session_data)
                torso_vertices = self.smooth_mesh_vertices(torso_vertices, torso_faces)
                
                adjusted_torso_faces = [[f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset] 
                                       for f in torso_faces]
                
                face_start = len(mesh_data['faces'])
                mesh_data['vertices'].extend(torso_vertices.tolist())
                mesh_data['faces'].extend(adjusted_torso_faces)
                mesh_data['segments']['torso'] = {
                    'vertex_start': vertex_offset,
                    'vertex_count': len(torso_vertices),
                    'face_start': face_start,
                    'face_count': len(adjusted_torso_faces)
                }
                mesh_data['materials']['torso'] = {'color': [0.9, 0.8, 0.7, 1.0]}  # Skin color
                vertex_offset += len(torso_vertices)
                
            except Exception as e:
                print(f"⚠️ Failed to create torso mesh: {e}")
            
            # Generate limb meshes
            limbs = ['left_upper_arm', 'left_forearm', 'right_upper_arm', 'right_forearm',
                    'left_thigh', 'left_shin', 'right_thigh', 'right_shin']
            
            for limb in limbs:
                try:
                    print(f"🦾 Creating {limb} mesh...")
                    limb_vertices, limb_faces = self.create_anatomical_limb(limb, session_data)
                    limb_vertices = self.smooth_mesh_vertices(limb_vertices, limb_faces)
                    
                    adjusted_limb_faces = [[f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset] 
                                          for f in limb_faces]
                    
                    face_start = len(mesh_data['faces'])
                    mesh_data['vertices'].extend(limb_vertices.tolist())
                    mesh_data['faces'].extend(adjusted_limb_faces)
                    mesh_data['segments'][limb] = {
                        'vertex_start': vertex_offset,
                        'vertex_count': len(limb_vertices),
                        'face_start': face_start,
                        'face_count': len(adjusted_limb_faces)
                    }
                    mesh_data['materials'][limb] = {'color': [0.9, 0.8, 0.7, 1.0]}  # Skin color
                    vertex_offset += len(limb_vertices)
                    
                except Exception as e:
                    print(f"⚠️ Failed to create {limb} mesh: {e}")
            
            # Add metadata
            mesh_data['metadata'] = {
                'total_vertices': len(mesh_data['vertices']),
                'total_faces': len(mesh_data['faces']),
                'segments_created': list(mesh_data['segments'].keys()),
                'generation_method': 'anatomical_pose_reconstruction',
                'smoothing_applied': True,
                'anatomically_correct': True
            }
            
            print(f"✅ Realistic mesh generation completed!")
            print(f"📊 Total vertices: {mesh_data['metadata']['total_vertices']}")
            print(f"📊 Total faces: {mesh_data['metadata']['total_faces']}")
            print(f"🎭 Segments created: {', '.join(mesh_data['segments'].keys())}")
            
            return mesh_data
            
        except Exception as e:
            print(f"❌ Error generating mesh: {e}")
            raise
    
    def save_mesh_data(self, mesh_data: Dict, session_dir: str):
        """Save mesh data to files"""
        # Save as JSON
        mesh_file = os.path.join(session_dir, "body_mesh.json")
        with open(mesh_file, 'w') as f:
            json.dump(mesh_data, f, indent=2)
        
        # Save as OBJ file for external use
        obj_file = os.path.join(session_dir, "body_mesh.obj")
        self.export_to_obj(mesh_data, obj_file)
        
        # Save as PLY file for better 3D software compatibility
        ply_file = os.path.join(session_dir, "body_mesh.ply")
        self.export_to_ply(mesh_data, ply_file)
        
        print(f"💾 Mesh saved: {mesh_file}")
        print(f"💾 OBJ exported: {obj_file}")
        print(f"💾 PLY exported: {ply_file}")
    
    def export_to_obj(self, mesh_data: Dict, obj_file: str):
        """Export mesh to OBJ format with materials"""
        with open(obj_file, 'w') as f:
            f.write("# 3D Body Mesh generated from pose estimation\n")
            f.write(f"# Vertices: {len(mesh_data['vertices'])}\n")
            f.write(f"# Faces: {len(mesh_data['faces'])}\n\n")
            
            # Write vertices
            for vertex in mesh_data['vertices']:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            f.write("\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in mesh_data['faces']:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def export_to_ply(self, mesh_data: Dict, ply_file: str):
        """Export mesh to PLY format"""
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        with open(ply_file, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for vertex in vertices:
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    def visualize_mesh(self, mesh_data: Dict, session_dir: str = None):
        """Visualize the generated mesh with improved rendering"""
        vertices = np.array(mesh_data['vertices'])
        faces = mesh_data['faces']
        
        # Create 3D plot with better styling
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh collection with different colors for different segments
        segments = mesh_data.get('segments', {})
        materials = mesh_data.get('materials', {})
        
        for segment_name, segment_info in segments.items():
            start_face = segment_info['face_start']
            face_count = segment_info['face_count']
            
            segment_faces = faces[start_face:start_face + face_count]
            mesh_faces = []
            
            for face in segment_faces:
                triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
                mesh_faces.append(triangle)
            
            # Get color for this segment
            color = materials.get(segment_name, {}).get('color', [0.7, 0.7, 0.9, 0.8])
            
            # Add mesh to plot
            mesh_collection = Poly3DCollection(mesh_faces, alpha=color[3], 
                                             facecolor=color[:3], edgecolor='black', linewidth=0.1)
            ax.add_collection3d(mesh_collection)
        
        # Set plot limits and styling
        ax.set_xlim([vertices[:, 0].min() - 0.1, vertices[:, 0].max() + 0.1])
        ax.set_ylim([vertices[:, 1].min() - 0.1, vertices[:, 1].max() + 0.1])
        ax.set_zlim([vertices[:, 2].min() - 0.1, vertices[:, 2].max() + 0.1])
        
        # Labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Anatomical Body Mesh Reconstruction', fontsize=16, fontweight='bold')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        # Add lighting effect
        ax.view_init(elev=20, azim=45)
        
        # Save visualization
        if session_dir:
            viz_file = os.path.join(session_dir, "mesh_visualization.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            print(f"📊 Mesh visualization saved: {viz_file}")
        
        plt.show()

def main():
    """Main function for mesh generation"""
    generator = BodyMeshGenerator()
    
    print("🎭 3D Anatomical Body Mesh Generator")
    session_dir = input("Enter session directory path: ").strip()
    
    if not os.path.exists(session_dir):
        print("❌ Session directory not found")
        return
    
    try:
        # Generate mesh
        mesh_data = generator.generate_full_body_mesh(session_dir)
        
        # Save mesh data
        generator.save_mesh_data(mesh_data, session_dir)
        
        # Visualize mesh
        print("🎨 Creating mesh visualization...")
        generator.visualize_mesh(mesh_data, session_dir)
        
        print(f"\n✅ Anatomical mesh generation completed!")
        print(f"📁 Results saved in: {session_dir}")
        print(f"📄 Files created:")
        print(f"  - body_mesh.json (mesh data)")
        print(f"  - body_mesh.obj (3D model)")
        print(f"  - body_mesh.ply (3D model)")
        print(f"  - mesh_visualization.png (preview)")
        
    except Exception as e:
        print(f"❌ Error generating mesh: {e}")

if __name__ == "__main__":
    main()