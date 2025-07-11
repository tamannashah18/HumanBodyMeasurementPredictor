import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2

class BodyMeshGenerator:
    """Generate 3D body mesh from pose landmarks and measurements"""
    
    def __init__(self):
        # Define body segments and their connections
        self.body_segments = {
            'head': {
                'landmarks': ['nose'],
                'connections': [],
                'type': 'sphere'
            },
            'torso': {
                'landmarks': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
                'connections': [
                    ('left_shoulder', 'right_shoulder'),
                    ('left_shoulder', 'left_hip'),
                    ('right_shoulder', 'right_hip'),
                    ('left_hip', 'right_hip')
                ],
                'type': 'rectangular_prism'
            },
            'left_arm': {
                'landmarks': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'connections': [
                    ('left_shoulder', 'left_elbow'),
                    ('left_elbow', 'left_wrist')
                ],
                'type': 'cylindrical'
            },
            'right_arm': {
                'landmarks': ['right_shoulder', 'right_elbow', 'right_wrist'],
                'connections': [
                    ('right_shoulder', 'right_elbow'),
                    ('right_elbow', 'right_wrist')
                ],
                'type': 'cylindrical'
            },
            'left_leg': {
                'landmarks': ['left_hip', 'left_knee', 'left_ankle'],
                'connections': [
                    ('left_hip', 'left_knee'),
                    ('left_knee', 'left_ankle')
                ],
                'type': 'cylindrical'
            },
            'right_leg': {
                'landmarks': ['right_hip', 'right_knee', 'right_ankle'],
                'connections': [
                    ('right_hip', 'right_knee'),
                    ('right_knee', 'right_ankle')
                ],
                'type': 'cylindrical'
            }
        }
        
        # Landmark mapping
        self.landmark_map = {
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
        """Get 3D position of a landmark"""
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
                return (landmark['x'], landmark['y'], landmark['z'])
        
        return None
    
    def create_cylinder_mesh(self, start_point: Tuple[float, float, float], 
                           end_point: Tuple[float, float, float], 
                           radius: float, segments: int = 8) -> Tuple[np.ndarray, List]:
        """Create a cylindrical mesh between two points"""
        start = np.array(start_point)
        end = np.array(end_point)
        
        # Calculate cylinder axis
        axis = end - start
        length = np.linalg.norm(axis)
        axis_normalized = axis / length
        
        # Create perpendicular vectors
        if abs(axis_normalized[2]) < 0.9:
            perp1 = np.cross(axis_normalized, [0, 0, 1])
        else:
            perp1 = np.cross(axis_normalized, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(axis_normalized, perp1)
        
        # Generate vertices
        vertices = []
        faces = []
        
        # Create circles at both ends
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            
            # Bottom circle
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(start + offset)
            
            # Top circle
            vertices.append(end + offset)
        
        # Create faces
        for i in range(segments):
            next_i = (i + 1) % segments
            
            # Side faces (two triangles per segment)
            bottom1, top1 = 2 * i, 2 * i + 1
            bottom2, top2 = 2 * next_i, 2 * next_i + 1
            
            faces.append([bottom1, top1, top2])
            faces.append([bottom1, top2, bottom2])
        
        return np.array(vertices), faces
    
    def create_box_mesh(self, corners: List[Tuple[float, float, float]]) -> Tuple[np.ndarray, List]:
        """Create a box mesh from corner points"""
        if len(corners) != 8:
            raise ValueError("Box mesh requires 8 corner points")
        
        vertices = np.array(corners)
        
        # Define faces (each face is defined by 4 vertices, split into 2 triangles)
        faces = [
            # Front face
            [0, 1, 2], [0, 2, 3],
            # Back face
            [4, 6, 5], [4, 7, 6],
            # Left face
            [0, 4, 5], [0, 5, 1],
            # Right face
            [2, 6, 7], [2, 7, 3],
            # Top face
            [1, 5, 6], [1, 6, 2],
            # Bottom face
            [0, 3, 7], [0, 7, 4]
        ]
        
        return vertices, faces
    
    def create_torso_mesh(self, session_data: Dict) -> Tuple[np.ndarray, List]:
        """Create torso mesh from shoulder and hip landmarks"""
        # Get landmark positions
        left_shoulder = self.get_landmark_position('left_shoulder', session_data)
        right_shoulder = self.get_landmark_position('right_shoulder', session_data)
        left_hip = self.get_landmark_position('left_hip', session_data)
        right_hip = self.get_landmark_position('right_hip', session_data)
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            raise ValueError("Missing required landmarks for torso mesh")
        
        # Estimate torso depth from measurements
        torso_depth = 0.3  # Default depth
        if 'measurements' in session_data and 'chest_circumference' in session_data['measurements']:
            chest_circ = session_data['measurements']['chest_circumference']['value']
            torso_depth = chest_circ / (2 * np.pi) * 0.5  # Approximate depth
        
        # Create 8 corners of torso box
        # Front face
        front_offset = np.array([0, 0, torso_depth / 2])
        corners = [
            np.array(left_shoulder) + front_offset,   # 0: left shoulder front
            np.array(right_shoulder) + front_offset,  # 1: right shoulder front
            np.array(right_hip) + front_offset,       # 2: right hip front
            np.array(left_hip) + front_offset,        # 3: left hip front
        ]
        
        # Back face
        back_offset = np.array([0, 0, -torso_depth / 2])
        corners.extend([
            np.array(left_shoulder) + back_offset,    # 4: left shoulder back
            np.array(right_shoulder) + back_offset,   # 5: right shoulder back
            np.array(right_hip) + back_offset,        # 6: right hip back
            np.array(left_hip) + back_offset,         # 7: left hip back
        ])
        
        return self.create_box_mesh(corners)
    
    def create_limb_mesh(self, segment_name: str, session_data: Dict) -> Tuple[np.ndarray, List]:
        """Create mesh for arm or leg segments"""
        segment = self.body_segments[segment_name]
        landmarks = segment['landmarks']
        
        # Get landmark positions
        positions = []
        for landmark in landmarks:
            pos = self.get_landmark_position(landmark, session_data)
            if pos:
                positions.append(pos)
        
        if len(positions) < 2:
            raise ValueError(f"Insufficient landmarks for {segment_name}")
        
        # Estimate limb radius from measurements
        radius = 0.05  # Default radius
        if 'measurements' in session_data:
            if 'arm' in segment_name and 'arm_length' in session_data['measurements']:
                radius = 0.03  # Arm radius
            elif 'leg' in segment_name and 'leg_length' in session_data['measurements']:
                radius = 0.04  # Leg radius
        
        # Create cylindrical segments
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for i in range(len(positions) - 1):
            vertices, faces = self.create_cylinder_mesh(positions[i], positions[i + 1], radius)
            
            # Adjust face indices
            adjusted_faces = [[f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset] for f in faces]
            
            all_vertices.extend(vertices)
            all_faces.extend(adjusted_faces)
            vertex_offset += len(vertices)
        
        return np.array(all_vertices), all_faces
    
    def generate_full_body_mesh(self, session_dir: str) -> Dict:
        """Generate complete body mesh from session data"""
        print(f"🎭 Generating 3D body mesh from: {session_dir}")
        
        # Load session data
        session_data = self.load_session_data(session_dir)
        
        if not session_data:
            raise ValueError("No session data found")
        
        mesh_data = {
            'vertices': [],
            'faces': [],
            'segments': {}
        }
        
        vertex_offset = 0
        
        try:
            # Generate torso mesh
            print("📦 Creating torso mesh...")
            torso_vertices, torso_faces = self.create_torso_mesh(session_data)
            
            # Adjust face indices
            adjusted_torso_faces = [[f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset] 
                                   for f in torso_faces]
            
            mesh_data['vertices'].extend(torso_vertices.tolist())
            mesh_data['faces'].extend(adjusted_torso_faces)
            mesh_data['segments']['torso'] = {
                'vertex_start': vertex_offset,
                'vertex_count': len(torso_vertices),
                'face_start': 0,
                'face_count': len(adjusted_torso_faces)
            }
            vertex_offset += len(torso_vertices)
            
            # Generate limb meshes
            limbs = ['left_arm', 'right_arm', 'left_leg', 'right_leg']
            
            for limb in limbs:
                try:
                    print(f"🦾 Creating {limb} mesh...")
                    limb_vertices, limb_faces = self.create_limb_mesh(limb, session_data)
                    
                    # Adjust face indices
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
                    vertex_offset += len(limb_vertices)
                    
                except Exception as e:
                    print(f"⚠️ Failed to create {limb} mesh: {e}")
            
            # Add metadata
            mesh_data['metadata'] = {
                'total_vertices': len(mesh_data['vertices']),
                'total_faces': len(mesh_data['faces']),
                'segments_created': list(mesh_data['segments'].keys()),
                'generation_method': 'pose_landmarks_to_mesh'
            }
            
            print(f"✅ Mesh generation completed!")
            print(f"📊 Total vertices: {mesh_data['metadata']['total_vertices']}")
            print(f"📊 Total faces: {mesh_data['metadata']['total_faces']}")
            
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
        
        print(f"💾 Mesh saved: {mesh_file}")
        print(f"💾 OBJ exported: {obj_file}")
    
    def export_to_obj(self, mesh_data: Dict, obj_file: str):
        """Export mesh to OBJ format"""
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
    
    def visualize_mesh(self, mesh_data: Dict, session_dir: str = None):
        """Visualize the generated mesh"""
        vertices = np.array(mesh_data['vertices'])
        faces = mesh_data['faces']
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh collection
        mesh_faces = []
        for face in faces:
            triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
            mesh_faces.append(triangle)
        
        # Add mesh to plot
        mesh_collection = Poly3DCollection(mesh_faces, alpha=0.7, facecolor='lightblue', edgecolor='black')
        ax.add_collection3d(mesh_collection)
        
        # Set plot limits
        ax.set_xlim([vertices[:, 0].min(), vertices[:, 0].max()])
        ax.set_ylim([vertices[:, 1].min(), vertices[:, 1].max()])
        ax.set_zlim([vertices[:, 2].min(), vertices[:, 2].max()])
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Body Mesh Reconstruction')
        
        # Save visualization
        if session_dir:
            viz_file = os.path.join(session_dir, "mesh_visualization.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            print(f"📊 Mesh visualization saved: {viz_file}")
        
        plt.show()

def main():
    """Main function for mesh generation"""
    generator = BodyMeshGenerator()
    
    print("🎭 3D Body Mesh Generator")
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
        
        print(f"\n✅ Mesh generation completed!")
        print(f"📁 Results saved in: {session_dir}")
        print(f"📄 Files created:")
        print(f"  - body_mesh.json (mesh data)")
        print(f"  - body_mesh.obj (3D model)")
        print(f"  - mesh_visualization.png (preview)")
        
    except Exception as e:
        print(f"❌ Error generating mesh: {e}")

if __name__ == "__main__":
    main()