#!/usr/bin/env python3
"""
3D Body Measurement Application
Main application that orchestrates the complete workflow
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional

# Import our modules
from multi_view_capture import MultiViewCapture
from pose_estimation_3d import PoseEstimation3D
from body_measurements_3d import BodyMeasurements3D
from mesh_generation import BodyMeshGenerator
from report_generator import MeasurementReportGenerator

class BodyMeasurementApp:
    """Main application class for 3D body measurement system"""
    
    def __init__(self):
        self.capture = MultiViewCapture()
        self.pose_estimator = PoseEstimation3D()
        self.measurement_calculator = BodyMeasurements3D()
        self.mesh_generator = BodyMeshGenerator()
        self.report_generator = MeasurementReportGenerator()
        
        self.current_session_dir = None
    
    def print_welcome(self):
        """Print welcome message and instructions"""
        print("=" * 80)
        print("🎯 3D BODY MEASUREMENT APPLICATION")
        print("=" * 80)
        print("📸 Capture 4-view photos for precise 3D body measurements")
        print("🤖 AI-powered pose estimation and 3D reconstruction")
        print("📏 Extract detailed body measurements with confidence scores")
        print("🎭 Generate 3D mesh visualization")
        print("📄 Create comprehensive measurement reports")
        print("=" * 80)
    
    def show_main_menu(self):
        """Display main menu options"""
        print("\n🎯 MAIN MENU")
        print("1. 📸 Start New Measurement Session")
        print("2. 🔄 Process Existing Session")
        print("3. 📁 List Previous Sessions")
        print("4. 📊 Generate Report for Session")
        print("5. 🎭 View 3D Mesh for Session")
        print("6. ❓ Help & Instructions")
        print("7. 🚪 Exit")
        
        return input("\nSelect option (1-7): ").strip()
    
    def start_new_session(self):
        """Start a complete new measurement session"""
        print("\n🚀 Starting New 3D Body Measurement Session")
        print("=" * 60)
        
        # Step 1: Capture multi-view images
        print("📸 STEP 1: Multi-View Image Capture")
        session_dir = self.capture.run_capture_session()
        
        if not session_dir:
            print("❌ Image capture failed or was cancelled")
            return None
        
        self.current_session_dir = session_dir
        
        # Step 2: Process poses and create 3D reconstruction
        print(f"\n🔍 STEP 2: 3D Pose Estimation")
        try:
            pose_data = self.pose_estimator.process_multi_view_session(session_dir)
            triangulated_data = self.pose_estimator.triangulate_3d_points(pose_data)
            
            if triangulated_data:
                # Save triangulated data
                triangulation_file = os.path.join(session_dir, "triangulated_3d.json")
                with open(triangulation_file, 'w') as f:
                    json.dump(triangulated_data, f, indent=2)
                print("✅ 3D pose estimation completed")
            else:
                print("⚠️ 3D triangulation failed, using 2D pose data")
        
        except Exception as e:
            print(f"⚠️ Pose estimation error: {e}")
            print("Continuing with available data...")
        
        # Step 3: Calculate body measurements
        print(f"\n📏 STEP 3: Body Measurement Calculation")
        try:
            measurements = self.measurement_calculator.calculate_all_measurements(session_dir)
            
            if measurements:
                self.measurement_calculator.print_measurements_report(measurements)
                self.measurement_calculator.save_measurements(measurements, session_dir)
                print("✅ Body measurements calculated")
            else:
                print("❌ Failed to calculate measurements")
                return session_dir
        
        except Exception as e:
            print(f"❌ Measurement calculation error: {e}")
            return session_dir
        
        # Step 4: Generate 3D mesh
        print(f"\n🎭 STEP 4: 3D Mesh Generation")
        try:
            mesh_data = self.mesh_generator.generate_full_body_mesh(session_dir)
            self.mesh_generator.save_mesh_data(mesh_data, session_dir)
            print("✅ 3D mesh generated")
        
        except Exception as e:
            print(f"⚠️ Mesh generation error: {e}")
            print("Continuing without 3D mesh...")
        
        # Step 5: Generate reports
        print(f"\n📄 STEP 5: Report Generation")
        try:
            pdf_report = self.report_generator.generate_pdf_report(session_dir)
            json_report = self.report_generator.generate_json_report(session_dir)
            print("✅ Reports generated")
        
        except Exception as e:
            print(f"⚠️ Report generation error: {e}")
        
        # Session complete
        print(f"\n🎉 SESSION COMPLETED SUCCESSFULLY!")
        print(f"📁 All files saved in: {session_dir}")
        self.show_session_summary(session_dir)
        
        return session_dir
    
    def process_existing_session(self):
        """Process an existing session directory"""
        print("\n🔄 Process Existing Session")
        
        # List available sessions
        sessions = self.capture.list_sessions()
        if not sessions:
            print("📭 No existing sessions found")
            return
        
        print("\n📁 Available sessions:")
        for i, session in enumerate(sessions, 1):
            print(f"{i}. {session}")
        
        try:
            choice = int(input(f"\nSelect session (1-{len(sessions)}): ")) - 1
            if 0 <= choice < len(sessions):
                session_id = sessions[choice]
                session_dir = os.path.join(self.capture.output_dir, session_id)
                
                print(f"\n🔄 Processing session: {session_id}")
                self.process_session_pipeline(session_dir)
            else:
                print("❌ Invalid selection")
        
        except ValueError:
            print("❌ Invalid input")
    
    def process_session_pipeline(self, session_dir: str):
        """Process a session through the complete pipeline"""
        print(f"🔄 Processing pipeline for: {session_dir}")
        
        # Check what's already been processed
        existing_files = {
            'pose': os.path.exists(os.path.join(session_dir, "pose_landmarks.json")),
            'triangulated': os.path.exists(os.path.join(session_dir, "triangulated_3d.json")),
            'measurements': os.path.exists(os.path.join(session_dir, "body_measurements.json")),
            'mesh': os.path.exists(os.path.join(session_dir, "body_mesh.json")),
            'reports': os.path.exists(os.path.join(session_dir, "measurement_report.pdf"))
        }
        
        print(f"📊 Current status:")
        for step, exists in existing_files.items():
            status = "✅" if exists else "❌"
            print(f"  {status} {step.replace('_', ' ').title()}")
        
        # Process missing steps
        if not existing_files['pose']:
            print("\n🔍 Processing pose estimation...")
            try:
                pose_data = self.pose_estimator.process_multi_view_session(session_dir)
                if not existing_files['triangulated']:
                    triangulated_data = self.pose_estimator.triangulate_3d_points(pose_data)
                    if triangulated_data:
                        triangulation_file = os.path.join(session_dir, "triangulated_3d.json")
                        with open(triangulation_file, 'w') as f:
                            json.dump(triangulated_data, f, indent=2)
            except Exception as e:
                print(f"❌ Pose estimation failed: {e}")
        
        if not existing_files['measurements']:
            print("\n📏 Calculating measurements...")
            try:
                measurements = self.measurement_calculator.calculate_all_measurements(session_dir)
                if measurements:
                    self.measurement_calculator.save_measurements(measurements, session_dir)
            except Exception as e:
                print(f"❌ Measurement calculation failed: {e}")
        
        if not existing_files['mesh']:
            print("\n🎭 Generating 3D mesh...")
            try:
                mesh_data = self.mesh_generator.generate_full_body_mesh(session_dir)
                self.mesh_generator.save_mesh_data(mesh_data, session_dir)
            except Exception as e:
                print(f"❌ Mesh generation failed: {e}")
        
        if not existing_files['reports']:
            print("\n📄 Generating reports...")
            try:
                self.report_generator.generate_pdf_report(session_dir)
                self.report_generator.generate_json_report(session_dir)
            except Exception as e:
                print(f"❌ Report generation failed: {e}")
        
        print("✅ Processing completed!")
        self.show_session_summary(session_dir)
    
    def list_sessions(self):
        """List all available sessions with details"""
        print("\n📁 Previous Sessions")
        print("=" * 60)
        
        sessions = self.capture.list_sessions()
        if not sessions:
            print("📭 No sessions found")
            return
        
        for session in sessions:
            session_dir = os.path.join(self.capture.output_dir, session)
            summary_file = os.path.join(session_dir, "session_summary.json")
            
            print(f"\n📂 {session}")
            
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    print(f"   📅 Date: {summary.get('timestamp', 'Unknown')[:19]}")
                    print(f"   📸 Views: {', '.join(summary.get('captured_views', []))}")
                    print(f"   📊 Status: {summary.get('status', 'Unknown')}")
                    
                    # Check processing status
                    files_exist = {
                        'Pose Data': os.path.exists(os.path.join(session_dir, "pose_landmarks.json")),
                        'Measurements': os.path.exists(os.path.join(session_dir, "body_measurements.json")),
                        '3D Mesh': os.path.exists(os.path.join(session_dir, "body_mesh.json")),
                        'PDF Report': os.path.exists(os.path.join(session_dir, "measurement_report.pdf"))
                    }
                    
                    processed = [name for name, exists in files_exist.items() if exists]
                    print(f"   ✅ Processed: {', '.join(processed) if processed else 'None'}")
                
                except Exception as e:
                    print(f"   ❌ Error reading summary: {e}")
            else:
                print("   ⚠️ No summary file found")
    
    def generate_report_for_session(self):
        """Generate report for a specific session"""
        print("\n📄 Generate Report for Session")
        
        sessions = self.capture.list_sessions()
        if not sessions:
            print("📭 No sessions found")
            return
        
        print("\n📁 Available sessions:")
        for i, session in enumerate(sessions, 1):
            print(f"{i}. {session}")
        
        try:
            choice = int(input(f"\nSelect session (1-{len(sessions)}): ")) - 1
            if 0 <= choice < len(sessions):
                session_id = sessions[choice]
                session_dir = os.path.join(self.capture.output_dir, session_id)
                
                print("\nSelect report format:")
                print("1. PDF Report")
                print("2. JSON Report")
                print("3. Both formats")
                
                format_choice = input("Enter choice (1-3): ").strip()
                
                try:
                    if format_choice in ["1", "3"]:
                        pdf_file = self.report_generator.generate_pdf_report(session_dir)
                        print(f"📄 PDF report: {pdf_file}")
                    
                    if format_choice in ["2", "3"]:
                        json_file = self.report_generator.generate_json_report(session_dir)
                        print(f"📊 JSON report: {json_file}")
                    
                    if format_choice not in ["1", "2", "3"]:
                        print("❌ Invalid choice")
                
                except Exception as e:
                    print(f"❌ Error generating report: {e}")
            else:
                print("❌ Invalid selection")
        
        except ValueError:
            print("❌ Invalid input")
    
    def view_3d_mesh(self):
        """View 3D mesh for a session"""
        print("\n🎭 View 3D Mesh")
        
        sessions = self.capture.list_sessions()
        if not sessions:
            print("📭 No sessions found")
            return
        
        print("\n📁 Available sessions:")
        for i, session in enumerate(sessions, 1):
            session_dir = os.path.join(self.capture.output_dir, session)
            mesh_exists = os.path.exists(os.path.join(session_dir, "body_mesh.json"))
            status = "🎭" if mesh_exists else "❌"
            print(f"{i}. {session} {status}")
        
        try:
            choice = int(input(f"\nSelect session (1-{len(sessions)}): ")) - 1
            if 0 <= choice < len(sessions):
                session_id = sessions[choice]
                session_dir = os.path.join(self.capture.output_dir, session_id)
                mesh_file = os.path.join(session_dir, "body_mesh.json")
                
                if os.path.exists(mesh_file):
                    try:
                        with open(mesh_file, 'r') as f:
                            mesh_data = json.load(f)
                        
                        print("🎨 Creating 3D visualization...")
                        self.mesh_generator.visualize_mesh(mesh_data, session_dir)
                    
                    except Exception as e:
                        print(f"❌ Error visualizing mesh: {e}")
                else:
                    print("❌ No 3D mesh found for this session")
                    generate = input("Generate mesh now? (y/n): ").lower()
                    if generate == 'y':
                        try:
                            mesh_data = self.mesh_generator.generate_full_body_mesh(session_dir)
                            self.mesh_generator.save_mesh_data(mesh_data, session_dir)
                            self.mesh_generator.visualize_mesh(mesh_data, session_dir)
                        except Exception as e:
                            print(f"❌ Error generating mesh: {e}")
            else:
                print("❌ Invalid selection")
        
        except ValueError:
            print("❌ Invalid input")
    
    def show_help(self):
        """Show help and instructions"""
        help_text = """
🎯 3D BODY MEASUREMENT APPLICATION - HELP

📸 CAPTURE INSTRUCTIONS:
• Stand 6-8 feet from camera
• Use good lighting (avoid shadows)
• Wear form-fitting clothes
• Use plain, contrasting background
• Stand straight with arms slightly away from body
• Capture all 4 views: front, left, right, back

🤖 PROCESSING PIPELINE:
1. Multi-view image capture
2. AI pose estimation (MediaPipe)
3. 3D triangulation from multiple views
4. Body measurement extraction
5. 3D mesh generation
6. Report generation (PDF/JSON)

📏 MEASUREMENTS EXTRACTED:
• Height (total body height)
• Shoulder width
• Chest circumference
• Waist circumference
• Hip circumference
• Arm length
• Leg length (inseam)

🎭 3D VISUALIZATION:
• Interactive 3D mesh model
• Measurement annotations
• Export to OBJ format

📄 REPORTS:
• PDF: Visual report with charts and diagrams
• JSON: Detailed data for integration

⚠️ ACCURACY NOTES:
• Results are AI estimates
• Accuracy depends on image quality
• Use for general measurements only
• Consult professionals for precise measurements

🔧 TROUBLESHOOTING:
• Ensure good lighting
• Check camera permissions
• Use plain background
• Wear form-fitting clothes
• Retake photos if confidence is low
        """
        print(help_text)
    
    def show_session_summary(self, session_dir: str):
        """Show summary of session files and results"""
        print(f"\n📊 SESSION SUMMARY: {os.path.basename(session_dir)}")
        print("=" * 60)
        
        files_to_check = {
            "📸 Session Summary": "session_summary.json",
            "🔍 Pose Landmarks": "pose_landmarks.json",
            "🔺 3D Triangulation": "triangulated_3d.json",
            "📏 Body Measurements": "body_measurements.json",
            "🎭 3D Mesh": "body_mesh.json",
            "📄 PDF Report": "measurement_report.pdf",
            "📊 JSON Report": "detailed_report.json",
            "🎨 Visualizations": "mesh_visualization.png"
        }
        
        for description, filename in files_to_check.items():
            filepath = os.path.join(session_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"✅ {description}: {filename} ({size:,} bytes)")
            else:
                print(f"❌ {description}: Not found")
        
        print(f"\n📁 Session directory: {session_dir}")
    
    def run(self):
        """Main application loop"""
        self.print_welcome()
        
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == "1":
                    self.start_new_session()
                
                elif choice == "2":
                    self.process_existing_session()
                
                elif choice == "3":
                    self.list_sessions()
                
                elif choice == "4":
                    self.generate_report_for_session()
                
                elif choice == "5":
                    self.view_3d_mesh()
                
                elif choice == "6":
                    self.show_help()
                
                elif choice == "7":
                    print("\n👋 Thank you for using 3D Body Measurement App!")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-7.")
                
                # Pause before showing menu again
                input("\nPress Enter to continue...")
            
            except KeyboardInterrupt:
                print("\n\n👋 Application interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("Please try again or contact support.")

def main():
    """Entry point for the application"""
    app = BodyMeasurementApp()
    app.run()

if __name__ == "__main__":
    main()