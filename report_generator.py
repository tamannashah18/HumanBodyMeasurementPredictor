import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import cv2

class MeasurementReportGenerator:
    """Generate comprehensive measurement reports in PDF and JSON formats"""
    
    def __init__(self):
        self.report_template = {
            'header': {
                'title': '3D Body Measurement Report',
                'subtitle': 'AI-Generated Body Analysis',
                'logo': None
            },
            'sections': [
                'summary',
                'measurements',
                'visualizations',
                'methodology',
                'disclaimer'
            ]
        }
    
    def load_session_data(self, session_dir: str) -> Dict:
        """Load all session data for report generation"""
        data = {}
        
        # Load session summary
        summary_file = os.path.join(session_dir, "session_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                data['session'] = json.load(f)
        
        # Load measurements
        measurements_file = os.path.join(session_dir, "body_measurements.json")
        if os.path.exists(measurements_file):
            with open(measurements_file, 'r') as f:
                data['measurements'] = json.load(f)
        
        # Load pose data
        pose_file = os.path.join(session_dir, "pose_landmarks.json")
        if os.path.exists(pose_file):
            with open(pose_file, 'r') as f:
                data['pose'] = json.load(f)
        
        # Load mesh data
        mesh_file = os.path.join(session_dir, "body_mesh.json")
        if os.path.exists(mesh_file):
            with open(mesh_file, 'r') as f:
                data['mesh'] = json.load(f)
        
        return data
    
    def create_measurement_summary_chart(self, measurements: Dict, output_path: str):
        """Create a visual summary chart of measurements"""
        # Filter out metadata
        measurement_data = {k: v for k, v in measurements.items() if k != 'metadata'}
        
        # Prepare data for plotting
        names = []
        values = []
        confidences = []
        
        for name, data in measurement_data.items():
            if isinstance(data, dict) and 'value' in data:
                names.append(name.replace('_', ' ').title())
                values.append(data['value'])
                confidences.append(data.get('confidence', 0.5))
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Measurement values bar chart
        bars = ax1.barh(names, values, color='lightblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Measurement (cm)')
        ax1.set_title('Body Measurements')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax1.text(value + max(values) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.1f} cm', va='center', fontsize=9)
        
        # Confidence levels
        confidence_colors = ['red' if c < 0.5 else 'orange' if c < 0.7 else 'green' for c in confidences]
        bars2 = ax2.barh(names, confidences, color=confidence_colors, alpha=0.7)
        ax2.set_xlabel('Confidence Level')
        ax2.set_title('Measurement Confidence')
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add confidence labels
        for i, (bar, conf) in enumerate(zip(bars2, confidences)):
            ax2.text(conf + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_body_diagram(self, measurements: Dict, output_path: str):
        """Create an annotated body diagram with measurements"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        
        # Draw simplified body outline
        # Head
        head = patches.Circle((0.5, 0.9), 0.08, linewidth=2, edgecolor='black', facecolor='lightgray')
        ax.add_patch(head)
        
        # Torso
        torso = patches.Rectangle((0.35, 0.4), 0.3, 0.4, linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.5)
        ax.add_patch(torso)
        
        # Arms
        left_arm = patches.Rectangle((0.25, 0.6), 0.1, 0.25, linewidth=2, edgecolor='black', facecolor='lightgreen', alpha=0.5)
        right_arm = patches.Rectangle((0.65, 0.6), 0.1, 0.25, linewidth=2, edgecolor='black', facecolor='lightgreen', alpha=0.5)
        ax.add_patch(left_arm)
        ax.add_patch(right_arm)
        
        # Legs
        left_leg = patches.Rectangle((0.4, 0.05), 0.08, 0.35, linewidth=2, edgecolor='black', facecolor='lightyellow', alpha=0.5)
        right_leg = patches.Rectangle((0.52, 0.05), 0.08, 0.35, linewidth=2, edgecolor='black', facecolor='lightyellow', alpha=0.5)
        ax.add_patch(left_leg)
        ax.add_patch(right_leg)
        
        # Add measurement annotations
        measurement_positions = {
            'height': (0.1, 0.5, 'Total Height'),
            'shoulder_width': (0.5, 0.82, 'Shoulder Width'),
            'chest_circumference': (0.8, 0.7, 'Chest'),
            'waist_circumference': (0.8, 0.55, 'Waist'),
            'arm_length': (0.15, 0.72, 'Arm Length'),
            'leg_length': (0.25, 0.25, 'Leg Length')
        }
        
        for measure_name, (x, y, label) in measurement_positions.items():
            if measure_name in measurements and isinstance(measurements[measure_name], dict):
                value = measurements[measure_name]['value']
                confidence = measurements[measure_name].get('confidence', 0.5)
                
                # Color based on confidence
                color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red'
                
                ax.annotate(f'{label}\n{value:.1f} cm', 
                           xy=(x, y), fontsize=10, ha='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Body Measurements Diagram', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_comparison_chart(self, measurements: Dict, output_path: str):
        """Create a chart comparing measurements to average values"""
        # Average adult measurements (approximate)
        average_measurements = {
            'height': 170.0,
            'shoulder_width': 45.0,
            'chest_circumference': 100.0,
            'waist_circumference': 85.0,
            'arm_length': 65.0,
            'leg_length': 85.0
        }
        
        # Prepare data
        names = []
        user_values = []
        avg_values = []
        
        for name, avg_val in average_measurements.items():
            if name in measurements and isinstance(measurements[name], dict):
                names.append(name.replace('_', ' ').title())
                user_values.append(measurements[name]['value'])
                avg_values.append(avg_val)
        
        if not names:
            return None
        
        # Create comparison chart
        x = np.arange(len(names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width/2, user_values, width, label='Your Measurements', color='lightblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, avg_values, width, label='Average Adult', color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Measurements')
        ax.set_ylabel('Value (cm)')
        ax.set_title('Your Measurements vs. Average Adult')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_pdf_report(self, session_dir: str) -> str:
        """Generate comprehensive PDF report"""
        print(f"📄 Generating PDF report for session: {session_dir}")
        
        # Load session data
        session_data = self.load_session_data(session_dir)
        
        if not session_data:
            raise ValueError("No session data found")
        
        # Create output file
        report_file = os.path.join(session_dir, "measurement_report.pdf")
        
        with PdfPages(report_file) as pdf:
            # Page 1: Title and Summary
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('3D Body Measurement Report', fontsize=24, fontweight='bold', y=0.95)
            
            # Add session info
            if 'session' in session_data:
                session_info = session_data['session']
                plt.text(0.1, 0.85, f"Session ID: {session_info.get('session_id', 'N/A')}", 
                        fontsize=12, transform=fig.transFigure)
                plt.text(0.1, 0.82, f"Date: {session_info.get('timestamp', 'N/A')[:10]}", 
                        fontsize=12, transform=fig.transFigure)
                plt.text(0.1, 0.79, f"Views Captured: {', '.join(session_info.get('captured_views', []))}", 
                        fontsize=12, transform=fig.transFigure)
            
            # Add measurement summary
            if 'measurements' in session_data:
                measurements = session_data['measurements']
                y_pos = 0.7
                plt.text(0.1, y_pos, "Measurement Summary:", fontsize=16, fontweight='bold', 
                        transform=fig.transFigure)
                y_pos -= 0.05
                
                for name, data in measurements.items():
                    if name != 'metadata' and isinstance(data, dict):
                        confidence_stars = "★" * int(data.get('confidence', 0.5) * 5)
                        plt.text(0.15, y_pos, f"{name.replace('_', ' ').title()}: {data['value']:.1f} {data['unit']} {confidence_stars}", 
                                fontsize=11, transform=fig.transFigure)
                        y_pos -= 0.04
            
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Measurement Charts
            if 'measurements' in session_data:
                # Create temporary chart files
                chart_file = os.path.join(session_dir, "temp_chart.png")
                self.create_measurement_summary_chart(session_data['measurements'], chart_file)
                
                # Add chart to PDF
                fig = plt.figure(figsize=(8.5, 11))
                img = plt.imread(chart_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title('Measurement Analysis', fontsize=16, fontweight='bold', pad=20)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Clean up temp file
                if os.path.exists(chart_file):
                    os.remove(chart_file)
            
            # Page 3: Body Diagram
            if 'measurements' in session_data:
                diagram_file = os.path.join(session_dir, "temp_diagram.png")
                self.create_body_diagram(session_data['measurements'], diagram_file)
                
                fig = plt.figure(figsize=(8.5, 11))
                img = plt.imread(diagram_file)
                plt.imshow(img)
                plt.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Clean up temp file
                if os.path.exists(diagram_file):
                    os.remove(diagram_file)
            
            # Page 4: Methodology and Disclaimer
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Methodology & Disclaimer', fontsize=18, fontweight='bold', y=0.95)
            
            methodology_text = """
METHODOLOGY:
• Multi-view image capture (front, left, right, back views)
• AI-powered pose estimation using MediaPipe
• 3D triangulation from multiple viewpoints
• Body measurement extraction from 3D landmarks
• Mesh generation for visualization

ACCURACY NOTES:
• Measurements are AI estimates based on pose detection
• Accuracy depends on image quality and pose visibility
• Best results with good lighting and plain background
• Form-fitting clothing recommended for precise measurements

CONFIDENCE LEVELS:
★★★★★ Very High (>80% confidence)
★★★★☆ High (60-80% confidence)
★★★☆☆ Medium (40-60% confidence)
★★☆☆☆ Low (20-40% confidence)
★☆☆☆☆ Very Low (<20% confidence)

DISCLAIMER:
This report provides estimated body measurements generated by AI analysis
of photographs. Results are for informational purposes only and should not
replace professional measurements for medical, fitness, or tailoring purposes.
Accuracy may vary based on image quality, pose, and individual body characteristics.
            """
            
            plt.text(0.1, 0.85, methodology_text, fontsize=10, transform=fig.transFigure, 
                    verticalalignment='top', fontfamily='monospace')
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"✅ PDF report generated: {report_file}")
        return report_file
    
    def generate_json_report(self, session_dir: str) -> str:
        """Generate detailed JSON report"""
        print(f"📊 Generating JSON report for session: {session_dir}")
        
        # Load all session data
        session_data = self.load_session_data(session_dir)
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0',
                'session_directory': session_dir
            },
            'session_info': session_data.get('session', {}),
            'measurements': session_data.get('measurements', {}),
            'pose_analysis': {
                'views_processed': list(session_data.get('pose', {}).keys()) if 'pose' in session_data else [],
                'landmarks_detected': True if 'pose' in session_data else False
            },
            'mesh_info': session_data.get('mesh', {}).get('metadata', {}) if 'mesh' in session_data else {},
            'quality_assessment': self._assess_measurement_quality(session_data),
            'recommendations': self._generate_recommendations(session_data)
        }
        
        # Save JSON report
        report_file = os.path.join(session_dir, "detailed_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ JSON report generated: {report_file}")
        return report_file
    
    def _assess_measurement_quality(self, session_data: Dict) -> Dict:
        """Assess overall quality of measurements"""
        quality = {
            'overall_score': 0.0,
            'factors': {},
            'issues': []
        }
        
        if 'measurements' in session_data:
            measurements = session_data['measurements']
            confidences = []
            
            for name, data in measurements.items():
                if name != 'metadata' and isinstance(data, dict):
                    conf = data.get('confidence', 0.0)
                    confidences.append(conf)
                    
                    if conf < 0.5:
                        quality['issues'].append(f"Low confidence in {name} measurement")
            
            if confidences:
                quality['overall_score'] = np.mean(confidences)
                quality['factors']['average_confidence'] = quality['overall_score']
                quality['factors']['measurements_count'] = len(confidences)
        
        # Check session completeness
        if 'session' in session_data:
            views = session_data['session'].get('captured_views', [])
            quality['factors']['views_captured'] = len(views)
            
            if len(views) < 4:
                quality['issues'].append("Incomplete view capture (missing views)")
        
        return quality
    
    def _generate_recommendations(self, session_data: Dict) -> List[str]:
        """Generate recommendations for improving measurements"""
        recommendations = []
        
        # Check measurement quality
        if 'measurements' in session_data:
            measurements = session_data['measurements']
            low_confidence_count = 0
            
            for name, data in measurements.items():
                if name != 'metadata' and isinstance(data, dict):
                    if data.get('confidence', 0.0) < 0.6:
                        low_confidence_count += 1
            
            if low_confidence_count > 2:
                recommendations.append("Consider retaking photos with better lighting and clearer pose")
                recommendations.append("Ensure you're wearing form-fitting clothing")
                recommendations.append("Use a plain, contrasting background")
        
        # Check session completeness
        if 'session' in session_data:
            views = session_data['session'].get('captured_views', [])
            if len(views) < 4:
                recommendations.append("Capture all 4 views (front, left, right, back) for best accuracy")
        
        # General recommendations
        recommendations.extend([
            "For professional measurements, consult a tailor or healthcare provider",
            "Use measurements as estimates for online shopping or fitness tracking",
            "Retake measurements periodically to track changes over time"
        ])
        
        return recommendations

def main():
    """Main function for report generation"""
    generator = MeasurementReportGenerator()
    
    print("📄 Measurement Report Generator")
    session_dir = input("Enter session directory path: ").strip()
    
    if not os.path.exists(session_dir):
        print("❌ Session directory not found")
        return
    
    print("\nSelect report format:")
    print("1. PDF Report")
    print("2. JSON Report")
    print("3. Both formats")
    
    choice = input("Enter choice (1-3): ").strip()
    
    try:
        if choice in ["1", "3"]:
            pdf_file = generator.generate_pdf_report(session_dir)
            print(f"📄 PDF report: {pdf_file}")
        
        if choice in ["2", "3"]:
            json_file = generator.generate_json_report(session_dir)
            print(f"📊 JSON report: {json_file}")
        
        if choice not in ["1", "2", "3"]:
            print("❌ Invalid choice")
            return
        
        print(f"\n✅ Report generation completed!")
        print(f"📁 Reports saved in: {session_dir}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")

if __name__ == "__main__":
    main()