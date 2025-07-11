# 3D Body Measurement Application

A comprehensive Python application for capturing multi-view photos and generating precise 3D body measurements using AI-powered pose estimation.

## Features

### 📸 Multi-View Capture
- Capture 4 views: front, left side, right side, and back
- Real-time guidance and countdown
- Image quality validation
- Session management

### 🤖 AI-Powered Analysis
- MediaPipe pose estimation
- 3D triangulation from multiple viewpoints
- Confidence scoring for measurements
- Robust error handling

### 📏 Precise Measurements
- Height (total body height)
- Shoulder width
- Chest circumference
- Waist circumference
- Hip circumference
- Arm length
- Leg length (inseam)
- Torso length

### 🎭 3D Visualization
- Generate 3D body mesh
- Interactive visualization
- Export to OBJ format
- Measurement annotations

### 📄 Comprehensive Reports
- PDF reports with charts and diagrams
- JSON data for integration
- Measurement confidence scores
- Quality assessment and recommendations

## Installation

### Quick Setup

1. **Clone or download this repository**
2. **Run the setup script:**
   ```bash
   python setup_3d_app.py
   ```
3. **Activate the environment:**
   - Windows: `activate_3d.bat`
   - Linux/macOS: `./activate_3d.sh`
4. **Start the application:**
   ```bash
   python main_app.py
   ```

### Manual Installation

1. **Create virtual environment:**
   ```bash
   python -m venv venv_3d
   ```

2. **Activate environment:**
   ```bash
   # Windows
   venv_3d\Scripts\activate
   
   # Linux/macOS
   source venv_3d/bin/activate
   ```

3. **Install requirements:**
   ```bash
   pip install -r requirements_3d.txt
   ```

## Usage

### Complete Workflow

1. **Start the application:**
   ```bash
   python main_app.py
   ```

2. **Select "Start New Measurement Session"**

3. **Follow the capture instructions:**
   - Stand 6-8 feet from camera
   - Use good lighting
   - Wear form-fitting clothes
   - Use plain background
   - Capture all 4 views when prompted

4. **Wait for processing:**
   - Pose estimation
   - 3D reconstruction
   - Measurement calculation
   - Mesh generation
   - Report creation

5. **Review results:**
   - View measurements
   - Check 3D visualization
   - Download PDF/JSON reports

### Individual Components

You can also run individual components:

```bash
# Capture multi-view images only
python multi_view_capture.py

# Process existing session
python pose_estimation_3d.py

# Calculate measurements
python body_measurements_3d.py

# Generate 3D mesh
python mesh_generation.py

# Create reports
python report_generator.py
```

## File Structure

```
captures/
├── session_YYYYMMDD_HHMMSS/
│   ├── front_view.jpg           # Front view image
│   ├── left_view.jpg            # Left side view
│   ├── right_view.jpg           # Right side view
│   ├── back_view.jpg            # Back view image
│   ├── session_summary.json     # Capture session info
│   ├── pose_landmarks.json      # Detected pose landmarks
│   ├── triangulated_3d.json     # 3D triangulated points
│   ├── body_measurements.json   # Calculated measurements
│   ├── body_mesh.json          # 3D mesh data
│   ├── body_mesh.obj           # 3D mesh (OBJ format)
│   ├── measurement_report.pdf   # Visual report
│   ├── detailed_report.json     # Detailed data
│   └── visualizations/          # Generated charts
```

## Measurement Accuracy

### Confidence Levels
- ★★★★★ Very High (>80% confidence)
- ★★★★☆ High (60-80% confidence)
- ★★★☆☆ Medium (40-60% confidence)
- ★★☆☆☆ Low (20-40% confidence)
- ★☆☆☆☆ Very Low (<20% confidence)

### Factors Affecting Accuracy
- **Image Quality**: Good lighting, clear focus
- **Background**: Plain, contrasting background
- **Clothing**: Form-fitting clothes
- **Pose**: Natural standing position
- **Distance**: 6-8 feet from camera
- **View Completeness**: All 4 views captured

### Typical Accuracy
- Height: ±2-5 cm
- Shoulder Width: ±2-4 cm
- Circumferences: ±3-8 cm (estimated from width)
- Limb Lengths: ±3-6 cm

## Technical Details

### AI Models Used
- **MediaPipe Pose**: Google's pose estimation model
- **Custom 3D Triangulation**: Multi-view geometry
- **Mesh Generation**: Procedural body modeling

### Processing Pipeline
1. **Image Capture**: Multi-view photography
2. **Pose Detection**: AI landmark extraction
3. **3D Reconstruction**: Triangulation from multiple views
4. **Measurement Extraction**: Anthropometric calculations
5. **Mesh Generation**: 3D body model creation
6. **Report Generation**: PDF and JSON outputs

### Privacy & Security
- **Local Processing**: All computation on your device
- **No Cloud Upload**: Images never leave your computer
- **Data Control**: You own all generated data
- **Optional Sharing**: Export only what you choose

## Troubleshooting

### Common Issues

**Camera not detected:**
- Check camera permissions
- Ensure camera isn't used by other apps
- Try different camera index

**Poor measurement accuracy:**
- Improve lighting conditions
- Use plain background
- Wear form-fitting clothes
- Ensure all 4 views are captured
- Stand at proper distance (6-8 feet)

**Pose detection fails:**
- Check image quality
- Ensure person is fully visible
- Avoid complex backgrounds
- Improve lighting

**Installation issues:**
- Update Python to 3.8+
- Install Visual C++ redistributables (Windows)
- Check internet connection
- Try manual package installation

### Getting Help

1. **Check the help section in the app**
2. **Review measurement confidence scores**
3. **Retake photos with better conditions**
4. **Check system requirements**

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- Webcam or camera
- 2GB free disk space

### Recommended
- Python 3.9+
- 8GB RAM
- HD webcam
- Good lighting setup
- 5GB free disk space

### Supported Platforms
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+)

## License

This project is open source. See individual package licenses for dependencies.

## Disclaimer

This application provides estimated body measurements generated by AI analysis of photographs. Results are for informational purposes only and should not replace professional measurements for medical, fitness, or tailoring purposes. Accuracy may vary based on image quality, pose, and individual body characteristics.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

- MediaPipe team for pose estimation models
- OpenCV community for computer vision tools
- Scientific Python ecosystem for data processing