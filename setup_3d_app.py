#!/usr/bin/env python3
"""
Setup script for 3D Body Measurement Application
"""

import subprocess
import sys
import os
import venv

def create_virtual_environment():
    """Create a virtual environment for 3D app"""
    venv_path = "venv_3d"
    
    if os.path.exists(venv_path):
        print(f"Virtual environment '{venv_path}' already exists.")
        return venv_path
    
    print("Creating virtual environment for 3D Body Measurement App...")
    try:
        venv.create(venv_path, with_pip=True)
        print(f"✓ Virtual environment created at '{venv_path}'")
        return venv_path
    except Exception as e:
        print(f"✗ Error creating virtual environment: {e}")
        return None

def get_venv_python(venv_path):
    """Get the Python executable path in the virtual environment"""
    if os.name == 'nt':  # Windows
        return os.path.join(venv_path, 'Scripts', 'python.exe')
    else:  # Unix/Linux/macOS
        return os.path.join(venv_path, 'bin', 'python')

def install_requirements(venv_path):
    """Install required packages in virtual environment"""
    print("Installing 3D app requirements...")
    
    python_path = get_venv_python(venv_path)
    pip_command = [python_path, "-m", "pip"]
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call(pip_command + ["install", "--upgrade", "pip"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Install requirements
        print("Installing packages from requirements_3d.txt...")
        subprocess.check_call(pip_command + ["install", "-r", "requirements_3d.txt"])
        
        print("✓ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        print("Trying essential packages only...")
        
        # Try installing essential packages individually
        try:
            essential_packages = [
                "opencv-python==4.8.1.78",
                "numpy==1.24.3",
                "mediapipe==0.10.7",
                "matplotlib==3.7.2",
                "Pillow==10.0.1",
                "scipy==1.11.1"
            ]
            
            for package in essential_packages:
                print(f"Installing {package}...")
                subprocess.check_call(pip_command + ["install", package])
            
            print("✓ Essential packages installed!")
            return True
            
        except subprocess.CalledProcessError as e2:
            print(f"✗ Essential package installation also failed: {e2}")
            return False
    except FileNotFoundError:
        print("✗ Python executable not found in virtual environment")
        return False

def test_imports(venv_path):
    """Test if key packages can be imported"""
    print("Testing package imports...")
    python_path = get_venv_python(venv_path)
    
    test_script = """
import sys
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    import matplotlib.pyplot as plt
    from PIL import Image
    print("IMPORTS_OK")
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
except Exception as e:
    print(f"OTHER_ERROR: {e}")
"""
    
    try:
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        
        if output == "IMPORTS_OK":
            print("✓ All key packages imported successfully!")
            return True
        else:
            print(f"✗ Import test failed: {output}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Import test timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing imports: {e}")
        return False

def create_activation_scripts(venv_path):
    """Create activation scripts for 3D app environment"""
    
    # Unix/Linux/macOS script
    activate_script = f"""#!/bin/bash
# 3D Body Measurement App - Virtual Environment Activation
echo "Activating 3D Body Measurement App environment..."
source {venv_path}/bin/activate
echo "3D app environment activated!"
echo ""
echo "Available commands:"
echo "  python main_app.py              - Start the main application"
echo "  python multi_view_capture.py    - Capture multi-view images"
echo "  python pose_estimation_3d.py    - Process pose estimation"
echo "  python body_measurements_3d.py  - Calculate measurements"
echo "  python mesh_generation.py       - Generate 3D mesh"
echo "  python report_generator.py      - Generate reports"
echo "  deactivate                      - Exit environment"
echo ""
echo "🎯 Start with: python main_app.py"
"""
    
    with open("activate_3d.sh", "w", encoding="utf-8") as f:
        f.write(activate_script)
    
    try:
        os.chmod("activate_3d.sh", 0o755)
    except:
        pass
    
    # Windows script
    activate_bat = f"""@echo off
REM 3D Body Measurement App - Virtual Environment Activation
echo Activating 3D Body Measurement App environment...
call {venv_path}\\Scripts\\activate.bat
echo 3D app environment activated!
echo.
echo Available commands:
echo   python main_app.py              - Start the main application
echo   python multi_view_capture.py    - Capture multi-view images
echo   python pose_estimation_3d.py    - Process pose estimation
echo   python body_measurements_3d.py  - Calculate measurements
echo   python mesh_generation.py       - Generate 3D mesh
echo   python report_generator.py      - Generate reports
echo   deactivate                      - Exit environment
echo.
echo 🎯 Start with: python main_app.py
"""
    
    with open("activate_3d.bat", "w", encoding="utf-8") as f:
        f.write(activate_bat)
    
    print("✓ Created 3D app activation scripts: activate_3d.sh and activate_3d.bat")

def create_requirements_if_missing():
    """Create requirements_3d.txt if it doesn't exist"""
    if not os.path.exists("requirements_3d.txt"):
        print("Creating requirements_3d.txt...")
        requirements_content = """# Core ML and CV libraries
opencv-python==4.8.1.78
tensorflow==2.13.0
numpy==1.24.3
mediapipe==0.10.7
Pillow==10.0.1
matplotlib==3.7.2
scikit-learn==1.3.0

# 3D processing and visualization
open3d==0.17.0
trimesh==3.23.5
scipy==1.11.1

# Report generation
reportlab==4.0.4
fpdf2==2.7.4
jinja2==3.1.2

# Data processing
pandas==2.0.3
tabulate==0.9.0

# Additional utilities
tqdm==4.66.1
requests==2.31.0
"""
        with open("requirements_3d.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        print("✓ Created requirements_3d.txt")

def main():
    """Main setup function for 3D app"""
    print("=== 3D Body Measurement Application Setup ===\n")
    
    # Create requirements file if missing
    create_requirements_if_missing()
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    if not venv_path:
        print("Setup failed. Could not create virtual environment.")
        return
    
    print()
    
    # Install requirements
    packages_installed = install_requirements(venv_path)
    
    print()
    
    # Test imports
    imports_ok = False
    if packages_installed:
        imports_ok = test_imports(venv_path)
    else:
        print("Skipping import test (packages not installed)")
    
    print()
    
    # Create activation scripts
    create_activation_scripts(venv_path)
    
    print("\n=== Setup Summary ===")
    print(f"✓ 3D app virtual environment created at '{venv_path}'")
    print(f"{'✓' if packages_installed else '✗'} Requirements {'installed' if packages_installed else 'installation failed'}")
    print(f"{'✓' if imports_ok else '✗'} Package imports {'successful' if imports_ok else 'failed'}")
    print("✓ Activation scripts created")
    
    print("\n=== Next Steps ===")
    
    if packages_installed and imports_ok:
        print("🎉 Setup completed successfully!")
        print("\n1. Activate the 3D app environment:")
        if os.name == 'nt':  # Windows
            print("   activate_3d.bat")
        else:  # Unix/Linux/macOS
            print("   ./activate_3d.sh")
        
        print("\n2. Start the main application:")
        print("   python main_app.py")
        
        print("\n🎯 Application Features:")
        print("   📸 Multi-view image capture (front, left, right, back)")
        print("   🤖 AI-powered 3D pose estimation")
        print("   📏 Precise body measurement extraction")
        print("   🎭 3D mesh generation and visualization")
        print("   📄 Comprehensive PDF and JSON reports")
        print("   🔒 Privacy-focused (all processing local)")
        
        print("\n📋 Usage Tips:")
        print("   - Use good lighting and plain background")
        print("   - Wear form-fitting clothes")
        print("   - Stand 6-8 feet from camera")
        print("   - Capture all 4 views for best accuracy")
        
    elif packages_installed:
        print("⚠️  Packages installed but imports failed.")
        print("Some advanced features may not work. You can still try:")
        print("\n1. Activate environment:")
        if os.name == 'nt':
            print("   activate_3d.bat")
        else:
            print("   ./activate_3d.sh")
        print("   python main_app.py")
        
    else:
        print("⚠️  Setup incomplete. Package installation failed.")
        print("\nTroubleshooting:")
        print("   - Ensure stable internet connection")
        print("   - Try running as administrator (Windows)")
        print("   - Check firewall/proxy settings")
        print("   - Some packages require system dependencies")

if __name__ == "__main__":
    main()