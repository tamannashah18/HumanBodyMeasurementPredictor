#!/usr/bin/env python3
"""
Setup script for Body Measurement Prediction System with virtual environment
"""

import subprocess
import sys
import os
import venv

def create_virtual_environment():
    """Create a virtual environment"""
    venv_path = "venv"
    
    if os.path.exists(venv_path):
        print(f"Virtual environment '{venv_path}' already exists.")
        return venv_path
    
    print("Creating virtual environment...")
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

def get_venv_pip(venv_path):
    """Get the pip executable path in the virtual environment"""
    if os.name == 'nt':  # Windows
        return os.path.join(venv_path, 'Scripts', 'pip.exe')
    else:  # Unix/Linux/macOS
        return os.path.join(venv_path, 'bin', 'pip')

def install_requirements(venv_path):
    """Install required packages in virtual environment"""
    print("Installing required packages in virtual environment...")
    
    pip_path = get_venv_pip(venv_path)
    python_path = get_venv_python(venv_path)
    
    # Check if pip exists in venv
    if not os.path.exists(pip_path):
        print(f"Using python -m pip instead of direct pip path")
        pip_command = [python_path, "-m", "pip"]
    else:
        pip_command = [pip_path]
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call(pip_command + ["install", "--upgrade", "pip"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Install requirements
        print("Installing packages from requirements.txt...")
        subprocess.check_call(pip_command + ["install", "-r", "requirements.txt"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        print("This might be due to WebContainer limitations.")
        return False
    except FileNotFoundError:
        print("✗ Pip not found in virtual environment")
        return False

def check_camera(venv_path):
    """Check if camera is accessible using the virtual environment"""
    print("Checking camera access...")
    python_path = get_venv_python(venv_path)
    
    # Create a simple camera test script
    test_script = """
import sys
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("CAMERA_OK")
        else:
            print("CAMERA_NO_FRAMES")
    else:
        print("CAMERA_NOT_ACCESSIBLE")
except ImportError:
    print("OPENCV_NOT_AVAILABLE")
except Exception as e:
    print(f"CAMERA_ERROR: {e}")
"""
    
    try:
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        
        if output == "CAMERA_OK":
            print("✓ Camera is accessible!")
            return True
        elif output == "CAMERA_NOT_ACCESSIBLE":
            print("✗ Camera found but not accessible")
            return False
        elif output == "CAMERA_NO_FRAMES":
            print("✗ Camera accessible but cannot read frames")
            return False
        elif output == "OPENCV_NOT_AVAILABLE":
            print("✗ OpenCV not installed")
            return False
        else:
            print(f"✗ Camera check failed: {output}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Camera check timed out")
        return False
    except Exception as e:
        print(f"✗ Error checking camera: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")

def create_activation_scripts(venv_path):
    """Create convenient activation scripts"""
    
    # Create activation script for Unix/Linux/macOS
    activate_script = f"""#!/bin/bash
# Body Measurement Predictor - Virtual Environment Activation
echo "Activating Body Measurement Predictor virtual environment..."
source {venv_path}/bin/activate
echo "Virtual environment activated!"
echo "Available commands:"
echo "  python train_model.py     - Train the AI model"
echo "  python webcam_predictor.py - Start prediction session"
echo "  deactivate                - Exit virtual environment"
"""
    
    with open("activate_env.sh", "w") as f:
        f.write(activate_script)
    
    # Make it executable
    try:
        os.chmod("activate_env.sh", 0o755)
    except:
        pass
    
    # Create activation script for Windows
    activate_bat = f"""@echo off
REM Body Measurement Predictor - Virtual Environment Activation
echo Activating Body Measurement Predictor virtual environment...
call {venv_path}\\Scripts\\activate.bat
echo Virtual environment activated!
echo Available commands:
echo   python train_model.py     - Train the AI model
echo   python webcam_predictor.py - Start prediction session
echo   deactivate                - Exit virtual environment
"""
    
    with open("activate_env.bat", "w") as f:
        f.write(activate_bat)
    
    print("✓ Created activation scripts: activate_env.sh and activate_env.bat")

def main():
    """Main setup function"""
    print("=== Body Measurement Prediction System Setup ===\n")
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    if not venv_path:
        print("Setup failed. Could not create virtual environment.")
        return
    
    print()
    
    # Install requirements
    packages_installed = install_requirements(venv_path)
    
    print()
    
    # Check camera (only if packages were installed)
    camera_ok = False
    if packages_installed:
        camera_ok = check_camera(venv_path)
    else:
        print("Skipping camera check (packages not installed)")
    
    print()
    
    # Setup directories
    setup_directories()
    
    print()
    
    # Create activation scripts
    create_activation_scripts(venv_path)
    
    print("\n=== Setup Summary ===")
    print(f"✓ Virtual environment created at '{venv_path}'")
    print(f"{'✓' if packages_installed else '✗'} Requirements {'installed' if packages_installed else 'installation failed'}")
    print(f"{'✓' if camera_ok else '✗'} Camera {'accessible' if camera_ok else 'not accessible'}")
    print("✓ Directories created")
    print("✓ Activation scripts created")
    
    print("\n=== Next Steps ===")
    
    if packages_installed:
        print("1. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print(f"   {venv_path}\\Scripts\\activate")
            print("   OR run: activate_env.bat")
        else:  # Unix/Linux/macOS
            print(f"   source {venv_path}/bin/activate")
            print("   OR run: ./activate_env.sh")
        
        print("2. Train the model:")
        print("   python train_model.py")
        print("3. Start making predictions:")
        print("   python webcam_predictor.py")
        
        if not camera_ok:
            print("\n⚠️  Camera issues detected. Please ensure:")
            print("   - Camera is connected and not used by other applications")
            print("   - Camera permissions are granted")
            print("   - You're running in a local environment (not WebContainer)")
    else:
        print("⚠️  Package installation failed. This might be due to:")
        print("   - WebContainer environment limitations")
        print("   - Network connectivity issues")
        print("   - Missing system dependencies")
        print("\nTo run locally:")
        print("1. Download this project to your local machine")
        print("2. Run this setup script in a local Python environment")

if __name__ == "__main__":
    main()