#!/usr/bin/env python3
"""
Setup script for Body Measurement Prediction System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    print("⚠️  Note: This is running in a WebContainer environment.")
    print("⚠️  Python package installation via pip is not supported.")
    print("⚠️  Only Python standard library modules are available.")
    print("✓ Skipping package installation (WebContainer limitation)")
    return True

def check_camera():
    """Check if camera is accessible"""
    print("Checking camera access...")
    print("⚠️  Note: Camera access is not available in WebContainer environment.")
    print("⚠️  This feature would work in a local Python environment.")
    print("✓ Skipping camera check (WebContainer limitation)")
    return False

def setup_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")

def main():
    """Main setup function"""
    print("=== Body Measurement Prediction System Setup ===\n")
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please install requirements manually.")
        return
    
    print()
    
    # Check camera
    camera_ok = check_camera()
    
    print()
    
    # Setup directories
    setup_directories()
    
    print("\n=== Setup Summary ===")
    print("✓ Requirements installation skipped (WebContainer limitation)")
    print(f"✗ Camera not accessible (WebContainer limitation)")
    print("✓ Directories created")
    
    print("\n=== WebContainer Environment Notes ===")
    print("• This environment only supports Python standard library")
    print("• External packages like OpenCV, TensorFlow are not available")
    print("• Camera access is not supported")
    print("• For full functionality, run this project in a local Python environment")
    
    print("\n=== To run locally ===")
    print("1. Install Python 3.8+ on your local machine")
    print("2. Run 'pip install -r requirements.txt'")
    print("3. Run 'python train_model.py' to train the AI model")
    print("4. Run 'python webcam_predictor.py' to start making predictions")

if __name__ == "__main__":
    main()