#!/usr/bin/env python3
"""
Setup script for AWS BodyM Dataset Training
"""

import subprocess
import sys
import os
import venv

def create_virtual_environment():
    """Create a virtual environment for AWS training"""
    venv_path = "venv_aws"
    
    if os.path.exists(venv_path):
        print(f"Virtual environment '{venv_path}' already exists.")
        return venv_path
    
    print("Creating virtual environment for AWS training...")
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

def install_aws_requirements(venv_path):
    """Install AWS-specific requirements"""
    print("Installing AWS training requirements...")
    
    pip_path = get_venv_pip(venv_path)
    python_path = get_venv_python(venv_path)
    
    if not os.path.exists(pip_path):
        pip_command = [python_path, "-m", "pip"]
    else:
        pip_command = [pip_path]
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call(pip_command + ["install", "--upgrade", "pip"])
        
        # Install AWS requirements
        print("Installing AWS training packages...")
        subprocess.check_call(pip_command + ["install", "-r", "requirements_aws.txt"])
        
        print("✓ All AWS packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def test_aws_connection(venv_path):
    """Test AWS S3 connection"""
    print("Testing AWS S3 connection...")
    python_path = get_venv_python(venv_path)
    
    test_script = """
import sys
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    
    # Test S3 connection
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    response = s3_client.list_objects_v2(Bucket='amazon-bodym', MaxKeys=1)
    
    if 'Contents' in response or 'KeyCount' in response:
        print("AWS_CONNECTION_OK")
    else:
        print("AWS_BUCKET_EMPTY")
        
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
except Exception as e:
    print(f"AWS_ERROR: {e}")
"""
    
    try:
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        
        if output == "AWS_CONNECTION_OK":
            print("✓ AWS S3 connection successful!")
            return True
        elif output == "AWS_BUCKET_EMPTY":
            print("✓ AWS S3 connected but bucket appears empty")
            return True
        elif "IMPORT_ERROR" in output:
            print(f"✗ Missing dependencies: {output}")
            return False
        else:
            print(f"✗ AWS connection failed: {output}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ AWS connection test timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing AWS connection: {e}")
        return False

def create_aws_activation_scripts(venv_path):
    """Create activation scripts for AWS training environment"""
    
    # Unix/Linux/macOS script
    activate_script = f"""#!/bin/bash
# AWS BodyM Training Environment
echo "Activating AWS BodyM training environment..."
source {venv_path}/bin/activate
echo "AWS training environment activated!"
echo ""
echo "Available commands:"
echo "  python aws_data_loader.py     - Test AWS data loading"
echo "  python train_real_model.py    - Train with real BodyM data"
echo "  python webcam_predictor.py    - Test predictions"
echo "  deactivate                    - Exit environment"
echo ""
echo "Dataset info:"
echo "  Bucket: s3://amazon-bodym"
echo "  Access: Public (no credentials needed)"
"""
    
    with open("activate_aws.sh", "w") as f:
        f.write(activate_script)
    
    try:
        os.chmod("activate_aws.sh", 0o755)
    except:
        pass
    
    # Windows script
    activate_bat = f"""@echo off
REM AWS BodyM Training Environment
echo Activating AWS BodyM training environment...
call {venv_path}\\Scripts\\activate.bat
echo AWS training environment activated!
echo.
echo Available commands:
echo   python aws_data_loader.py     - Test AWS data loading
echo   python train_real_model.py    - Train with real BodyM data
echo   python webcam_predictor.py    - Test predictions
echo   deactivate                    - Exit environment
echo.
echo Dataset info:
echo   Bucket: s3://amazon-bodym
echo   Access: Public (no credentials needed)
"""
    
    with open("activate_aws.bat", "w") as f:
        f.write(activate_bat)
    
    print("✓ Created AWS activation scripts: activate_aws.sh and activate_aws.bat")

def main():
    """Main setup function for AWS training"""
    print("=== AWS BodyM Dataset Training Setup ===\n")
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    if not venv_path:
        print("Setup failed. Could not create virtual environment.")
        return
    
    print()
    
    # Install AWS requirements
    packages_installed = install_aws_requirements(venv_path)
    
    print()
    
    # Test AWS connection
    aws_ok = False
    if packages_installed:
        aws_ok = test_aws_connection(venv_path)
    else:
        print("Skipping AWS test (packages not installed)")
    
    print()
    
    # Create activation scripts
    create_aws_activation_scripts(venv_path)
    
    print("\n=== Setup Summary ===")
    print(f"✓ AWS virtual environment created at '{venv_path}'")
    print(f"{'✓' if packages_installed else '✗'} AWS requirements {'installed' if packages_installed else 'installation failed'}")
    print(f"{'✓' if aws_ok else '✗'} AWS S3 connection {'successful' if aws_ok else 'failed'}")
    print("✓ AWS activation scripts created")
    
    print("\n=== Next Steps ===")
    
    if packages_installed and aws_ok:
        print("🎉 Setup completed successfully!")
        print("\n1. Activate the AWS environment:")
        if os.name == 'nt':  # Windows
            print("   activate_aws.bat")
        else:  # Unix/Linux/macOS
            print("   ./activate_aws.sh")
        
        print("\n2. Test AWS data loading:")
        print("   python aws_data_loader.py")
        
        print("\n3. Train with real BodyM data:")
        print("   python train_real_model.py")
        
        print("\n4. Test predictions:")
        print("   python webcam_predictor.py")
        
        print("\n📊 Training Tips:")
        print("   - Start with 100-500 images for testing")
        print("   - Use 1000+ images for better accuracy")
        print("   - Training time depends on dataset size")
        print("   - Monitor GPU usage if available")
        
    else:
        print("⚠️  Setup incomplete. Issues detected:")
        if not packages_installed:
            print("   - Package installation failed")
        if not aws_ok:
            print("   - AWS S3 connection failed")
        
        print("\nTroubleshooting:")
        print("   - Ensure stable internet connection")
        print("   - Try running in local environment")
        print("   - Check firewall/proxy settings")

if __name__ == "__main__":
    main()