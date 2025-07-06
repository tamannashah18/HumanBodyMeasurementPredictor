#!/usr/bin/env node
/**
 * Setup script for AWS BodyM Dataset Training (Node.js version)
 * This replaces the Python setup script to avoid Python installation issues
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

function createVirtualEnvironment() {
    console.log("Creating virtual environment for AWS training...");
    const venvPath = "venv_aws";
    
    if (fs.existsSync(venvPath)) {
        console.log(`Virtual environment '${venvPath}' already exists.`);
        return venvPath;
    }
    
    try {
        execSync(`python3 -m venv ${venvPath}`, { stdio: 'inherit' });
        console.log(`✓ Virtual environment created at '${venvPath}'`);
        return venvPath;
    } catch (error) {
        console.log(`✗ Error creating virtual environment: ${error.message}`);
        return null;
    }
}

function getVenvPython(venvPath) {
    return process.platform === 'win32' 
        ? path.join(venvPath, 'Scripts', 'python.exe')
        : path.join(venvPath, 'bin', 'python');
}

function getVenvPip(venvPath) {
    return process.platform === 'win32'
        ? path.join(venvPath, 'Scripts', 'pip.exe')
        : path.join(venvPath, 'bin', 'pip');
}

function installAWSRequirements(venvPath) {
    console.log("Installing AWS training requirements...");
    
    const pipPath = getVenvPip(venvPath);
    const pythonPath = getVenvPython(venvPath);
    
    const pipCommand = fs.existsSync(pipPath) ? pipPath : `${pythonPath} -m pip`;
    
    try {
        console.log("Upgrading pip...");
        execSync(`${pipCommand} install --upgrade pip`, { stdio: 'inherit' });
        
        console.log("Installing AWS training packages...");
        execSync(`${pipCommand} install -r requirements_aws.txt`, { stdio: 'inherit' });
        
        console.log("✓ All AWS packages installed successfully!");
        return true;
    } catch (error) {
        console.log(`✗ Error installing packages: ${error.message}`);
        return false;
    }
}

function testAWSConnection(venvPath) {
    console.log("Testing AWS S3 connection...");
    const pythonPath = getVenvPython(venvPath);
    
    const testScript = `
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
`;
    
    try {
        const result = execSync(`${pythonPath} -c "${testScript}"`, { 
            encoding: 'utf8', 
            timeout: 30000 
        }).trim();
        
        if (result === "AWS_CONNECTION_OK") {
            console.log("✓ AWS S3 connection successful!");
            return true;
        } else if (result === "AWS_BUCKET_EMPTY") {
            console.log("✓ AWS S3 connected but bucket appears empty");
            return true;
        } else if (result.includes("IMPORT_ERROR")) {
            console.log(`✗ Missing dependencies: ${result}`);
            return false;
        } else {
            console.log(`✗ AWS connection failed: ${result}`);
            return false;
        }
    } catch (error) {
        console.log(`✗ Error testing AWS connection: ${error.message}`);
        return false;
    }
}

function createAWSActivationScripts(venvPath) {
    // Unix/Linux/macOS script
    const activateScript = `#!/bin/bash
# AWS BodyM Training Environment
echo "Activating AWS BodyM training environment..."
source ${venvPath}/bin/activate
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
`;
    
    fs.writeFileSync("activate_aws.sh", activateScript);
    
    try {
        fs.chmodSync("activate_aws.sh", 0o755);
    } catch (error) {
        // Ignore chmod errors on systems that don't support it
    }
    
    // Windows script
    const activateBat = `@echo off
REM AWS BodyM Training Environment
echo Activating AWS BodyM training environment...
call ${venvPath}\\Scripts\\activate.bat
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
`;
    
    fs.writeFileSync("activate_aws.bat", activateBat);
    
    console.log("✓ Created AWS activation scripts: activate_aws.sh and activate_aws.bat");
}

function main() {
    console.log("=== AWS BodyM Dataset Training Setup ===\n");
    
    // Create virtual environment
    const venvPath = createVirtualEnvironment();
    if (!venvPath) {
        console.log("Setup failed. Could not create virtual environment.");
        return;
    }
    
    console.log();
    
    // Install AWS requirements
    const packagesInstalled = installAWSRequirements(venvPath);
    
    console.log();
    
    // Test AWS connection
    let awsOk = false;
    if (packagesInstalled) {
        awsOk = testAWSConnection(venvPath);
    } else {
        console.log("Skipping AWS test (packages not installed)");
    }
    
    console.log();
    
    // Create activation scripts
    createAWSActivationScripts(venvPath);
    
    console.log("\n=== Setup Summary ===");
    console.log(`✓ AWS virtual environment created at '${venvPath}'`);
    console.log(`${packagesInstalled ? '✓' : '✗'} AWS requirements ${packagesInstalled ? 'installed' : 'installation failed'}`);
    console.log(`${awsOk ? '✓' : '✗'} AWS S3 connection ${awsOk ? 'successful' : 'failed'}`);
    console.log("✓ AWS activation scripts created");
    
    console.log("\n=== Next Steps ===");
    
    if (packagesInstalled && awsOk) {
        console.log("🎉 Setup completed successfully!");
        console.log("\n1. Activate the AWS environment:");
        if (process.platform === 'win32') {
            console.log("   activate_aws.bat");
        } else {
            console.log("   ./activate_aws.sh");
        }
        
        console.log("\n2. Test AWS data loading:");
        console.log("   python aws_data_loader.py");
        
        console.log("\n3. Train with real BodyM data:");
        console.log("   python train_real_model.py");
        
        console.log("\n4. Test predictions:");
        console.log("   python webcam_predictor.py");
        
        console.log("\n📊 Training Tips:");
        console.log("   - Start with 100-500 images for testing");
        console.log("   - Use 1000+ images for better accuracy");
        console.log("   - Training time depends on dataset size");
        console.log("   - Monitor GPU usage if available");
        
    } else {
        console.log("⚠️  Setup incomplete. Issues detected:");
        if (!packagesInstalled) {
            console.log("   - Package installation failed");
        }
        if (!awsOk) {
            console.log("   - AWS S3 connection failed");
        }
        
        console.log("\nTroubleshooting:");
        console.log("   - Ensure stable internet connection");
        console.log("   - Try running in local environment");
        console.log("   - Check firewall/proxy settings");
    }
}

if (require.main === module) {
    main();
}