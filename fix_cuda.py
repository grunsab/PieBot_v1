#!/usr/bin/env python3
"""
CUDA Fix Script for RTX 5080 on Vast.AI
This script diagnoses and fixes CUDA detection issues.
"""

import subprocess
import sys
import os

def run_command(cmd, check=False):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Error running: {cmd}")
            print(f"Error: {result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"Exception running {cmd}: {e}")
        return None

def diagnose_cuda():
    """Diagnose CUDA installation issues."""
    print("=" * 60)
    print("CUDA Diagnostic Report")
    print("=" * 60)
    
    # Check nvidia-smi
    print("\n1. Checking NVIDIA driver...")
    nvidia_smi = run_command("nvidia-smi")
    if nvidia_smi:
        print("✓ NVIDIA driver is installed")
        print(nvidia_smi[:500])  # Print first 500 chars
    else:
        print("✗ nvidia-smi not found or not working")
        print("  This means NVIDIA drivers are not properly installed")
    
    # Check CUDA toolkit
    print("\n2. Checking CUDA toolkit...")
    nvcc = run_command("nvcc --version")
    if nvcc:
        print("✓ CUDA toolkit is installed")
        print(nvcc)
    else:
        print("✗ CUDA toolkit (nvcc) not found")
    
    # Check PyTorch CUDA support
    print("\n3. Checking PyTorch CUDA support...")
    pytorch_check = run_command("python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda if hasattr(torch.version, \"cuda\") else \"N/A\"}\")'")
    if pytorch_check:
        print(pytorch_check)
        if "CUDA available: False" in pytorch_check:
            print("✗ PyTorch does not have CUDA support")
        else:
            print("✓ PyTorch has CUDA support")
    
    # Check environment variables
    print("\n4. Checking environment variables...")
    cuda_paths = []
    for var in ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']:
        val = os.environ.get(var, "Not set")
        print(f"  {var}: {val}")
        if var == 'LD_LIBRARY_PATH' and val != "Not set":
            cuda_paths.extend([p for p in val.split(':') if 'cuda' in p.lower()])
    
    # Check for CUDA libraries
    print("\n5. Checking for CUDA libraries...")
    lib_paths = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/opt/cuda/lib64"
    ]
    for path in lib_paths:
        if os.path.exists(path):
            print(f"  ✓ Found CUDA library path: {path}")
            # Check for specific libraries
            for lib in ['libcudart.so', 'libcublas.so', 'libcudnn.so']:
                lib_path = os.path.join(path, lib)
                if os.path.exists(lib_path) or os.path.exists(lib_path + ".11") or os.path.exists(lib_path + ".12"):
                    print(f"    ✓ {lib} found")
                else:
                    print(f"    ✗ {lib} not found")

def fix_pytorch_cuda():
    """Attempt to fix PyTorch CUDA installation."""
    print("\n" + "=" * 60)
    print("Attempting to fix PyTorch CUDA support...")
    print("=" * 60)
    
    # First, check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    # Get CUDA version from nvidia-smi
    cuda_version = None
    nvidia_smi = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if nvidia_smi:
        # Try to determine CUDA version
        cuda_version_output = run_command("nvidia-smi | grep 'CUDA Version' | awk '{print $9}'")
        if cuda_version_output:
            cuda_version = cuda_version_output.replace(".", "").replace(":", "")[:3]  # e.g., "121" for 12.1
            print(f"Detected CUDA version: {cuda_version}")
    
    if not cuda_version:
        # Try alternative method
        nvcc_version = run_command("nvcc --version | grep 'release' | awk '{print $6}' | cut -d',' -f1")
        if nvcc_version:
            cuda_version = nvcc_version.replace(".", "")[:3]
            print(f"Detected CUDA version from nvcc: {cuda_version}")
    
    if not cuda_version:
        print("Could not detect CUDA version. Assuming CUDA 12.1")
        cuda_version = "121"
    
    # Determine the correct PyTorch installation command
    if cuda_version.startswith("12"):
        torch_index = "https://download.pytorch.org/whl/cu121"
        cuda_suffix = "cu121"
    elif cuda_version.startswith("11"):
        torch_index = "https://download.pytorch.org/whl/cu118"
        cuda_suffix = "cu118"
    else:
        print(f"Unsupported CUDA version: {cuda_version}")
        return False
    
    print(f"\nReinstalling PyTorch with CUDA {cuda_suffix} support...")
    
    # Uninstall existing PyTorch
    print("1. Uninstalling existing PyTorch...")
    run_command("pip3 uninstall -y torch torchvision torchaudio")
    
    # Install PyTorch with CUDA support
    print(f"2. Installing PyTorch with CUDA {cuda_suffix} support...")
    install_cmd = f"pip3 install torch torchvision torchaudio --index-url {torch_index}"
    print(f"Running: {install_cmd}")
    result = run_command(install_cmd, check=True)
    
    if result is not None:
        print("✓ PyTorch installation completed")
        
        # Verify installation
        print("\n3. Verifying installation...")
        verify = run_command("python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\")'")
        if verify and "CUDA available: True" in verify:
            print("✓ PyTorch CUDA support successfully installed!")
            print(verify)
            return True
        else:
            print("✗ PyTorch still cannot detect CUDA")
            if verify:
                print(verify)
    else:
        print("✗ PyTorch installation failed")
    
    return False

def setup_environment():
    """Set up environment variables for CUDA."""
    print("\n" + "=" * 60)
    print("Setting up CUDA environment variables...")
    print("=" * 60)
    
    cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda"
    ]
    
    cuda_home = None
    for path in cuda_paths:
        if os.path.exists(path):
            cuda_home = path
            break
    
    if cuda_home:
        print(f"Found CUDA installation at: {cuda_home}")
        
        # Create shell script for environment setup
        env_script = f"""#!/bin/bash
# CUDA environment setup for RTX 5080
export CUDA_HOME={cuda_home}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA environment variables set:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  PATH includes $CUDA_HOME/bin"
echo "  LD_LIBRARY_PATH includes $CUDA_HOME/lib64"
"""
        
        with open("setup_cuda_env.sh", "w") as f:
            f.write(env_script)
        
        print("Created setup_cuda_env.sh")
        print("Run: source setup_cuda_env.sh")
        
        # Also export for current session
        os.environ['CUDA_HOME'] = cuda_home
        os.environ['PATH'] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_home}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        return True
    else:
        print("✗ Could not find CUDA installation")
        return False

def main():
    print("RTX 5080 CUDA Fix Script for Vast.AI")
    print("=" * 60)
    
    # Step 1: Diagnose
    diagnose_cuda()
    
    # Step 2: Ask user if they want to attempt fix
    print("\n" + "=" * 60)
    response = input("Do you want to attempt to fix PyTorch CUDA support? (y/n): ")
    
    if response.lower() == 'y':
        # Step 3: Setup environment
        setup_environment()
        
        # Step 4: Fix PyTorch
        success = fix_pytorch_cuda()
        
        if success:
            print("\n" + "=" * 60)
            print("SUCCESS! PyTorch CUDA support has been fixed.")
            print("You can now run your training script.")
            print("\nIMPORTANT: If you're in a Vast.AI instance, you may need to:")
            print("1. Source the environment: source setup_cuda_env.sh")
            print("2. Restart your Python kernel/session")
        else:
            print("\n" + "=" * 60)
            print("FAILED to fix PyTorch CUDA support automatically.")
            print("\nManual steps to try:")
            print("1. Check NVIDIA driver: sudo apt-get update && sudo apt-get install nvidia-driver-535")
            print("2. Install CUDA toolkit: sudo apt-get install cuda-toolkit-12-1")
            print("3. Reinstall PyTorch: pip3 install torch --index-url https://download.pytorch.org/whl/cu121")
            print("4. Restart the instance")
    else:
        print("\nDiagnostic complete. No changes made.")

if __name__ == "__main__":
    main()