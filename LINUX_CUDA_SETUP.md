# Linux CUDA Setup Guide for PyTorch AlphaZero

This guide covers setting up the CUDA-optimized MCTS on Linux systems with NVIDIA GPUs.

## Prerequisites

### 1. System Requirements
- Linux (Ubuntu 20.04+, Debian 11+, Fedora 34+, or similar)
- NVIDIA GPU with compute capability 7.0+ (RTX 2000 series or newer)
- GCC/G++ compiler
- Python 3.8+

### 2. Install Build Tools

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install build-essential python3-dev
sudo apt install libomp-dev  # For OpenMP support
```

**Fedora/RHEL/CentOS:**
```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel
sudo dnf install libomp-devel  # For OpenMP support
```

**Arch Linux:**
```bash
sudo pacman -S base-devel python
sudo pacman -S openmp
```

### 3. Install NVIDIA Drivers

**Ubuntu/Debian (recommended method):**
```bash
# Add NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-530  # Replace with your version

# Reboot
sudo reboot
```

**Using NVIDIA's official installer:**
```bash
# Download from https://www.nvidia.com/Download/index.aspx
chmod +x NVIDIA-Linux-x86_64-*.run
sudo ./NVIDIA-Linux-x86_64-*.run
```

### 4. Install CUDA Toolkit

**Check PyTorch CUDA version first:**
```python
import torch
print(torch.version.cuda)  # e.g., 11.8 or 12.1
```

**Install matching CUDA Toolkit:**

For CUDA 11.8:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-11-8
```

For CUDA 12.1:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-12-1
```

**Add CUDA to PATH:**
```bash
# Add to ~/.bashrc or ~/.profile
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Reload
source ~/.bashrc
```

**Verify installation:**
```bash
nvcc --version
nvidia-smi
```

## Building the Extensions

### 1. Clone and Navigate to Repository
```bash
cd pytorch-alpha-zero
```

### 2. Build Extensions
```bash
# Make build script executable
chmod +x build_linux.sh

# Run build script
./build_linux.sh
```

Or manually:
```bash
python3 setup_extensions.py build_ext --inplace
```

### 3. Verify Build
```bash
# Check if extensions were built
ls -la *.so | grep mcts

# Should see:
# mcts_cpp.cpython-38-x86_64-linux-gnu.so
# mcts_cuda.cpython-38-x86_64-linux-gnu.so
```

## Testing the Installation

### Quick Test
```bash
python3 test_cuda_optimizations.py --model weights/AlphaZeroNet_20x256.pt
```

### Performance Benchmark
```bash
# Compare implementations
python3 compare_all_mcts.py --model weights/AlphaZeroNet_20x256.pt --rollouts 1000
```

## Usage

### Playing Chess
```bash
# With CUDA optimizations
python3 playchess_cuda.py --model weights/model.pt --rollouts 1000 --threads 16 --verbose

# Profile mode
python3 playchess_cuda.py --model weights/model.pt --mode p --rollouts 2000 --threads 32
```

### UCI Engine
```bash
# Run UCI engine
python3 UCI_engine_cuda.py --model weights/model.pt --threads 32

# Use with chess GUI (e.g., cutechess)
cutechess-cli -engine cmd="python3 UCI_engine_cuda.py --model weights/model.pt --threads 32" \
              -engine cmd=stockfish -each tc=60+1 -games 100
```

## Performance Tuning for Linux

### 1. CPU Governor
Set CPU to performance mode:
```bash
# Check current governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set to performance
sudo cpupower frequency-set -g performance
```

### 2. GPU Settings
```bash
# Set GPU to maximum performance
sudo nvidia-smi -pm 1  # Enable persistence mode
sudo nvidia-smi -pl 300  # Set power limit (adjust for your GPU)

# For compute-only mode (better performance)
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
```

### 3. Memory Settings
Edit `/etc/sysctl.conf`:
```bash
# Increase shared memory
kernel.shmmax = 68719476736
kernel.shmall = 4294967296

# Apply changes
sudo sysctl -p
```

### 4. Process Priority
```bash
# Run with high priority
nice -n -10 python3 playchess_cuda.py --model weights/model.pt
```

## Optimization Parameters

Edit `MCTS_cuda_optimized.py`:

```python
# For Linux servers with powerful GPUs
BATCH_SIZE = 512  # Increase for A100/H100
MAX_BATCH_WAIT_TIME = 0.002
POSITION_CACHE_SIZE = 100000  # Use more RAM
MOVE_CACHE_SIZE = 100000
```

## Troubleshooting

### CUDA Not Found
```bash
# Check CUDA installation
which nvcc
ldconfig -p | grep cuda

# If not found, add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Library Loading Errors
```bash
# Missing libraries
ldd mcts_cuda.*.so | grep "not found"

# Install missing libraries
sudo apt install libcudnn8  # For cuDNN
```

### Permission Errors
```bash
# NVIDIA driver permissions
sudo usermod -a -G video $USER
# Log out and back in
```

### Build Errors
```bash
# Clear build cache
rm -rf build/
rm *.so

# Rebuild with verbose output
python3 setup_extensions.py build_ext --inplace --verbose
```

## Docker Support

### Dockerfile for CUDA PyTorch AlphaZero
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libomp-dev \
    git

# Install PyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone and build
WORKDIR /app
COPY . .
RUN chmod +x build_linux.sh && ./build_linux.sh

# Run
CMD ["python3", "playchess_cuda.py", "--model", "weights/model.pt"]
```

### Run with Docker
```bash
# Build
docker build -t alphazero-cuda .

# Run with GPU
docker run --gpus all -it alphazero-cuda
```

## Performance Expectations

On Linux with modern NVIDIA GPUs:

| GPU | Expected NPS | Speedup vs CPU |
|-----|--------------|----------------|
| RTX 3060 | 1500-2500 | 3-4x |
| RTX 3080 | 2500-4000 | 4-6x |
| RTX 4090 | 4000-8000 | 6-10x |
| A100 | 6000-12000 | 10-15x |

Factors affecting performance:
- GPU memory bandwidth
- CPU-GPU PCIe bandwidth
- Number of CPU cores
- Memory speed
- Model size (FP16 vs FP32)

## Advanced Linux Optimizations

### 1. NUMA Awareness
For multi-socket systems:
```bash
# Check NUMA topology
numactl --hardware

# Run on specific NUMA node
numactl --cpunodebind=0 --membind=0 python3 playchess_cuda.py
```

### 2. Huge Pages
```bash
# Enable transparent huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled

# Or configure explicit huge pages
echo 1000 > /proc/sys/vm/nr_hugepages
```

### 3. CPU Affinity
```bash
# Pin to specific cores
taskset -c 0-15 python3 playchess_cuda.py
```

### 4. GPU Direct
For multi-GPU systems:
```bash
# Enable GPUDirect
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1530  # Lock GPU clocks
```

## Monitoring Performance

### GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Detailed monitoring
nvidia-smi dmon -s pucvmet
```

### System Monitoring
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs

# Monitor everything
htop  # CPU/Memory
iotop  # Disk I/O
nvidia-smi  # GPU
```

This setup will give you optimal performance on Linux systems with NVIDIA GPUs!