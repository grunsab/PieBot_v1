#!/bin/bash
# Build script for Linux CUDA systems

echo "========================================"
echo "Building MCTS Extensions for Linux"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
echo "Checking build requirements..."

# Check for g++
if ! command -v g++ &> /dev/null; then
    echo -e "${RED}ERROR: g++ compiler not found!${NC}"
    echo "Please install build-essential:"
    echo "  Ubuntu/Debian: sudo apt-get install build-essential"
    echo "  Fedora/RHEL: sudo dnf install gcc-c++"
    exit 1
else
    echo -e "${GREEN}✓ g++ found:${NC} $(g++ --version | head -n1)"
fi

# Check for nvcc (CUDA compiler)
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}WARNING: CUDA compiler (nvcc) not found!${NC}"
    echo "CUDA extensions will not be built."
    echo "To install CUDA toolkit:"
    echo "  1. Check PyTorch CUDA version: python -c 'import torch; print(torch.version.cuda)'"
    echo "  2. Download from: https://developer.nvidia.com/cuda-toolkit"
    echo "  3. Add to PATH: export PATH=/usr/local/cuda/bin:\$PATH"
    CUDA_AVAILABLE=0
else
    echo -e "${GREEN}✓ CUDA found:${NC}"
    nvcc --version
    CUDA_AVAILABLE=1
fi

# Check for Python and PyTorch
echo ""
echo "Checking Python environment..."
if ! python3 -c "import torch" &> /dev/null; then
    echo -e "${RED}ERROR: PyTorch not found!${NC}"
    echo "Please install PyTorch with CUDA support:"
    echo "  pip3 install torch torchvision torchaudio"
    exit 1
else
    echo -e "${GREEN}✓ PyTorch found:${NC}"
    python3 -c "import torch; print(f'  Version: {torch.__version__}')"
    python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        python3 -c "import torch; print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
    fi
fi

# Check for OpenMP
echo ""
echo "Checking for OpenMP..."
if echo | g++ -fopenmp -dM -E - | grep -q _OPENMP; then
    echo -e "${GREEN}✓ OpenMP supported${NC}"
else
    echo -e "${YELLOW}WARNING: OpenMP not available${NC}"
    echo "Install OpenMP for better CPU performance:"
    echo "  Ubuntu/Debian: sudo apt-get install libomp-dev"
    echo "  Fedora/RHEL: sudo dnf install libomp-devel"
fi

# Create directories if needed
echo ""
echo "Setting up directories..."
mkdir -p cpp_extensions
mkdir -p cuda_extensions

# Set compiler flags for optimization
export CFLAGS="-O3 -march=native -mtune=native"
export CXXFLAGS="-O3 -march=native -mtune=native"

# Build extensions
echo ""
echo "Building extensions..."
python3 setup_extensions.py build_ext --inplace

# Check build results
echo ""
echo "========================================"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo "========================================"
    echo ""
    echo "Extensions built:"
    ls -la *.so 2>/dev/null | grep -E "(mcts_cpp|mcts_cuda)"
    
    echo ""
    echo "To test the installation:"
    echo "  python3 test_cuda_optimizations.py --model your_model.pt"
    
    # Set library path if needed
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        echo ""
        echo "If you encounter library loading errors, add CUDA to library path:"
        echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    fi
else
    echo -e "${RED}Build failed!${NC}"
    echo "========================================"
    echo "Check the error messages above."
    echo ""
    echo "Common issues:"
    echo "1. CUDA version mismatch with PyTorch"
    echo "2. Missing development headers"
    echo "3. Incompatible g++ version"
    exit 1
fi

# Make scripts executable
chmod +x build_linux.sh
chmod +x test_cuda_optimizations.py
chmod +x playchess_cuda.py
chmod +x UCI_engine_cuda.py

echo ""
echo "Setup complete!"