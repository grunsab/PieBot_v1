# Makefile for building MCTS extensions on Linux

# Detect Python and PyTorch paths
PYTHON := python3
PYTHON_CONFIG := python3-config
TORCH_PATH := $(shell $(PYTHON) -c "import torch; print(torch.__path__[0])")
CUDA_PATH ?= /usr/local/cuda

# Compiler settings
CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc

# Check if CUDA is available
CUDA_AVAILABLE := $(shell $(PYTHON) -c "import torch; print(int(torch.cuda.is_available()))")

# Python and PyTorch includes
PYTHON_INCLUDES := $(shell $(PYTHON_CONFIG) --includes)
TORCH_INCLUDES := -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include

# Compiler flags
CXXFLAGS := -O3 -march=native -fPIC -std=c++14 $(PYTHON_INCLUDES) $(TORCH_INCLUDES)
LDFLAGS := -shared

# OpenMP flags
OPENMP_FLAGS := -fopenmp
OPENMP_LIBS := -fopenmp

# Check if OpenMP is available
HAS_OPENMP := $(shell echo | $(CXX) -fopenmp -dM -E - 2>/dev/null | grep -c _OPENMP)
ifeq ($(HAS_OPENMP), 1)
    CXXFLAGS += $(OPENMP_FLAGS)
    LDFLAGS += $(OPENMP_LIBS)
else
    $(warning OpenMP not available, building without parallel support)
endif

# CUDA flags
ifeq ($(CUDA_AVAILABLE), 1)
    CUDA_ARCH := $(shell $(PYTHON) -c "import torch; cc = torch.cuda.get_device_capability(); print(f'sm_{cc[0]}{cc[1]}')")
    NVCCFLAGS := -O3 --use_fast_math -arch=$(CUDA_ARCH) -Xcompiler -fPIC
    CUDA_INCLUDES := -I$(CUDA_PATH)/include
    CUDA_LIBS := -L$(CUDA_PATH)/lib64 -lcudart
endif

# Targets
.PHONY: all clean test

all: mcts_cpp.so
ifeq ($(CUDA_AVAILABLE), 1)
all: mcts_cuda.so
endif

# C++ extension
mcts_cpp.so: cpp_extensions/mcts_cpp.cpp
	@echo "Building C++ extension..."
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o build/mcts_cpp.o
	$(CXX) $(LDFLAGS) build/mcts_cpp.o -o $@
	@echo "✓ Built $@"

# CUDA extension
ifeq ($(CUDA_AVAILABLE), 1)
mcts_cuda.so: cuda_extensions/mcts_cuda.cu
	@echo "Building CUDA extension..."
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(TORCH_INCLUDES) $(CUDA_INCLUDES) -c $< -o build/mcts_cuda.o
	$(NVCC) -shared build/mcts_cuda.o $(CUDA_LIBS) -o $@
	@echo "✓ Built $@"
else
mcts_cuda.so:
	@echo "CUDA not available, skipping CUDA extension"
endif

# Test build
test: all
	@echo "Testing extensions..."
	@$(PYTHON) -c "import mcts_cpp; print('✓ C++ extension loaded')" || echo "✗ Failed to load C++ extension"
ifeq ($(CUDA_AVAILABLE), 1)
	@$(PYTHON) -c "import mcts_cuda; print('✓ CUDA extension loaded')" || echo "✗ Failed to load CUDA extension"
endif

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@rm -f *.so
	@rm -f *.pyd
	@echo "✓ Clean complete"

# Help
help:
	@echo "MCTS Extensions Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make         - Build all extensions"
	@echo "  make test    - Build and test extensions"
	@echo "  make clean   - Remove build artifacts"
	@echo "  make help    - Show this help"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_PATH    - Path to CUDA installation (default: /usr/local/cuda)"
	@echo "  CXX          - C++ compiler (default: g++)"
	@echo ""
	@echo "Current settings:"
	@echo "  Python:      $(PYTHON)"
	@echo "  PyTorch:     $(TORCH_PATH)"
	@echo "  CUDA:        $(if $(filter 1,$(CUDA_AVAILABLE)),Available ($(CUDA_ARCH)),Not available)"
	@echo "  OpenMP:      $(if $(filter 1,$(HAS_OPENMP)),Available,Not available)"

# Installation info
install-info:
	@echo "Installation Instructions for Linux:"
	@echo ""
	@echo "1. Install build dependencies:"
	@echo "   Ubuntu/Debian:"
	@echo "     sudo apt install build-essential python3-dev libomp-dev"
	@echo "   Fedora/RHEL:"
	@echo "     sudo dnf install gcc-c++ python3-devel libomp-devel"
	@echo ""
	@echo "2. Install CUDA (if using GPU):"
	@echo "   Check PyTorch CUDA version:"
	@echo "     python3 -c 'import torch; print(torch.version.cuda)'"
	@echo "   Download matching CUDA toolkit from:"
	@echo "     https://developer.nvidia.com/cuda-toolkit"
	@echo ""
	@echo "3. Build extensions:"
	@echo "     make"
	@echo ""
	@echo "4. Test installation:"
	@echo "     make test"