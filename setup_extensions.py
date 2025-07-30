from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
import os
import platform

# Check if CUDA is available
USE_CUDA = torch.cuda.is_available()

# Detect platform
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
IS_MAC = platform.system() == 'Darwin'

ext_modules = []

# Platform-specific compiler flags
if IS_WINDOWS:
    cpp_extra_compile_args = ['/O2', '/openmp']
    cpp_extra_link_args = []
elif IS_LINUX:
    cpp_extra_compile_args = ['-O3', '-fopenmp', '-march=native', '-fPIC']
    cpp_extra_link_args = ['-fopenmp']
elif IS_MAC:
    # macOS doesn't have OpenMP by default
    cpp_extra_compile_args = ['-O3', '-std=c++14']
    cpp_extra_link_args = []
    # Check if OpenMP is available via Homebrew
    if os.path.exists('/usr/local/opt/libomp'):
        cpp_extra_compile_args.extend(['-Xpreprocessor', '-fopenmp'])
        cpp_extra_link_args.extend(['-lomp'])
else:
    cpp_extra_compile_args = ['-O3']
    cpp_extra_link_args = []

# C++ extension
cpp_extension = CppExtension(
    'mcts_cpp',
    ['cpp_extensions/mcts_cpp.cpp'],
    extra_compile_args=cpp_extra_compile_args,
    extra_link_args=cpp_extra_link_args,
    include_dirs=[torch.utils.cpp_extension.include_paths()]
)
ext_modules.append(cpp_extension)

# CUDA extension (only if CUDA is available)
if USE_CUDA:
    # Get CUDA compute capability
    cuda_capability = None
    if torch.cuda.is_available():
        cuda_capability = torch.cuda.get_device_capability()
        sm_arch = f"sm_{cuda_capability[0]}{cuda_capability[1]}"
    else:
        sm_arch = "sm_70"  # Default fallback
    
    print(f"Building with CUDA support for architecture: {sm_arch}")
    
    # CUDA-specific flags
    nvcc_flags = [
        '-O3',
        '--use_fast_math',
        f'-arch={sm_arch}',
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    
    if IS_LINUX:
        nvcc_flags.extend(['-Xcompiler', '-fPIC'])
    
    cuda_extension = CUDAExtension(
        'mcts_cuda',
        ['cuda_extensions/mcts_cuda.cu'],
        extra_compile_args={
            'cxx': ['-O3'] if not IS_WINDOWS else ['/O2'],
            'nvcc': nvcc_flags
        },
        include_dirs=[torch.utils.cpp_extension.include_paths()]
    )
    ext_modules.append(cuda_extension)
else:
    print("CUDA not available, building CPU-only version")

# Print build configuration
print(f"Platform: {platform.system()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
if USE_CUDA:
    print(f"CUDA: {torch.version.cuda}")

setup(
    name='mcts_extensions',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)