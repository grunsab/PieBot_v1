from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
import os

# Check if CUDA is available
USE_CUDA = torch.cuda.is_available()

ext_modules = []

# C++ extension
cpp_extension = CppExtension(
    'mcts_cpp',
    ['cpp_extensions/mcts_cpp.cpp'],
    extra_compile_args={'cxx': ['-O3', '-fopenmp']},
    extra_link_args=['-fopenmp']
)
ext_modules.append(cpp_extension)

# CUDA extension (only if CUDA is available)
if USE_CUDA:
    cuda_extension = CUDAExtension(
        'mcts_cuda',
        ['cuda_extensions/mcts_cuda.cu'],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math', '-arch=sm_70']  # Adjust architecture as needed
        }
    )
    ext_modules.append(cuda_extension)
    print("Building with CUDA support")
else:
    print("CUDA not available, building CPU-only version")

setup(
    name='mcts_extensions',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)