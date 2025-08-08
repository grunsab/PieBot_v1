import sys
from setuptools import setup, Extension
import pybind11

# Define the C++ extension module
ext_modules = [
    Extension(
        'mcts_cpp_engine',
        ['mcts_cpp.cpp', 'bindings.cpp'],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=['/std:c++17', '/DPYBIND11_DETAILED_ERROR_MESSAGES'] if sys.platform == 'win32' else ['-std=c++17', '-DPYBIND11_DETAILED_ERROR_MESSAGES'],
    ),
]

setup(
    name='mcts_cpp_engine',
    version='1.0',
    author='Your Name',
    description='C++ MCTS engine with Python bindings',
    ext_modules=ext_modules,
)
