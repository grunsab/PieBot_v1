@echo off
REM Build script for Windows CUDA systems

echo ========================================
echo Building MCTS Extensions for Windows
echo ========================================

REM Check for Visual Studio
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Visual Studio C++ compiler not found!
    echo Please install Visual Studio 2019 or later with C++ tools
    echo Or run this from "x64 Native Tools Command Prompt"
    exit /b 1
)

REM Check for CUDA
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: CUDA compiler not found!
    echo CUDA extensions will not be built
    echo Install CUDA Toolkit for GPU acceleration
) else (
    echo CUDA found: 
    nvcc --version
)

REM Check Python and PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"
if %errorlevel% neq 0 (
    echo ERROR: PyTorch not found!
    exit /b 1
)

REM Create directories if needed
if not exist cpp_extensions mkdir cpp_extensions
if not exist cuda_extensions mkdir cuda_extensions

REM Build extensions
echo.
echo Building extensions...
python setup_extensions.py build_ext --inplace

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Build completed successfully!
    echo ========================================
    echo.
    echo Extensions built:
    dir *.pyd 2>nul
    
    echo.
    echo To test the installation:
    echo   python test_cuda_optimizations.py --model your_model.pt
) else (
    echo.
    echo ========================================
    echo Build failed!
    echo ========================================
    echo Check the error messages above
)

pause