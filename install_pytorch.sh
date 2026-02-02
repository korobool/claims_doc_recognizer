#!/bin/bash

# PyTorch Installation Script with GPU/CUDA Support
# Automatically detects platform and installs appropriate PyTorch version

set -e

echo "============================================================"
echo "PyTorch Installation Script with GPU Support"
echo "============================================================"

# Detect architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Detect OS
OS=$(uname -s)
echo "OS: $OS"

# Check for NVIDIA GPU
NVIDIA_GPU=false
if command -v nvidia-smi &> /dev/null; then
    NVIDIA_GPU=true
    echo "NVIDIA GPU: Detected"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
    
    # Get CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "Driver Version: $CUDA_VERSION"
else
    echo "NVIDIA GPU: Not detected"
fi

# Check for Apple Silicon
APPLE_SILICON=false
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    APPLE_SILICON=true
    echo "Apple Silicon: Detected (MPS acceleration available)"
fi

echo ""
echo "Uninstalling existing PyTorch..."
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

echo ""
echo "Installing PyTorch..."

# Installation logic based on platform
if [[ "$NVIDIA_GPU" == true ]]; then
    if [[ "$ARCH" == "x86_64" ]]; then
        # x86_64 with NVIDIA GPU - use CUDA wheels
        echo "Installing PyTorch with CUDA support (x86_64)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$ARCH" == "aarch64" ]]; then
        # ARM64 with NVIDIA GPU (e.g., Jetson, DGX Spark)
        echo "Installing PyTorch for ARM64 + NVIDIA GPU..."
        echo ""
        echo "For ARM64 + NVIDIA GPU, try one of these options:"
        echo ""
        echo "Option 1: NVIDIA PyPI (recommended for DGX/Jetson):"
        echo "  pip install torch --index-url https://pypi.nvidia.com"
        echo ""
        echo "Option 2: Build from source or use NGC container"
        echo "  See: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch"
        echo ""
        echo "Option 3: Use conda with NVIDIA channel:"
        echo "  conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia"
        echo ""
        echo "Attempting NVIDIA PyPI installation..."
        pip install torch torchvision --index-url https://pypi.nvidia.com || {
            echo ""
            echo "NVIDIA PyPI failed. Trying standard PyTorch..."
            pip install torch torchvision
        }
    else
        echo "Unknown architecture with NVIDIA GPU. Installing standard PyTorch..."
        pip install torch torchvision
    fi
elif [[ "$APPLE_SILICON" == true ]]; then
    # Apple Silicon - MPS support is built into standard PyTorch
    echo "Installing PyTorch with MPS support (Apple Silicon)..."
    pip install torch torchvision
elif [[ "$ARCH" == "x86_64" ]]; then
    # x86_64 without NVIDIA - CPU only
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    # Fallback
    echo "Installing standard PyTorch..."
    pip install torch torchvision
fi

echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda if torch.version.cuda else \"N/A\"}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon): Available')
else:
    print('Acceleration: CPU only')
"

echo ""
echo "============================================================"
echo "Installation complete!"
echo "============================================================"
