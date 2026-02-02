#!/bin/bash

# PyTorch Installation Script with GPU/CUDA Support
# Automatically detects platform and installs appropriate PyTorch version
# Supports: Mac (MPS), NVIDIA GPU (CUDA), DGX Spark (CUDA 13.0)

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

# Check for NVIDIA GPU and CUDA version
NVIDIA_GPU=false
CUDA_MAJOR_VERSION=""
if command -v nvidia-smi &> /dev/null; then
    NVIDIA_GPU=true
    echo "NVIDIA GPU: Detected"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
    
    # Get CUDA version from nvidia-smi
    CUDA_VERSION_FULL=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' 2>/dev/null || echo "")
    if [[ -n "$CUDA_VERSION_FULL" ]]; then
        CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION_FULL" | cut -d. -f1)
        echo "CUDA Version: $CUDA_VERSION_FULL (Major: $CUDA_MAJOR_VERSION)"
    fi
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
    if [[ "$ARCH" == "aarch64" ]]; then
        # ARM64 with NVIDIA GPU (DGX Spark uses CUDA 13.0)
        if [[ "$CUDA_MAJOR_VERSION" == "13" ]]; then
            echo "Installing PyTorch for DGX Spark (CUDA 13.0, aarch64)..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
        else
            echo "Installing PyTorch for ARM64 + NVIDIA GPU (CUDA 12.x)..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        fi
    elif [[ "$ARCH" == "x86_64" ]]; then
        # x86_64 with NVIDIA GPU - use CUDA 12.1 wheels
        echo "Installing PyTorch with CUDA support (x86_64)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
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
