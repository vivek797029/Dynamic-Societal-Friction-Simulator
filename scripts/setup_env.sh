#!/bin/bash
# ============================================================
# Setup script for Dynamic Society Friction Simulator
# ============================================================
set -e

echo "=== DSFS Environment Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install project in editable mode with all dependencies
echo "Installing DSFS and dependencies..."
pip install -e ".[dev,eval]"

# Check GPU availability
echo ""
echo "=== GPU Check ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected. Training will be very slow.')
    print('Consider using Google Colab or a cloud GPU instance.')
"

# Generate initial synthetic data
echo ""
echo "=== Generating Initial Training Data ==="
python3 -m src.model.data_pipeline

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with: source .venv/bin/activate"
echo "Run 'dsfs --help' to see available commands."
