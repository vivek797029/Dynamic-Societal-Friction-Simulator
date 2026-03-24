#!/bin/bash
# ============================================================
# Dynamic Society Friction Simulator — GCP VM Setup Script
# ============================================================
# Run this on a fresh GCP VM (Deep Learning VM recommended).
# Tested on: Ubuntu 22.04 with NVIDIA drivers pre-installed.
#
# Usage:
#   chmod +x scripts/setup_gcp.sh
#   ./scripts/setup_gcp.sh
# ============================================================

set -euo pipefail

echo "============================================"
echo "  DSFS — GCP Setup Script"
echo "============================================"

# ---- Check GPU ----
echo ""
echo "[1/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed."
fi

# ---- System dependencies ----
echo ""
echo "[2/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip python3-venv

# ---- Create virtual environment ----
echo ""
echo "[3/6] Setting up Python environment..."
VENV_DIR="$HOME/dsfs-venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created venv at $VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ---- Install project ----
echo ""
echo "[4/6] Installing project and dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install -e ".[dev,eval]" -q

# Try installing flash-attn for A100/H100
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
if echo "$GPU_NAME" | grep -qiE "A100|H100"; then
    echo "Installing Flash Attention 2 for $GPU_NAME..."
    pip install flash-attn --no-build-isolation -q || echo "Flash Attention install failed (non-fatal)"
fi

# ---- Generate data (if not exists) ----
echo ""
echo "[5/6] Checking training data..."
if [ -f "data/processed/train.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < data/processed/train.jsonl)
    echo "Training data already exists: $TRAIN_COUNT samples"
else
    echo "Generating training data (50K samples)..."
    dsfs generate-data --num-samples 50000 --output-dir data/processed --seed 42
fi

# ---- Ready ----
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To start training:"
echo "  source $VENV_DIR/bin/activate"
echo "  dsfs train --config configs/model_config.yaml --verbose"
echo ""
echo "To resume interrupted training:"
echo "  dsfs train --resume --verbose"
echo ""
echo "To run in background (survives SSH disconnect):"
echo "  nohup dsfs train --verbose > training.log 2>&1 &"
echo "  tail -f training.log"
echo ""
echo "To monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
