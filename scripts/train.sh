#!/bin/bash
# ============================================================
# Training launch script with recommended settings
# ============================================================
set -e

# Activate environment
source .venv/bin/activate

# Set environment variables for efficient training
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="dsfs-model-gcp"

echo "=== Starting DSFS Model Training ==="
echo "Config: configs/model_config.yaml"
echo ""

# Launch training
python3 -m src.model.trainer

echo ""
echo "=== Training Complete ==="
echo "Adapter saved to: outputs/checkpoints/final_adapter/"
