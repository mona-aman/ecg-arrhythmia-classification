#!/bin/bash
# GPU Interactive Training Script
# Run this once your interactive GPU session starts

echo "=== GPU ECG Training Session ==="
echo "Starting at: $(date)"

# Setup environment
echo "Setting up environment..."
module load anaconda3/2024.10-1
cd /ocean/projects/cis250019p/maman/ecg-arrhythmia-classification

# Check GPU availability
echo "Checking GPU..."
nvidia-smi
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Run 1D baseline training with all data
echo "Starting 1D transformer training with full dataset..."
python scripts/train_model.py \
    --config config/gpu_config.yaml \
    --model-type 1d \
    --data-dir data/processed \
    --output-dir outputs/gpu_runs/1d_baseline \
    --device cuda

echo "Training completed at: $(date)"
echo "Check outputs/gpu_runs/1d_baseline/ for results"