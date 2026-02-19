#!/bin/bash
#SBATCH -J model_job          # Job name
#SBATCH -p GPU-shared         # Partition (queue)
#SBATCH --gpus=1              # Number of GPUs (can request more if needed)
#SBATCH -t 2:00:00            # Run time (hh:mm:ss)
#SBATCH -A cis250019p         # Allocation name
#SBATCH -o slurm-%j.out       # Standard output file
#SBATCH -e slurm-%j.err       # Standard error file

# Load any modules you need
# module load cuda/11.7.1        # Adjust CUDA version as needed
# module load python/3.9.12      # Adjust Python version as needed

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

source /ocean/projects/cis250085p/shared/ecg-arrhythmia-classification/.venv/bin/activate

module load AI/pytorch_23.02-1.13.1-py3
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MPLCONFIGDIR=$PWD/.mplconfig
cd /ocean/projects/cis250085p/shared/ecg-arrhythmia-classification

# Run Phase 1 training
echo "Starting Phase 1: Contrastive Alignment Training..."

python scripts/train_phase1_contrastive.py \
    --model-1d-path outputs/1d_small_1762289628/best_model.pth \
    --model-2d-path outputs/2d_stft_small/best_model.pth \
    --data-dir data/processed_batched \
    --representation-type stft \
    --batch-size 64 \
    --num-workers 4 \
    --projection-dim 128 \
    --temperature 0.5 \
    --num-epochs 30 \
    --learning-rate 1e-3 \
    --patience 10 \
    --device cuda \
    --output-dir outputs/contrastive_alignment_stft_FIXED
# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Phase 1 Training Completed Successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Phase 1 Training Failed!"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi