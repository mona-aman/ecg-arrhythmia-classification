#!/bin/bash
#SBATCH -J model_job          # Job name
#SBATCH -p GPU-shared         # Partition (queue)
#SBATCH --gpus=1              # Number of GPUs (can request more if needed)
#SBATCH -t 8:00export:00            # Run time (hh:mm:ss)
#SBATCH -A cis250019p         # Allocation name
#SBATCH -o slurm-%j.out       # Standard output file
#SBATCH -e slurm-%j.err       # Standard error file

# Load any modules you need
# module load cuda/11.7.1        # Adjust CUDA version as needed
# module load python/3.9.12      # Adjust Python version as needed
module load AI/pytorch_23.02-1.13.1-py3
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MPLCONFIGDIR=$PWD/.mplconfig

cd /ocean/projects/cis250085p/shared/ecg-arrhythmia-classification
python scripts/train_model.py \
    --model-type 2d \
    --model-size small \
    --representation-type cwt \
    --data-dir data/processed_batched \
    --learning-rate 1e-4 \
    --no-mixed-precision \
    --batch-size 2 \
    --num-epochs 50