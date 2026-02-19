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

source /ocean/projects/cis250085p/shared/ecg-arrhythmia-classification/.venv/bin/activate

module load AI/pytorch_23.02-1.13.1-py3
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MPLCONFIGDIR=$PWD/.mplconfig

# cd /ocean/projects/cis250085p/shared/ecg-arrhythmia-classification
# python scripts/train_phase2_fusion.py \
#   --aligned-encoder-1d outputs/contrastive_alignment_stft/aligned_encoder_1d.pth \
#   --aligned-encoder-2d outputs/contrastive_alignment_stft/aligned_encoder_2d.pth \
#   --data-dir data/processed_batched \
#   --representation-type stft \
#   --output-dir outputs/phase2_fusion_tuned_headonly \
#   --device cuda \
#   --freeze-encoders \
#   --batch-size 64 \
#   --num-epochs 60 \
#   --learning-rate 2e-4 \
#   --dropout 0.15 \
#   --patience 12


python scripts/train_phase2_fusion.py \
   --aligned-encoder-1d outputs/contrastive_alignment_stft/aligned_encoder_1d.pth \
   --aligned-encoder-2d outputs/contrastive_alignment_stft/aligned_encoder_2d.pth \
   --data-dir data/processed_batched \
   --representation-type stft \
   --batch-size 32 \
   --num-epochs 50 \
   --learning-rate 1e-4 \
   --patience 5 \
   --feature-dim 384 \
   --num-classes 4 \
   --num-heads 8 \
   --dropout 0.10 \
   --freeze-encoders \
   --output-dir outputs/phase2_fusion_working_2 \
   --device cuda



  # --output-dir outputs/fusion_stable_fair \
  # --num-epochs 40 \
  # --batch-size 32 \
  # --feature-dim 384 --num-classes 4 --num-heads 8 --dropout 0.15 \
  # --encoder-lr 0          \
  # --head-lr 1e-4          \
  # --encoder-wd 0.01 --head-wd 0.01 \
  # --label-smoothing 0.05  \
  # --grad-accum-steps 1    \
  # --lr-warmup-epochs 0    \
  # --patience 20           \
  # --use-ema 



