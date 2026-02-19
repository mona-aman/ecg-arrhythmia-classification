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

cd /ocean/projects/cis250085p/shared/ecg-arrhythmia-classification
export PYTHONPATH=/ocean/projects/cis250085p/shared/ecg-arrhythmia-classification:$PYTHONPATH
python scripts/export_saliency_1d2d_fusion_minimal.py \
  --ckpt-1d outputs/1d_small_1762289628/best_model.pth \
  --ckpt-2d outputs/2d_stft_small/best_model.pth \
  --ckpt-fusion-stable outputs/phase2_fusion_working/best_fusion_model.pth \
  --data-dir data/processed_batched \
  --repr stft \
  --device cuda \
  --dim-1d 4 --dim-2d 4 \
  --fs 250 \
  --sigma 5 --thr 0.5 --max-scan 100000 \
  --focus-class AFIB \
  --allow-mismatch \
  --out outputs/saliency/afib_single_lastone_1.png

#   --ckpt-1d outputs/1d_small_1762289628/best_model.pth \
#   --ckpt-2d outputs/2d_stft_small/best_model.pth \
#   --ckpt-fusion-stable outputs/fusion_stft_stable/best_fusion_model.pth \
#   --ckpt-fusion-tuned outputs/fusion_stft_tuned_v2/best_fusion_model.pth \
#   --data-dir data/processed_batched \
#   --repr stft \
#   --dim-1d 4 --dim-2d 4 \
#   --fs 250 \
# --allow-mismatch \
#   --out outputs/saliency/fig_saliency_all_4_models.png \
#   --sigma 5 --thr 0.5 --max-scan 15000