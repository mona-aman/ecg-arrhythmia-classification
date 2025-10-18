#!/bin/bash
#SBATCH --job-name=ecg_1d_transformer
#SBATCH --output=outputs/test_runs/1d_training_%j.out
#SBATCH --error=outputs/test_runs/1d_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-shared
#SBATCH --time=02:00:00
#SBATCH --account=cis250019p
#SBATCH --mem=32G

# Load modules
module load anaconda3/2024.10-1

# Navigate to project directory
cd /ocean/projects/cis250019p/maman/ecg-arrhythmia-classification

# Activate environment if needed
# source activate your_env_name

# Create output directory
mkdir -p outputs/test_runs

# Run training
python scripts/train_model.py --model-type 1d --config config/test_config.yaml

echo "Training completed!"