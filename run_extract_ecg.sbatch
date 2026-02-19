#!/bin/bash
## SBATCH wrapper to run the ECG extraction script with wfdb available.
## It creates a persistent virtualenv under the project (first run will install wfdb).

#######################
## SBATCH directives
#######################
#SBATCH -J extract_ecg
#SBATCH -p CPU-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -A cis250019p
#SBATCH -o /ocean/projects/cis250085p/shared/ecg-arrhythmia-classification/slurm-%j.out
#SBATCH -e /ocean/projects/cis250085p/shared/ecg-arrhythmia-classification/slurm-%j.err

set -euo pipefail

PROJECT_DIR=/ocean/projects/cis250085p/shared/ecg-arrhythmia-classification
VENV_DIR=$PROJECT_DIR/.venv_wfdb
PYTHON=${PYTHON:-python3}

echo "Job: $SLURM_JOB_ID running on $(hostname)"
echo "Using Python: $(which $PYTHON 2>/dev/null || echo 'not-found')"

# Create persistent virtualenv (only on first run)
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv at $VENV_DIR and installing wfdb..."
  $PYTHON -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip setuptools wheel
  pip install wfdb
  deactivate
else
  echo "Using existing virtualenv at $VENV_DIR"
fi

# Activate and run
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python "$PROJECT_DIR/src/data/extract_ecg_signals.py" \
  --base-dir "$PROJECT_DIR/physionet.org/files/ecg-arrhythmia/1.0.0" \
  --out-dir "$PROJECT_DIR/processed_signals" \
  --workers ${SLURM_CPUS_PER_TASK:-8}

deactivate

echo "Extraction finished: $(date)"
