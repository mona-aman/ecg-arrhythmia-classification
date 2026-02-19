#!/bin/bash
#SBATCH -J download_ecg_job    # Job name
#SBATCH -p CPU-shared         # Partition (queue) - request CPU partition (adjust if needed)
#SBATCH --ntasks=1            # number of tasks
#SBATCH --cpus-per-task=4     # CPUs per task (adjust as needed)
#SBATCH -t 2:00:00            # Run time (hh:mm:ss)
#SBATCH -A cis250019p         # Allocation name
#SBATCH -o slurm-%j.out       # Standard output file
#SBATCH -e slurm-%j.err       # Standard error file

set -euo pipefail

TARGET_DIR=/ocean/projects/cis250085p/shared/ecg-arrhythmia-classification
URL=https://physionet.org/files/ecg-arrhythmia/1.0.0/

echo "Starting download job"
echo "Job: $SLURM_JOB_ID on $(hostname)" || true
echo "Start: $(date)"

# Ensure target directory exists and move into it
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Recommended wget invocation (resume-safe, tidy local layout)
# --cut-dirs=2 removes the leading "files/ecg-arrhythmia" path components
# so you'll get e.g. ./1.0.0/... under $TARGET_DIR
WGET_CMD=(wget -r -np -c -N -nH --cut-dirs=2 --reject "index.html*" \
  --tries=0 --retry-connrefused --wait=1 --random-wait \
  "$URL")

echo "Running: ${WGET_CMD[*]}"
"${WGET_CMD[@]}"

echo "Finished: $(date)"

## Notes:
# - Re-run this script (sbatch the same file) to resume where it left off. wget -c and -N
#   will resume partial files and skip files that are already fully downloaded.
# - If a particular file is corrupted or stuck, remove that file and re-run; wget will
#   download it from scratch.
# - If you want to limit bandwidth add e.g. --limit-rate=1m to the wget options.
