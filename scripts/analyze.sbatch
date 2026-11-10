#!/usr/bin/env bash
#SBATCH --job-name=be_diff_analysis
#SBATCH --account=INF26_biophys
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=analysis-%j.out
#SBATCH --error=analysis-%j.err
set -euo pipefail
: "${BE_RUNTIME_ENV:?Set BE_RUNTIME_ENV to your working activation script}"
: "${BE_INPUTS:?Set BE_INPUTS to the verified external inputs}"
: "${BE_RUN_DIR:?Set BE_RUN_DIR to the campaign output directory}"
cd "${SLURM_SUBMIT_DIR:?Submit from the cloned repository root}"
source "$BE_RUNTIME_ENV"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HUMMER_CACHE_DIR="$BE_RUN_DIR/runtime/hummer-cache"
# primary first; suite includes lag/bin/discard, matched prep and five held-out folds.
# Re-submit the same variant after a walltime interruption; completed chunks are reused.
python -m be_pilot analyze --phase all --variant "${BE_ANALYSIS_VARIANT:-primary}" --workers 4
