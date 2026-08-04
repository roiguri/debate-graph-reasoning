#!/bin/bash
#SBATCH --job-name=ge-smoke
#SBATCH --output=results/smoke.%j.out
#SBATCH --error=results/smoke.%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --gpus=1
#SBATCH --constraint=geforce_rtx_2080
# De-risk run: pins the RTX 2080 nodes (Turing, native fp16, 11GB). Loads a
# small instruct model, generates on one prompt, reaches the token-accounting
# check, writes results/smoke.json. Keep mem/time modest -- shared quota.

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source slurm/_activate.sh

echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

python -u scripts/smoke.py --out results/smoke.json
echo "Done. See results/smoke.json"
