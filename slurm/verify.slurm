#!/bin/bash
#SBATCH --job-name=ge-verify
#SBATCH --output=results/logs/%x.%j.out
#SBATCH --error=results/logs/%x.%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --gpus=1
#SBATCH --constraint=geforce_rtx_2080
# Reproducibility spot check on the GPU: re-run an even-spaced sample of persisted
# rows through the real model and confirm the parsed answers still match. Usage:
#   mkdir -p results/logs
#   sbatch slurm/verify.slurm         # default sample of 100
#   sbatch slurm/verify.slurm 200     # custom sample size
# Reads results/main/baseline (the reference rows) and data/main.jsonl. Sources
# _activate.sh so HF_HOME points at the netapp cache (no model re-download).

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source slurm/_activate.sh

K="${1:-100}"
echo "host=$(hostname) verify-sample K=$K"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

python -u -m gedebate.eval.runner --config configs/matrix.toml --verify-sample "$K"
echo "Done. See the reproduce count above."
