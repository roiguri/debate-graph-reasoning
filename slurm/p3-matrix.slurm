#!/bin/bash
#SBATCH --job-name=ge-p3-matrix
#SBATCH --output=results/logs/%x.%A_%a.out
#SBATCH --error=results/logs/%x.%A_%a.err
#SBATCH --partition=studentkillable
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --gpus=1
#SBATCH --constraint=geforce_rtx_2080
# P3: full 3x3 baseline at N=200, sharded across a SLURM job array. Each array task
# runs one shard (--shard i/n derived from the array env), resumably. Submit as:
#   mkdir -p results/logs                     # log dir must exist before submit
#   sbatch --array=0-7%3 slurm/p3-matrix.slurm
# %3 caps concurrency at 3 -- safe under an unknown per-student quota (extra shards
# pend, they don't error); raise once `sacctmgr` confirms the real limit. Killed
# shards rerun and skip already-done instances. Non-array `sbatch` runs shard 0/1
# (the whole matrix in one job).

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source slurm/_activate.sh

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "host=$(hostname) array=${SLURM_ARRAY_JOB_ID:-NA} task=${SHARD}/${NSHARDS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

python -u -m gedebate.eval.runner --config configs/p3-matrix.toml --shard "${SHARD}/${NSHARDS}"
echo "Done shard ${SHARD}/${NSHARDS}. Pull results locally with scripts/pull_results.sh"
