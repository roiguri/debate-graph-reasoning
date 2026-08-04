#!/bin/bash
#SBATCH --job-name=ge-matrix
#SBATCH --output=results/logs/%x.%A_%a.out
#SBATCH --error=results/logs/%x.%A_%a.err
#SBATCH --partition=studentkillable
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --gpus=1
#SBATCH --constraint=geforce_rtx_2080
# Full 3x3 baseline at N=200, sharded across a SLURM job array. Each array task
# runs one shard (--shard i/n derived from the array env), resumably. Submit as:
#   mkdir -p results/logs                     # log dir must exist before submit
#   sbatch --array=0-7 slurm/matrix.slurm     # default: seed 7 (configs/matrix.toml)
#
# Which dataset/seed to run is chosen at submit time via the CONFIG env var
# (default configs/matrix.toml), so this one script drives every seed. Pass a
# matching --job-name so each seed gets its own log files (%x is the job name):
#   sbatch --export=ALL,CONFIG=configs/seed11.toml --job-name=ge-seed11 --array=0-7 slurm/matrix.slurm
#   sbatch --export=ALL,CONFIG=configs/seed13.toml --job-name=ge-seed13 --array=0-7 slurm/matrix.slurm
#
# The QOS caps concurrent GPUs (measured: gres/gpu=12, not the 24 this comment used
# to claim), but the real limit is the hardware: only s-004 and s-005 carry
# geforce_rtx_2080, 16 GPUs between them, shared with every other student. Expect
# shards to trickle rather than all start at once, and throttle with `%k` (e.g. %4)
# to stay a good citizen. Killed shards rerun and skip already-done instances.
#
# `--cpus-per-task` is 2, not 4: the 2080 nodes fragment badly. s-004 has sat with 3
# free GPUs and 0 free CPUs while s-005 had 8 free CPUs and 0 free GPUs, so a 4-CPU
# ask could not land on either. Inference needs one GPU and very little CPU, so the
# smaller ask fits gaps the larger one cannot. It does not touch the model, the
# decoding, or determinism.
# Non-array `sbatch` runs shard 0/1 (the whole matrix in one job).

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source slurm/_activate.sh

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
CONFIG="${CONFIG:-configs/matrix.toml}"   # override at submit: --export=ALL,CONFIG=configs/seed11.toml

echo "host=$(hostname) array=${SLURM_ARRAY_JOB_ID:-NA} task=${SHARD}/${NSHARDS} config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

python -u -m gedebate.eval.runner --config "${CONFIG}" --shard "${SHARD}/${NSHARDS}"
echo "Done shard ${SHARD}/${NSHARDS} of ${CONFIG}. Pull results locally with scripts/pull_results.sh"
