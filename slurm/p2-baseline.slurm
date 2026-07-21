#!/bin/bash
#SBATCH --job-name=ge-p2-baseline
#SBATCH --output=results/p2-baseline.%j.out
#SBATCH --error=results/p2-baseline.%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --gpus=1
#SBATCH --constraint=geforce_rtx_2080
# P2.5 pilot: the baseline condition end-to-end on real GPU. Loads Qwen2.5-3B
# (from configs/baseline.toml), runs edge_existence x adjacency over N graphs,
# writes attempt-level JSONL under results/p2-baseline/, and prints a per-encoding
# summary. Resumable: re-submitting skips already-done instances. Keep mem/time
# modest -- shared quota. First run also downloads the model (~6GB) to HF_HOME.

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source slurm/_activate.sh

# Optional config arg, so the same job runs the single slice or the matrix smoke:
#   sbatch slurm/p2-baseline.slurm                          # edge_existence x adjacency
#   sbatch slurm/p2-baseline.slurm configs/pilot-matrix.toml  # all 3x3, small N
CONFIG="${1:-configs/baseline.toml}"

echo "host=$(hostname)"
echo "config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

python -u -m gedebate.eval.runner --config "$CONFIG"
echo "Done. See the out_dir named in $CONFIG (JSONL + manifest) and the summary above."
