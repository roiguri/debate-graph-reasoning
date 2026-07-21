#!/bin/bash
# Shared cluster activation, sourced by the slurm jobs. All machine-specific
# values (NETAPP path, HF_TOKEN) come from the gitignored .env so nothing
# host-specific is ever committed. Copy .env.example -> .env first.

# Load secrets + NETAPP from .env.
if [ -f ./.env ]; then set -a; source ./.env; set +a; fi
: "${NETAPP:?NETAPP not set -- copy .env.example to .env and set your netapp path}"

# Model/dataset cache on netapp (NOT home). Must be set before python imports
# transformers, so we export it in the shell here.
export HF_HOME="$NETAPP/hf_cache"

# Force eager execution. The pinned RTX 2080 nodes have Triton, but if a job
# falls back to a Pascal node (Titan Xp, no Triton), torch.compile'd kernels
# crash the Inductor backend. Eager is correctness-safe and plenty fast for
# inference. Remove if you never see Pascal fallback and want compile speed.
export TORCHDYNAMO_DISABLE=1

# conda activate trips `set -u`; keep nounset off during activation.
set +u
source "$NETAPP/miniconda3/etc/profile.d/conda.sh"
conda activate "$NETAPP/envs/gedebate"
