#!/bin/bash
# One-time environment setup on the MLWG2026 SLURM cluster.
# Run ON A CLIENT NODE (after ssh to the cluster), from the project directory.
# Installs conda + the inference stack on the netapp path (NOT home -- home has
# no quota for a conda env + model weights).
#
#   cp .env.example .env      # then edit NETAPP (+ HF_TOKEN if needed)
#   bash slurm/setup_env.sh
#
set -euo pipefail

# NETAPP comes from .env (same source the jobs use).
if [ -f ./.env ]; then set -a; source ./.env; set +a; fi
: "${NETAPP:?NETAPP not set -- copy .env.example to .env and set your netapp path}"

ENV_PREFIX="$NETAPP/envs/gedebate"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "netapp=$NETAPP"
echo "project=$PROJECT_DIR"
mkdir -p "$NETAPP/hf_cache" "$NETAPP/envs" "$NETAPP/pkgs"

# --- 1. Ensure conda (install Miniconda to netapp if absent) ---------------
if [ ! -d "$NETAPP/miniconda3" ]; then
  echo "Installing Miniconda -> $NETAPP/miniconda3 ..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc_$USER.sh
  bash /tmp/mc_$USER.sh -b -p "$NETAPP/miniconda3"
  rm -f /tmp/mc_$USER.sh
fi
# shellcheck disable=SC1091
source "$NETAPP/miniconda3/etc/profile.d/conda.sh"
conda config --add pkgs_dirs "$NETAPP/pkgs" 2>/dev/null || true

# Newer conda blocks installs from Anaconda's default channels until their Terms
# of Service are accepted (no-op if already accepted or if conda predates `tos`).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# --- 2. Create env on netapp (by prefix, so it never lands in home) --------
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"

# --- 3. Install the package + inference stack ------------------------------
pip install --no-cache-dir -e "$PROJECT_DIR[inference]"

echo
echo "Setup complete."
echo "  env:     $ENV_PREFIX"
echo "  HF_HOME: $NETAPP/hf_cache"
python -c "import torch, transformers, gedebate; print('torch', torch.__version__, '| transformers', transformers.__version__, '| gedebate', gedebate.__version__)"
echo "Next:  sbatch slurm/smoke.slurm"
