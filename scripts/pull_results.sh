#!/bin/bash
# Pull run results FROM the cluster into the local repo's results/ for analysis.
# Analysis (report/show_results) needs no torch, so it runs locally after this.
#
# Run from your LOCAL project root:
#   NETAPP=/home/yandex/MLWG2026/<your-username> bash scripts/pull_results.sh
#   NETAPP=... bash scripts/pull_results.sh p3-matrix   # just one run subdir
#
# Requires the TAU VPN up and an ssh host alias `slurm-client` (see cluster-runbook.md).
set -eo pipefail
cd "$(dirname "$0")/.."

: "${NETAPP:?set NETAPP to your cluster netapp path, e.g. /home/yandex/MLWG2026/<user>}"
HOST="${SLURM_HOST:-slurm-client}"
REMOTE="$HOST:$NETAPP/graph-encodings-with-debate/results/"
SUBDIR="${1:-}"   # optional: restrict to one run dir (e.g. p3-matrix)

mkdir -p results
if [ -n "$SUBDIR" ]; then
  echo "Pulling results/$SUBDIR/ from $HOST ..."
  mkdir -p "results/$SUBDIR"
  rsync -av "$REMOTE$SUBDIR/" "results/$SUBDIR/"
else
  # Data only: skip the loose job logs (results/logs, *.out/*.err).
  echo "Pulling all run data from $HOST (excluding logs) ..."
  rsync -av --exclude 'logs/' --exclude '*.out' --exclude '*.err' "$REMOTE" "results/"
fi
echo "Done. Analyze locally, e.g.:"
echo "  python scripts/show_results.py results/${SUBDIR:-p3-matrix} --fragility"
