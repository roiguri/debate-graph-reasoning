#!/bin/bash
# Clear accumulated SLURM job logs + smoke artifacts from results/, KEEPING
# every run data dir and manifest.json. Run on the cluster from the project root:
#   bash slurm/clean_logs.sh          # list what would be removed (dry run)
#   bash slurm/clean_logs.sh --force  # actually remove
#
# Only touches loose log/smoke files at the top of results/ -- never the per-run
# subdirectories (results/p2-baseline/, results/p3-matrix/, ...) or their manifests.
set -eo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/logs   # where future array jobs write (see slurm/p3-matrix.slurm)

# Loose top-level clutter only: job logs and the smoke outputs.
mapfile -t victims < <(find results -maxdepth 1 -type f \
  \( -name '*.out' -o -name '*.err' -o -name 'smoke.json' \) | sort)

if [ "${#victims[@]}" -eq 0 ]; then
  echo "Nothing to clean (results/ top level is already tidy)."
  exit 0
fi

echo "Would remove ${#victims[@]} file(s):"
printf '  %s\n' "${victims[@]}"

if [ "${1:-}" != "--force" ]; then
  echo
  echo "Dry run. Re-run with --force to delete. (Run data dirs + manifests are never touched.)"
  exit 0
fi

rm -f -- "${victims[@]}"
echo "Removed ${#victims[@]} file(s). Kept all run data dirs + manifests."
