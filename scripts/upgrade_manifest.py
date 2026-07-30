"""One-off: upgrade a run dir's manifest.json from v1 (flat) to v2 (per-condition).

v1 stored only the last-writing condition's provenance at the top level, so a
multi-condition dir (baseline + majority_vote sharing one out_dir) mislabeled the
others -- e.g. results/main/manifest.json ended up describing the MV run and lost
baseline's greedy provenance. This rewrites <run_dir>/manifest.json into v2, with one
entry per condition subdir under `conditions`.

Safety: this ONLY writes manifest.json. Result rows (*/*.jsonl) are never touched.
Idempotent: a manifest already at v2 is left as-is. Fields we cannot recover for a
condition that did not write the flat manifest are reconstructed best-effort from
known configs + the shard files' mtime, and marked "reconstructed": true.

    python scripts/upgrade_manifest.py results/main
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from gedebate.eval import results

# What we can reconstruct for a condition that is NOT the flat manifest's own writer.
_KNOWN = {
    "baseline": {"decoding": "greedy", "config": "configs/matrix.toml"},
}
# Per-condition provenance keys (everything that is not a shared/top-level invariant).
_PROV_KEYS = ("decoding", "config", "max_new_tokens", "n_samples",
              "gedebate_version", "git_commit", "created")


def _mtime_iso(paths: list[Path]) -> str | None:
    if not paths:
        return None
    ts = min(os.path.getmtime(p) for p in paths)  # earliest write = first shard
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _flat_condition(flat: dict) -> str:
    """Which condition the flat (v1) manifest's top-level provenance describes."""
    stem = Path(str(flat.get("config", ""))).stem
    return "majority_vote" if "mv" in stem else "baseline"


def upgrade(run_dir: str) -> None:
    run = Path(run_dir)
    flat = results.read_manifest(run)
    if flat is None:
        raise SystemExit(f"no manifest.json in {run}")
    if flat.get("manifest_version") == results.MANIFEST_VERSION or "conditions" in flat:
        print(f"{run}/manifest.json is already v{results.MANIFEST_VERSION}; nothing to do.")
        return

    own = _flat_condition(flat)
    own_prov = {k: flat[k] for k in _PROV_KEYS if k in flat}

    subdirs = sorted(d.name for d in run.iterdir() if d.is_dir() and any(d.glob("*.jsonl")))
    conditions: dict = {}
    for cond in subdirs:
        if cond == own:
            conditions[cond] = own_prov
        else:
            recon = dict(_KNOWN.get(cond, {}))
            recon["max_new_tokens"] = flat.get("max_new_tokens")
            recon["gedebate_version"] = flat.get("gedebate_version")
            recon["git_commit"] = "unknown"
            recon["created"] = _mtime_iso(list(Path(run, cond).glob("*.jsonl")))
            recon["reconstructed"] = True
            conditions[cond] = recon

    v2 = {"manifest_version": results.MANIFEST_VERSION, "model": flat["model"]}
    for k in ("dataset", "dataset_sha256"):
        if flat.get(k) is not None:
            v2[k] = flat[k]
    v2["conditions"] = conditions

    print("=== BEFORE (v1, flat) ===")
    print(json.dumps(flat, indent=2))
    print("\n=== AFTER (v2, per-condition) ===")
    print(json.dumps(v2, indent=2))
    results.manifest_path(run).write_text(json.dumps(v2, indent=2), encoding="utf-8")
    print(f"\nwrote {results.manifest_path(run)}  (result rows untouched)")


if __name__ == "__main__":
    upgrade(sys.argv[1] if len(sys.argv) > 1 else "results/main")
