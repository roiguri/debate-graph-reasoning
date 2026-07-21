"""Materialize / verify the frozen dataset artifact.

    python scripts/build_dataset.py --verify   # rebuild from the RECORDED spec, assert hash matches (read only)
    python scripts/build_dataset.py            # regenerate data/main.jsonl from the recorded spec (idempotent)
    python scripts/build_dataset.py --init      # FIRST-TIME creation from BOOTSTRAP_SPEC (no meta must exist)

Source of truth is `data/main.meta.json`, which records the build `spec`, the
`sha256`, and provenance. Once it exists, both `build` and `--verify` rebuild from
`meta["spec"]`, so the committed artifact (not any constant in this file) defines
the dataset. `build` refuses to write anything that does not match the recorded
hash, so it can never silently replace the dataset; changing it is a deliberate act
(remove the artifact, edit BOOTSTRAP_SPEC, `--init`). Growing it means appending a
new seed's instances; existing instance_ids never move.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import gedebate
from gedebate.data.dataset import build_dataset
from gedebate.data.store import dataset_sha256, dump_dataset

DATA_PATH = Path("data/main.jsonl")
META_PATH = Path("data/main.meta.json")

# Used ONLY by --init to create the artifact the first time. After that the spec
# lives in data/main.meta.json and this constant is never read.
BOOTSTRAP_SPEC = {
    "algorithm": "er",
    "dataset_seed": 7,
    "n_graphs": 200,
    "tasks": ["edge_existence", "node_degree", "connected_nodes"],
    "encodings": ["adjacency", "incident", "friendship"],
}


def _build_from_spec(spec: dict):
    return build_dataset(
        n_graphs=spec["n_graphs"],
        seed=spec["dataset_seed"],
        algorithm=spec["algorithm"],
        tasks=tuple(spec["tasks"]),
        encodings=tuple(spec["encodings"]),
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _read_meta() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def init() -> None:
    """First-time creation from BOOTSTRAP_SPEC. Refuses to clobber an existing artifact."""
    if META_PATH.exists():
        raise SystemExit(f"{META_PATH} already exists; --init is only for first creation")
    instances = _build_from_spec(BOOTSTRAP_SPEC)
    meta = {
        "name": "main",
        "spec": BOOTSTRAP_SPEC,
        "n_instances": len(instances),
        "sha256": dataset_sha256(instances),
        "gedebate_version": gedebate.__version__,
        "git_commit": _git_commit(),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    dump_dataset(instances, DATA_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"created {DATA_PATH} ({meta['n_instances']} instances) sha256={meta['sha256'][:16]}...")


def build() -> None:
    """Regenerate data/main.jsonl from the RECORDED spec. Idempotent and hash-guarded."""
    if not META_PATH.exists():
        raise SystemExit(f"{META_PATH} missing; create the dataset first with --init")
    meta = _read_meta()
    instances = _build_from_spec(meta["spec"])
    got = dataset_sha256(instances)
    if got != meta["sha256"]:
        raise SystemExit(
            "rebuild from meta['spec'] does NOT match meta['sha256'] "
            f"({got[:16]}... vs {meta['sha256'][:16]}...); generation drifted. "
            "Refusing to overwrite. Investigate before changing the dataset."
        )
    dump_dataset(instances, DATA_PATH)  # identical bytes; safe. Provenance in meta is preserved.
    print(f"regenerated {DATA_PATH} from recorded spec; hash unchanged ({got[:16]}...)")


def verify() -> None:
    """Reproducibility check: rebuild from the recorded spec, assert it matches meta."""
    if not META_PATH.exists():
        raise SystemExit(f"{META_PATH} missing; nothing to verify")
    meta = _read_meta()
    instances = _build_from_spec(meta["spec"])
    got = dataset_sha256(instances)
    n = len(instances)
    print(f"spec (from meta): {meta['spec']}")
    print(f"rebuilt {n} instances; sha256={got[:16]}...")
    print(f"recorded {meta['n_instances']} instances; sha256={meta['sha256'][:16]}...")
    if got != meta["sha256"] or n != meta["n_instances"]:
        raise SystemExit("VERIFY FAILED: rebuild does not match the committed dataset")
    print("VERIFY OK: dataset reproduces exactly from the recorded spec.")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--verify", action="store_true", help="rebuild from meta spec and check the hash (read only)")
    g.add_argument("--init", action="store_true", help="first-time creation from BOOTSTRAP_SPEC")
    args = ap.parse_args()
    if args.verify:
        verify()
    elif args.init:
        init()
    else:
        build()


if __name__ == "__main__":
    main()
