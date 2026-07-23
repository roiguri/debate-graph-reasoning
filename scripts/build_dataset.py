"""Materialize / verify a frozen dataset artifact.

    python scripts/build_dataset.py --verify   # rebuild from the RECORDED spec, assert hash matches (read only)
    python scripts/build_dataset.py            # regenerate data/main.jsonl from the recorded spec (idempotent)
    python scripts/build_dataset.py --init      # FIRST-TIME creation from BOOTSTRAP_SPEC (no meta must exist)

Source of truth is `data/<name>.meta.json`, which records the build `spec`, the
`sha256`, and provenance. Once it exists, both `build` and `--verify` rebuild from
`meta["spec"]`, so the committed artifact (not any constant in this file) defines
the dataset. `build` refuses to write anything that does not match the recorded
hash, so it can never silently replace the dataset; changing it is a deliberate act
(remove the artifact, edit BOOTSTRAP_SPEC, `--init`).

Replication is a **sibling** artifact, not a grown one: build an independent seed
under its own `--name` so the frozen `main` stays byte-identical (its sha256, and
every existing run's manifest guard, are unaffected). instance_ids are namespaced
by dataset_seed, so pooling two seeds in analysis never collides. Everything but
the seed is held fixed, so the samples are matched:

    python scripts/build_dataset.py --init --name seed11 --seed 11   # data/seed11.jsonl @ seed 11
    python scripts/build_dataset.py --verify --name seed11           # check it reproduces
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

DATA_DIR = Path("data")


def _paths(name: str) -> tuple[Path, Path]:
    """The (jsonl, meta) pair for a named dataset artifact under data/."""
    return DATA_DIR / f"{name}.jsonl", DATA_DIR / f"{name}.meta.json"


# Used ONLY by --init to create an artifact the first time. After that the spec
# lives in data/<name>.meta.json and this constant is never read. `--seed`
# overrides `dataset_seed` for a sibling (replication) artifact; N, tasks, and
# encodings stay fixed so seeds are matched.
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


def _read_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def init(name: str, seed: int | None = None) -> None:
    """First-time creation from BOOTSTRAP_SPEC. Refuses to clobber an existing artifact.

    `seed` overrides the spec's `dataset_seed` (for a replication sibling); `name`
    sets which artifact is written. Everything else stays at BOOTSTRAP_SPEC so the
    sibling is matched to `main` on N/tasks/encodings.
    """
    data_path, meta_path = _paths(name)
    if meta_path.exists():
        raise SystemExit(f"{meta_path} already exists; --init is only for first creation")
    spec = dict(BOOTSTRAP_SPEC)
    if seed is not None:
        spec["dataset_seed"] = seed
    instances = _build_from_spec(spec)
    meta = {
        "name": name,
        "spec": spec,
        "n_instances": len(instances),
        "sha256": dataset_sha256(instances),
        "gedebate_version": gedebate.__version__,
        "git_commit": _git_commit(),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    dump_dataset(instances, data_path)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"created {data_path} ({meta['n_instances']} instances, seed={spec['dataset_seed']}) "
          f"sha256={meta['sha256'][:16]}...")


def build(name: str) -> None:
    """Regenerate data/<name>.jsonl from the RECORDED spec. Idempotent and hash-guarded."""
    data_path, meta_path = _paths(name)
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} missing; create the dataset first with --init")
    meta = _read_meta(meta_path)
    instances = _build_from_spec(meta["spec"])
    got = dataset_sha256(instances)
    if got != meta["sha256"]:
        raise SystemExit(
            "rebuild from meta['spec'] does NOT match meta['sha256'] "
            f"({got[:16]}... vs {meta['sha256'][:16]}...); generation drifted. "
            "Refusing to overwrite. Investigate before changing the dataset."
        )
    dump_dataset(instances, data_path)  # identical bytes; safe. Provenance in meta is preserved.
    print(f"regenerated {data_path} from recorded spec; hash unchanged ({got[:16]}...)")


def verify(name: str) -> None:
    """Reproducibility check: rebuild from the recorded spec, assert it matches meta."""
    _, meta_path = _paths(name)
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} missing; nothing to verify")
    meta = _read_meta(meta_path)
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
    ap.add_argument("--name", default="main",
                    help="artifact name under data/ (default: main; e.g. seed11)")
    ap.add_argument("--seed", type=int, default=None,
                    help="with --init only: override dataset_seed for a replication sibling")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--verify", action="store_true", help="rebuild from meta spec and check the hash (read only)")
    g.add_argument("--init", action="store_true", help="first-time creation from BOOTSTRAP_SPEC")
    args = ap.parse_args()
    if args.seed is not None and not args.init:
        ap.error("--seed is only valid with --init (existing artifacts take the seed from their meta)")
    if args.verify:
        verify(args.name)
    elif args.init:
        init(args.name, args.seed)
    else:
        build(args.name)


if __name__ == "__main__":
    main()
