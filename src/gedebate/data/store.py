"""Read/write a materialized dataset: one `Instance` per JSONL line.

The dataset is a **frozen artifact** every run loads (never regenerated in the run
path). `dataset_sha256` gives it a stable identity for provenance + the run
manifest guard. Serialization is deterministic, so a rebuild reproduces the same
bytes and hash -- that is the reproducibility check (see scripts/build_dataset.py).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gedebate.data.instance import Instance


def dataset_text(instances: list[Instance]) -> str:
    """Canonical JSONL text for a dataset. Deterministic given the instances."""
    return "".join(json.dumps(inst.to_dict()) + "\n" for inst in instances)


def dataset_sha256(instances: list[Instance]) -> str:
    """Stable content hash of the dataset (identity for provenance / guards)."""
    return hashlib.sha256(dataset_text(instances).encode("utf-8")).hexdigest()


def dump_dataset(instances: list[Instance], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dataset_text(instances), encoding="utf-8")


def load_dataset(path: str | Path) -> list[Instance]:
    """Reconstruct instances from a materialized dataset file."""
    out: list[Instance] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        d = json.loads(s)
        d.pop("instance_id", None)  # a derived property, not a constructor field
        out.append(Instance(**d))
    return out


def meta_path_for(dataset_path: str | Path) -> Path:
    """The provenance sidecar next to a dataset (data/main.jsonl -> data/main.meta.json)."""
    return Path(dataset_path).with_suffix(".meta.json")


def dataset_identity(dataset_path: str | Path) -> str:
    """The dataset's content hash: read from the committed meta (source of truth),
    or computed from the file if no meta sits alongside it."""
    meta = meta_path_for(dataset_path)
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))["sha256"]
    return dataset_sha256(load_dataset(dataset_path))
