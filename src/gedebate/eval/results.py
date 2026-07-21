"""Persistence contract: attempt-level JSONL rows, resume, and a run manifest.

The keystone the whole harness (and P4/P5) inherit. Design and rationale are in
docs/notes.md -> "Persistence contract". In short:

- **One row per completed attempt**, written atomically at attempt end. Baseline
  = 1 row/instance; majority-vote = N rows (one per `sample_index`); debate = 1 row
  with summed tokens. A killed attempt leaves no row, so re-running is clean and
  never double-counts tokens.
- **One uniform, lean schema for every condition** (`make_row`), so baseline's row
  *is* the final schema. Verbose debate traces live in a sidecar, not here.
- **Resume by counting**: an instance is done under condition C when it has
  >= `expected_attempts(C)` rows. Done-ids are read from the *union* of all
  `*.jsonl` in the run dir, so changing shard count never redoes work.
- **Kill tolerance**: append + flush + fsync per row; the reader drops an
  unparseable *trailing* line (a torn write) but raises on mid-file corruption.
- **Manifest guard**: a per-run `manifest.json` pins the model; resuming with a
  different model is an error (prevents mixing two models into one accuracy).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gedebate.data.instance import Instance

SCHEMA_VERSION = 1

# Fields persisted per attempt, in order. Identical across all three conditions.
ROW_FIELDS = (
    "schema_version",
    "instance_id",
    "task",
    "encoding",
    "condition",
    "model",
    "sample_index",  # 0 for baseline/debate; 0..N-1 for majority-vote
    "temperature",   # None => greedy
    "seed",          # generation seed (None for greedy baseline)
    "raw_output",
    "parsed_answer",
    "parse_ok",
    "correct",
    "ground_truth",
    "n_prompt_tokens",
    "n_gen_tokens",
)


def make_row(
    instance: "Instance",
    model: str,
    attempt: dict,
    *,
    sample_index: int = 0,
    temperature: float | None = None,
    seed: int | None = None,
) -> dict:
    """Assemble one persisted row from an instance + a `run_instance` attempt.

    The prompt is intentionally *not* stored -- it is reproducible from the
    instance via `build_prompt`, and keeping rows lean matters at matrix scale.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance.instance_id,
        "task": instance.task,
        "encoding": instance.encoding,
        "condition": attempt["condition"],
        "model": model,
        "sample_index": sample_index,
        "temperature": temperature,
        "seed": seed,
        "raw_output": attempt["raw_output"],
        "parsed_answer": attempt["parsed_answer"],
        "parse_ok": attempt["parse_ok"],
        "correct": attempt["correct"],
        "ground_truth": attempt["ground_truth"],
        "n_prompt_tokens": attempt["n_prompt_tokens"],
        "n_gen_tokens": attempt["n_gen_tokens"],
    }


# --- file layout --------------------------------------------------------------

def shard_file(run_dir: str | Path, condition: str, shard: int = 0) -> Path:
    """Path of the JSONL a given shard writes. One writer per file (no contention)."""
    return Path(run_dir) / f"{condition}-shard{shard:03d}.jsonl"


def result_files(run_dir: str | Path) -> list[Path]:
    """All result JSONL files in a run dir (any condition, any shard)."""
    return sorted(Path(run_dir).glob("*.jsonl"))


# --- writing ------------------------------------------------------------------

def append_row(path: str | Path, row: dict) -> None:
    """Append one row and flush+fsync, so a kill loses at most the in-flight row."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row) + "\n"
    with open(p, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


# --- reading (torn-trailing-line tolerant) ------------------------------------

def read_rows(path: str | Path) -> list[dict]:
    """Parse a JSONL file, tolerating a torn *trailing* line from a killed job.

    A non-terminal unparseable line signals real corruption and raises -- one
    writer per file means torn lines only ever occur at the end.
    """
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # torn final write -- tolerate
            raise ValueError(f"corrupt (non-terminal) JSON at {p}:{i + 1}")
    return rows


# --- resume -------------------------------------------------------------------

def expected_attempts(condition: str, n_samples: int = 1) -> int:
    """Attempts that make an instance 'done': N for majority-vote, else 1."""
    return n_samples if condition == "majority_vote" else 1


def load_progress(run_dir: str | Path) -> dict[tuple[str, str], set[int]]:
    """Map (condition, instance_id) -> set of sample_indexes already persisted.

    Read from the union of all result files under the run, so re-sharding between
    runs never redoes completed work. The set naturally dedupes accidental repeats.
    """
    progress: dict[tuple[str, str], set[int]] = {}
    for f in result_files(run_dir):
        for row in read_rows(f):
            key = (row["condition"], row["instance_id"])
            progress.setdefault(key, set()).add(row.get("sample_index", 0))
    return progress


def is_instance_done(
    progress: dict[tuple[str, str], set[int]],
    condition: str,
    instance_id: str,
    *,
    n_samples: int = 1,
) -> bool:
    seen = progress.get((condition, instance_id), ())
    return len(seen) >= expected_attempts(condition, n_samples)


def missing_samples(
    progress: dict[tuple[str, str], set[int]],
    condition: str,
    instance_id: str,
    n_samples: int,
) -> list[int]:
    """Sample indexes still to run for majority-vote resume (top-up after a kill)."""
    seen = progress.get((condition, instance_id), set())
    return [i for i in range(n_samples) if i not in seen]


# --- manifest -----------------------------------------------------------------

def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "manifest.json"


def read_manifest(run_dir: str | Path) -> dict | None:
    p = manifest_path(run_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ensure_manifest(run_dir: str | Path, model: str, **extra) -> dict:
    """Create the run manifest once, or verify the model matches on resume.

    Guards against silently mixing two models' rows into one accuracy -- resuming a
    run dir with a different model raises. `extra` snapshots config/commit/host/gpu.
    """
    existing = read_manifest(run_dir)
    if existing is not None:
        if existing.get("model") != model:
            raise ValueError(
                f"run dir {run_dir!s} was built with model {existing.get('model')!r}, "
                f"not {model!r} -- use a fresh out_dir"
            )
        return existing
    manifest = {"schema_version": SCHEMA_VERSION, "model": model, **extra}
    p = manifest_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
