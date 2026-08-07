"""Persistence contract: attempt-level JSONL rows, resume, and a run manifest.

The keystone the whole harness (all conditions) inherits. Design and rationale are in
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
- **Manifest guard**: a per-run `manifest.json` doubles as a reproduction record and
  pins the model + dataset hash; resuming against a different model or dataset is an
  error (prevents mixing two models or datasets into one accuracy).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gedebate.data.instance import Instance

SCHEMA_VERSION = 1

# Conditions that write N rows per instance (one per `sample_index`) rather than one.
# Named here, in the layer that decides when an instance is "done", so adding a vote arm
# never leaves a resume check silently treating N rows as N finished instances.
VOTE_CONDITIONS = ("majority_vote", "majority_vote_cot")

# Conditions whose `raw_output` is a numbered-claim trace ending in an `ANSWER:` line,
# i.e. that prompted with the debate Proposer wording. They must be read by
# `prompts.debate.parse_proposer`, never by the shared `scoring.parse`, which scans the
# whole text and would harvest labels out of the reasoning ("not connected to 3" -> 3).
# Named here so anything re-deriving an answer from stored text picks the parser by fact
# rather than by guessing from the condition name.
PROPOSER_FORMAT_CONDITIONS = ("debate", "majority_vote_cot")

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
    "n_responses",   # model calls this row represents: 1 for baseline/majority-vote
                     # (per sample), = # turns for debate. Sums to the compute metric.
    "prompt_version",  # debate only (None elsewhere): which Proposer wording produced
                     # this row. Persisted PER ROW, not just in the manifest, so a v1 and
                     # a v2 debate row can never be pooled into one accuracy by accident
                     # -- rows from different run dirs are routinely read together.
                     # Rows written before this field existed are all v1 (see
                     # DEFAULT_ROW_PROMPT_VERSION).
)

# What a row without a `prompt_version` means: the single pre-versioning wording, v1.
#
# No such rows remain in the tree -- the v1 debate rows were deleted when the project
# consolidated on the frozen v2 prompt, and every debate row on disk now carries
# `prompt_version: "v2"`. This stays as a guard rather than being deleted with them: an
# unlabelled debate row can only have come from before the field existed, and defaulting
# it to the surviving version would silently pool a v1 answer into a v2 accuracy. It must
# never be "helpfully" changed to v2.
DEFAULT_ROW_PROMPT_VERSION = "v1"


def row_prompt_version(row: dict) -> str:
    """The Proposer wording a row was produced under, defaulting old rows to v1."""
    return row.get("prompt_version") or DEFAULT_ROW_PROMPT_VERSION


def make_row(
    instance: "Instance",
    model: str,
    attempt: dict,
    *,
    sample_index: int = 0,
    temperature: float | None = None,
    seed: int | None = None,
    n_responses: int = 1,
    prompt_version: str | None = None,
) -> dict:
    """Assemble one persisted row from an instance + a condition's attempt record.

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
        "n_responses": n_responses,
        "prompt_version": prompt_version,
    }


# --- file layout --------------------------------------------------------------

def shard_file(run_dir: str | Path, condition: str, shard: int = 0) -> Path:
    """Path a shard writes: <run_dir>/<condition>/shard<i>.jsonl.

    A run dir is one *dataset* (model + seed + N, pinned by its manifest); the
    conditions (baseline / majority_vote / debate) live in sibling subfolders so
    they share the dataset and can be joined for the matched-compute comparison.
    One writer per file → no append contention.
    """
    return Path(run_dir) / condition / f"shard{shard:03d}.jsonl"


def result_files(run_dir: str | Path) -> list[Path]:
    """All result JSONL under a run dir, across every condition subfolder.

    Excludes `*.trace.jsonl` sidecars (verbose debate transcripts, read separately)."""
    return sorted(p for p in Path(run_dir).glob("**/*.jsonl")
                  if not p.name.endswith(".trace.jsonl"))


# --- debate trace sidecar (verbose transcripts, kept out of the lean rows) -----

def trace_file(run_dir: str | Path, condition: str, shard: int = 0) -> Path:
    """Path a shard's traces write: <run_dir>/<condition>/shard<i>.trace.jsonl.

    One line per instance: {instance_id, turns}. Debate stores its full Proposer-Critic
    transcript here; the result row stays lean (final answer + summed compute)."""
    return Path(run_dir) / condition / f"shard{shard:03d}.trace.jsonl"


def trace_files(run_dir: str | Path) -> list[Path]:
    """All trace sidecars under a run dir (the complement of `result_files`)."""
    return sorted(Path(run_dir).glob("**/*.trace.jsonl"))


def append_trace(path: str | Path, instance_id: str, turns: list[dict]) -> None:
    """Append one instance's transcript (flush+fsync), mirroring append_row's kill safety."""
    append_row(path, {"instance_id": instance_id, "turns": turns})


def read_traces(path: str | Path) -> list[dict]:
    """Parse a trace sidecar into [{instance_id, turns}, ...] (torn-trailing tolerant)."""
    return read_rows(path)


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
    """Attempts that make an instance 'done': N for either vote arm, else 1."""
    return n_samples if condition in VOTE_CONDITIONS else 1


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


# The manifest is versioned separately from the row schema (SCHEMA_VERSION): v2
# records per-condition provenance under `conditions`, where v1 was flat.
MANIFEST_VERSION = 2

# Top-level invariants a run dir shares across every condition. `dataset` +
# `dataset_sha256` identify the exact frozen instance set; `model` is passed
# explicitly. A resume must not change the guarded ones, else two models or datasets
# would mix into one accuracy.
_SHARED_KEYS = ("dataset", "dataset_sha256")
_GUARDED_KEYS = ("model", "dataset_sha256")


def ensure_manifest(run_dir: str | Path, model: str, condition: str, **fields) -> dict:
    """Create or update the run manifest, recording per-condition provenance.

    A run dir holds one dataset + model (the shared, guarded invariants at top level)
    and one or more conditions (baseline / majority_vote / debate) as siblings under
    a shared `out_dir`. Each condition's provenance (decoding, config, tokens, ...) is
    stored under `conditions[condition]` and written **once**: a later condition adds
    its own entry without clobbering the others, and a resume of the same condition
    preserves its original record. Guards `model` + `dataset_sha256` (whichever are
    supplied) so a resume can never mix two models -- or two datasets -- into one
    accuracy. Shared fields from earlier calls are preserved if not re-supplied.
    """
    shared = {k: fields[k] for k in _SHARED_KEYS if k in fields}
    provenance = {k: v for k, v in fields.items() if k not in _SHARED_KEYS}
    guard_vals = {"model": model, "dataset_sha256": shared.get("dataset_sha256")}

    existing = read_manifest(run_dir)
    conditions: dict = {}
    if existing is not None:
        for k in _GUARDED_KEYS:
            want = guard_vals.get(k)
            if want is not None and existing.get(k) != want:
                raise ValueError(
                    f"run dir {run_dir!s} was built with {k}={existing.get(k)!r}, "
                    f"not {want!r} -- use a fresh out_dir"
                )
        conditions = dict(existing.get("conditions", {}))
        # carry shared fields set by earlier calls that this one did not re-supply
        shared = {**{k: existing[k] for k in _SHARED_KEYS if k in existing}, **shared}

    if condition not in conditions:
        conditions[condition] = provenance

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "model": model,
        **shared,
        "conditions": conditions,
    }
    p = manifest_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
