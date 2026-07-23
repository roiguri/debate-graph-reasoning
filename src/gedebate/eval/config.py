"""Run configuration, loaded from TOML (stdlib `tomllib`, no new dep).

The config is the **single source of truth for the model id** and the run matrix
(tasks x encodings x N graphs). The full matrix is run by editing these files +
`--shard`; no code changes.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from gedebate.data.dataset import ENCODINGS as ALL_ENCODINGS
from gedebate.data.dataset import TASKS as ALL_TASKS

KNOWN_CONDITIONS = ("baseline", "majority_vote")  # debate added later


@dataclass(frozen=True)
class RunConfig:
    model: str
    out_dir: str
    dataset: str  # path to a frozen dataset artifact (data/main.jsonl)
    condition: str = "baseline"
    # tasks/encodings act as FILTERS over the loaded dataset (default: all of it).
    tasks: tuple[str, ...] = tuple(ALL_TASKS)
    encodings: tuple[str, ...] = tuple(ALL_ENCODINGS)
    max_new_tokens: int = 64
    # majority-vote decoding (ignored by baseline): draws per instance + temperature.
    n_samples: int = 5
    temperature: float = 0.7

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for req in ("model", "out_dir", "dataset"):
            if req not in data:
                raise ValueError(f"config must set '{req}'")

        tasks = tuple(data.get("tasks", ALL_TASKS))
        encodings = tuple(data.get("encodings", ALL_ENCODINGS))
        _reject(set(tasks) - set(ALL_TASKS), "task", ALL_TASKS)
        _reject(set(encodings) - set(ALL_ENCODINGS), "encoding", ALL_ENCODINGS)
        condition = data.get("condition", "baseline")
        if condition not in KNOWN_CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; known: {KNOWN_CONDITIONS}")

        n_samples = int(data.get("n_samples", 5))
        temperature = float(data.get("temperature", 0.7))
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 (sampling), got {temperature}")

        return cls(
            model=data["model"],
            out_dir=data["out_dir"],
            dataset=data["dataset"],
            condition=condition,
            tasks=tasks,
            encodings=encodings,
            max_new_tokens=int(data.get("max_new_tokens", 64)),
            n_samples=n_samples,
            temperature=temperature,
        )


def _reject(bad: set, kind: str, allowed) -> None:
    if bad:
        raise ValueError(f"unknown {kind}(s) {sorted(bad)}; allowed: {tuple(allowed)}")


def load_config(path: str | Path) -> RunConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return RunConfig.from_dict(data)
