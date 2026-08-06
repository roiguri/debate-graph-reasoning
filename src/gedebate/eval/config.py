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
from gedebate.prompts.debate import DEFAULT_PROMPT_VERSION, PROMPT_VERSIONS

KNOWN_CONDITIONS = ("baseline", "majority_vote", "debate")
# Where `model` is served from. "hf" loads it in-process on a GPU; "together" posts to
# the Together.ai API. The two are NOT interchangeable within an analysis: token counts
# come from different tokenizers and sampled draws only replay under "hf".
KNOWN_PROVIDERS = ("hf", "together")


@dataclass(frozen=True)
class RunConfig:
    model: str
    out_dir: str
    dataset: str  # path to a frozen dataset artifact (data/main.jsonl)
    condition: str = "baseline"
    # Serving backend for `model`. Defaults to "hf" so every existing config keeps
    # meaning exactly what it meant. Recorded in the manifest: rows from different
    # providers carry token counts from different tokenizers and must not be pooled.
    provider: str = "hf"
    # tasks/encodings act as FILTERS over the loaded dataset (default: all of it).
    tasks: tuple[str, ...] = tuple(ALL_TASKS)
    encodings: tuple[str, ...] = tuple(ALL_ENCODINGS)
    max_new_tokens: int = 64
    # majority-vote decoding (ignored by baseline): draws per instance + sampling.
    # top_p/top_k are set explicitly (not inherited from the model's generation_config)
    # so the sampling truncation is a deliberate, in-manifest choice. Defaults are
    # Qwen2.5-Instruct's recommended values; top_p=1.0 + top_k=0 disables truncation.
    n_samples: int = 10  # the locked P4 budget; a config omitting it runs the full budget
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    # Debate Proposer/revision wording (ignored by baseline/majority-vote). Versioned so
    # a prompt fix cannot retroactively change what an existing config would emit; the
    # default is v1, which is what results/main was run under. See prompts/debate.py.
    prompt_version: str = DEFAULT_PROMPT_VERSION

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
        provider = data.get("provider", "hf")
        if provider not in KNOWN_PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; known: {KNOWN_PROVIDERS}")

        n_samples = int(data.get("n_samples", 10))
        temperature = float(data.get("temperature", 0.7))
        top_p = float(data.get("top_p", 0.8))
        top_k = int(data.get("top_k", 20))
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 (sampling), got {temperature}")
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 disables), got {top_k}")
        prompt_version = data.get("prompt_version", DEFAULT_PROMPT_VERSION)
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(
                f"unknown prompt_version {prompt_version!r}; "
                f"known: {tuple(PROMPT_VERSIONS)}"
            )

        return cls(
            model=data["model"],
            out_dir=data["out_dir"],
            dataset=data["dataset"],
            condition=condition,
            provider=provider,
            tasks=tasks,
            encodings=encodings,
            max_new_tokens=int(data.get("max_new_tokens", 64)),
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prompt_version=prompt_version,
        )


def _reject(bad: set, kind: str, allowed) -> None:
    if bad:
        raise ValueError(f"unknown {kind}(s) {sorted(bad)}; allowed: {tuple(allowed)}")


def load_config(path: str | Path) -> RunConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return RunConfig.from_dict(data)
