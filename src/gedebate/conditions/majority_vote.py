"""Majority-vote condition: N sampled answers per instance, then vote.

Self-consistency. Where baseline generates once greedily, majority-vote draws N
samples at temperature > 0 and takes the mode of the *parsed* answers. Each sample
is one attempt record with the **same shape** baseline produces, so the persistence
schema (`results.make_row`) is unchanged: majority-vote is simply N rows per
instance (one per `sample_index`). The vote itself is derived at report time from
those rows, never stored, so the vote rule stays re-derivable.

Torch-free like `baseline`: `run_sample` only needs a duck-typed model with
`.generate(prompt, max_new_tokens=, temperature=, seed=) -> GenResult`, so it stays
importable and stub-testable off the cluster.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from gedebate.eval.scoring import parse, score
from gedebate.prompts import build_prompt

if TYPE_CHECKING:
    from gedebate.data.instance import GroundTruth, Instance
    from gedebate.model import Model

CONDITION = "majority_vote"


def sample_seed(instance_id: str, sample_index: int) -> int:
    """Deterministic per-draw seed from (instance_id, sample_index).

    A stable hash (not Python's salted `hash`) so a killed shard tops up exactly the
    missing samples and a rerun reproduces the same draws. Truncated to 32 bits,
    which is a safe range for `torch.manual_seed`.
    """
    h = hashlib.sha256(f"{instance_id}:{sample_index}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def run_sample(
    model: "Model",
    instance: "Instance",
    *,
    sample_index: int,
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
    max_new_tokens: int = 128,
) -> dict:
    """One sampled draw -> parse -> score -> attempt record (baseline's shape + seed).

    The record carries the `seed` and `sample_index` used so the runner can persist
    them via `results.make_row` without recomputing. `top_p`/`top_k` are passed
    through to the model so the sampling truncation is explicit, not the model's
    shipped default.
    """
    prompt = build_prompt(instance)
    seed = sample_seed(instance.instance_id, sample_index)
    gen = model.generate(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, top_k=top_k, seed=seed,
    )
    parsed, parse_ok = parse(
        instance.task, gen.text, encoding=instance.encoding, node_ids=instance.node_ids
    )
    correct = score(parsed, instance.ground_truth)
    return {
        "condition": CONDITION,
        "task": instance.task,
        "encoding": instance.encoding,
        "prompt": prompt,
        "raw_output": gen.text,
        "parsed_answer": parsed,
        "parse_ok": parse_ok,
        "correct": correct,
        "ground_truth": instance.ground_truth,
        "n_prompt_tokens": gen.n_prompt_tokens,
        "n_gen_tokens": gen.n_gen_tokens,
        "sample_index": sample_index,
        "seed": seed,
    }


def vote(parsed_answers: list["GroundTruth | None"]) -> tuple["GroundTruth | None", bool, int]:
    """Mode over the parsed answers -> (voted_answer, parse_ok, support).

    Parse-failures (`None`) are excluded from the tally (self-consistency
    convention). Ties break toward the answer whose first supporting sample has the
    lowest index (stable, reproducible). Returns `(None, False, 0)` when every sample
    failed to parse. `parsed_answers` is expected in `sample_index` order; `support`
    is the winning answer's vote count.
    """
    # key -> [count, first_index, representative]; list answers keyed by tuple so
    # connected_nodes (unhashable list) still tallies.
    tally: dict = {}
    for idx, ans in enumerate(parsed_answers):
        if ans is None:
            continue
        key = tuple(ans) if isinstance(ans, list) else ans
        if key not in tally:
            tally[key] = [0, idx, ans]
        tally[key][0] += 1

    if not tally:
        return None, False, 0

    count, _first_idx, rep = min(
        tally.values(), key=lambda v: (-v[0], v[1])
    )
    return rep, True, count
