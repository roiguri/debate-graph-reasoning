"""Baseline condition: one greedy, zero-shot answer per instance.

`run_instance` is the atomic unit the whole harness is built around: build the
prompt, generate once (greedy), parse, score, and return one attempt-level record.
In P2.1 that record is printed; P2.2 persists it under the shared schema, and P4/P5
produce the *same* record shape (N samples / a debate) so nothing is reopened.

Deliberately does not import `gedebate.model` at runtime (it pulls torch, a
cluster-only extra) -- `run_instance` only needs a duck-typed object with a
`.generate(prompt, max_new_tokens=...) -> GenResult`, so it stays importable and
testable locally with a stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gedebate.eval.scoring import parse, score
from gedebate.prompts import build_prompt

if TYPE_CHECKING:
    from gedebate.data.instance import Instance
    from gedebate.model import Model

CONDITION = "baseline"


def run_instance(
    model: "Model", instance: "Instance", *, max_new_tokens: int = 64
) -> dict:
    """Prompt -> greedy generate -> parse -> score -> one attempt record."""
    prompt = build_prompt(instance)
    gen = model.generate(prompt, max_new_tokens=max_new_tokens)  # greedy (temp=None)
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
    }
