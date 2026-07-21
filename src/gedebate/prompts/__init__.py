"""Prompt templates: a terse-format instruction + GraphQA's question verbatim.

Faithful to Fatemi 2024's zero-shot setting -- the question block (encoded graph +
"Q: ...\\nA: ") is unchanged. We only *prefix* a minimal instruction that nudges an
instruct-*chat* model to emit the same short answer a raw-completion model produced
after "A: ". Templates live here (data, not logic) per the workplan. See
docs/notes.md -> "Baseline prompt" decision.

P2.1 ships the edge_existence template; node_degree / connected_nodes are added in
P2.4 (that's all it takes to widen to the full matrix).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing the data layer at runtime just for a type hint
    from gedebate.data.instance import Instance

# One terse-format instruction per task, mirroring GraphQA's gold answer shape.
TASK_INSTRUCTION = {
    "edge_existence": 'Answer with exactly "Yes" or "No" and nothing else.',
    "node_degree": "Answer with a single integer (the degree) and nothing else.",
    "connected_nodes": (
        "Answer with the connected nodes as a comma-separated list "
        '(or "none" if there are none) and nothing else.'
    ),
}


def build_prompt(instance: "Instance") -> str:
    """Instruction, then GraphQA's question block verbatim (which ends in 'A: ')."""
    try:
        instruction = TASK_INSTRUCTION[instance.task]
    except KeyError:
        raise NotImplementedError(
            f"no prompt template for task '{instance.task}' yet (added in P2.4)"
        )
    return f"{instruction}\n\n{instance.question}"
