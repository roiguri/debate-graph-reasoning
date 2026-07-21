"""Unit tests for `gedebate.prompts.build_prompt`.

Independent of the data layer: `build_prompt` reads only `.task` and `.question`,
so a lightweight stand-in stands in for an Instance. Verifies the terse instruction
is prefixed and GraphQA's question block is kept verbatim (P2.1 = edge_existence).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gedebate.prompts import build_prompt

_QUESTION = (
    "In an undirected graph, (i,j) means that node i and node j are connected "
    "with an undirected edge. G describes a graph among nodes 0, 1, and 2.\n"
    "The edges in G are: (0, 1) (1, 2).\n"
    "Q: Is node 0 connected to node 2?\nA: "
)


def test_prefixes_instruction_and_keeps_question_verbatim():
    inst = SimpleNamespace(task="edge_existence", question=_QUESTION)
    prompt = build_prompt(inst)
    assert prompt.startswith('Answer with exactly "Yes" or "No" and nothing else.')
    assert prompt.endswith(_QUESTION)  # GraphQA block unchanged, still ends in "A: "


def test_unimplemented_task_raises():
    inst = SimpleNamespace(task="node_degree", question=_QUESTION)
    with pytest.raises(NotImplementedError):
        build_prompt(inst)
