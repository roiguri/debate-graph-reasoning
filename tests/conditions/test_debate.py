"""Tests for the debate loop (`conditions.debate.run_debate`) with a scripted stub.

The stub returns canned turn outputs in call order, so we drive each stopping path:
converge, revise-then-agree, no-progress, response budget, and unparseable verdict.
Torch-free (no model). Prompt/parse behavior is covered in tests/prompts/test_debate.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from gedebate.conditions.debate import run_debate
from gedebate.data.dataset import build_dataset


def _inst():
    return next(i for i in build_dataset(n_graphs=4, seed=7)
                if i.task == "node_degree" and i.encoding == "adjacency")


@dataclass
class _Gen:
    text: str
    n_gen_tokens: int
    n_prompt_tokens: int


class _ScriptStub:
    """Returns queued replies in call order; fixed token counts (2 gen / 10 prompt)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, prompt, *, max_new_tokens=256, **_):
        r = self.replies[self.calls]
        self.calls += 1
        return _Gen(r, 2, 10)


def test_converges_immediately_on_agree():
    inst = _inst()
    gt = inst.ground_truth
    stub = _ScriptStub([f"1. some claim\nANSWER: {gt}", "VERDICT: AGREE"])
    rec, turns = run_debate(stub, inst, max_responses=10)
    assert rec["n_responses"] == 2 and rec["correct"] is True and rec["parsed_answer"] == gt
    assert [t["role"] for t in turns] == ["proposer", "critic"]
    assert rec["n_gen_tokens"] == 4 and rec["n_prompt_tokens"] == 20  # 2 turns x (2,10)
    assert rec["stopped_on_unparsed_verdict"] is False  # real AGREE, not a default


def test_revises_then_agrees():
    inst = _inst()
    gt = inst.ground_truth
    stub = _ScriptStub([
        f"ANSWER: {gt + 1}",                       # proposer wrong
        "VERDICT: REVISE\n- claim about (2,3) is wrong",
        f"1. (0,3) is an edge\nANSWER: {gt}",       # revision correct
        "VERDICT: AGREE",
    ])
    rec, turns = run_debate(stub, inst, max_responses=10)
    assert rec["n_responses"] == 4 and rec["correct"] is True and rec["parsed_answer"] == gt
    assert [t["role"] for t in turns] == ["proposer", "critic", "proposer", "critic"]


def test_no_progress_stops_on_repeated_answer():
    inst = _inst()
    wrong = inst.ground_truth + 1
    stub = _ScriptStub([
        f"ANSWER: {wrong}",
        "VERDICT: REVISE\n- something",
        f"ANSWER: {wrong}",   # revision repeats the same answer -> no progress
    ])
    rec, turns = run_debate(stub, inst, max_responses=10)
    assert rec["n_responses"] == 3 and rec["parsed_answer"] == wrong and rec["correct"] is False


def test_response_budget_caps_the_loop():
    inst = _inst()
    stub = _ScriptStub([
        "ANSWER: 7", "VERDICT: REVISE\n- x",
        "ANSWER: 8", "VERDICT: REVISE\n- y",   # len hits 4 -> no room to revise -> stop
    ])
    rec, turns = run_debate(stub, inst, max_responses=4)
    assert rec["n_responses"] == 4
    assert rec["parsed_answer"] == 8  # final = last Proposer answer (turn 3)


def test_unparseable_verdict_counts_and_stops():
    inst = _inst()
    gt = inst.ground_truth
    stub = _ScriptStub([f"ANSWER: {gt}", "this looks fine to me"])  # no VERDICT line
    rec, turns = run_debate(stub, inst, max_responses=10)
    assert rec["n_responses"] == 2 and rec["stopped_on_unparsed_verdict"] is True
    assert rec["correct"] is True  # unparseable -> AGREE, final is the correct Proposer answer
