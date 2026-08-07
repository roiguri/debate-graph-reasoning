"""Tests for the debate loop (`conditions.debate.run_debate`) with a scripted stub.

The stub returns canned turn outputs in call order, so we drive each stopping path:
converge, revise-then-agree, no-progress, response budget, and unparseable verdict.
Torch-free (no model). Prompt/parse behavior is covered in tests/prompts/test_debate.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from gedebate.conditions.debate import run_debate
from gedebate.prompts.debate import critic_prompt, proposer_prompt, revision_prompt
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
    rec, turns = run_debate(stub, inst, prompt_version="v2", max_responses=10)
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
    rec, turns = run_debate(stub, inst, prompt_version="v2", max_responses=10)
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
    rec, turns = run_debate(stub, inst, prompt_version="v2", max_responses=10)
    assert rec["n_responses"] == 3 and rec["parsed_answer"] == wrong and rec["correct"] is False


def test_response_budget_caps_the_loop():
    inst = _inst()
    stub = _ScriptStub([
        "ANSWER: 7", "VERDICT: REVISE\n- x",
        "ANSWER: 8", "VERDICT: REVISE\n- y",   # len hits 4 -> no room to revise -> stop
    ])
    rec, turns = run_debate(stub, inst, prompt_version="v2", max_responses=4)
    assert rec["n_responses"] == 4
    assert rec["parsed_answer"] == 8  # final = last Proposer answer (turn 3)


def test_unparseable_verdict_counts_and_stops():
    inst = _inst()
    gt = inst.ground_truth
    stub = _ScriptStub([f"ANSWER: {gt}", "this looks fine to me"])  # no VERDICT line
    rec, turns = run_debate(stub, inst, prompt_version="v2", max_responses=10)
    assert rec["n_responses"] == 2 and rec["stopped_on_unparsed_verdict"] is True
    assert rec["correct"] is True  # unparseable -> AGREE, final is the correct Proposer answer


def test_every_role_uses_the_configured_prompt_version():
    """All three prompts must come from the SAME version.

    The Critic call used to omit `prompt_version` and fall back to the module default, so
    a v3 run sent v3 Proposer + revision prompts and a v2 Critic prompt -- a hybrid no
    manifest could describe, and invisible while only one version existed.
    """
    inst = _inst()

    class _Recorder:
        def __init__(self, replies):
            self.replies, self.calls, self.prompts = replies, 0, []

        def generate(self, prompt, *, max_new_tokens=256, **_):
            self.prompts.append(prompt)
            r = self.replies[self.calls]
            self.calls += 1
            return _Gen(r, 2, 10)

    # proposer -> critic REVISE -> revision -> critic AGREE: exercises all three builders
    model = _Recorder(["1. a\nANSWER: 1", "VERDICT: REVISE\n- edge (0, 1)",
                       "1. b\nANSWER: 2", "VERDICT: AGREE"])
    _record, turns = run_debate(model, inst, prompt_version="v3", max_responses=6)

    assert model.prompts[0] == proposer_prompt(inst, "v3")
    assert model.prompts[1] == critic_prompt(inst, turns[:1], "v3")
    assert model.prompts[2] == revision_prompt(inst, turns[:2], "v3")
    assert model.prompts[3] == critic_prompt(inst, turns[:3], "v3")
    # and none of them leaked the other version's wording
    for p in model.prompts:
        assert p != critic_prompt(inst, turns[:1], "v2")
