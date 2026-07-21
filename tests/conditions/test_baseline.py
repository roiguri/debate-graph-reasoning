"""Tests for the baseline pipe -- `conditions.baseline.run_instance` end-to-end.

Integration of the P2.1 slice: a real edge_existence x adjacency instance flows
through prompt -> generate -> parse -> score, with a stub model standing in for the
(torch-only) HF model. Scoring and prompt building are unit-tested separately in
test_scoring.py / test_prompts.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from gedebate.conditions.baseline import run_instance
from gedebate.eval.runner import first_instance


@dataclass
class _StubGen:
    text: str
    n_gen_tokens: int
    n_prompt_tokens: int


class _StubModel:
    """Duck-types Model.generate -> GenResult, returning a canned reply."""

    def __init__(self, reply: str):
        self.reply = reply

    def generate(self, prompt: str, *, max_new_tokens: int = 64, **_):
        return _StubGen(self.reply, n_gen_tokens=7, n_prompt_tokens=11)


def _edge_instance():
    return first_instance(n_graphs=4, seed=7, task="edge_existence", encoding="adjacency")


def test_run_instance_record_shape_and_scoring():
    inst = _edge_instance()
    gold_reply = "Yes." if inst.ground_truth else "No."
    rec = run_instance(_StubModel(gold_reply), inst)

    assert rec["condition"] == "baseline"
    assert rec["task"] == "edge_existence"
    assert rec["encoding"] == "adjacency"
    assert rec["parse_ok"] is True
    assert rec["correct"] is True
    assert rec["ground_truth"] == inst.ground_truth
    assert rec["n_gen_tokens"] == 7 and rec["n_prompt_tokens"] == 11


def test_run_instance_wrong_answer_scores_incorrect():
    inst = _edge_instance()
    wrong_reply = "No." if inst.ground_truth else "Yes."
    rec = run_instance(_StubModel(wrong_reply), inst)
    assert rec["parse_ok"] is True
    assert rec["correct"] is False


def test_run_instance_unparseable_output():
    inst = _edge_instance()
    rec = run_instance(_StubModel("hmm not sure"), inst)
    assert rec["parse_ok"] is False
    assert rec["parsed_answer"] is None
    assert rec["correct"] is False


def test_first_instance_matches_request():
    inst = first_instance(n_graphs=4, seed=7, task="edge_existence", encoding="adjacency")
    assert inst.task == "edge_existence"
    assert inst.encoding == "adjacency"
    assert isinstance(inst.ground_truth, bool)
