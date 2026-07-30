"""Tests for debate prompts + their co-located parsers (torch-free, no model).

Prompt builders assemble the approved wording + the running transcript; the parsers
extract the ANSWER value / claims / VERDICT. Prompt format and parser change together,
so these tests pin the contract between them.
"""

from __future__ import annotations

import pytest

from gedebate.data.dataset import build_dataset
from gedebate.prompts import debate


def _node_degree_instance():
    return next(i for i in build_dataset(n_graphs=4, seed=7)
                if i.task == "node_degree" and i.encoding == "adjacency")


# --- prompt builders ----------------------------------------------------------

def test_proposer_prompt_wraps_question_with_instruction():
    inst = _node_degree_instance()
    p = debate.proposer_prompt(inst)
    assert p.endswith(inst.question)          # question verbatim at the end
    assert "numbered list of\natomic claims" in p
    assert "ANSWER: <a single integer, the degree>" in p


def test_unsupported_task_raises():
    inst = next(i for i in build_dataset(n_graphs=4, seed=7)
                if i.task == "edge_existence")
    with pytest.raises(NotImplementedError):
        debate.proposer_prompt(inst)


def test_render_transcript_accumulates_turns():
    inst = _node_degree_instance()
    assert debate.render_transcript(inst, []) == inst.question
    turns = [
        {"role": "proposer", "raw": "1. (0,3) is an edge.\nANSWER: 2"},
        {"role": "critic", "raw": "VERDICT: REVISE\n- Claim about (2,3) is wrong."},
        {"role": "proposer", "raw": "1. (0,3) is an edge.\nANSWER: 1"},
    ]
    t = debate.render_transcript(inst, turns)
    assert t.startswith(inst.question + "1. (0,3) is an edge.\nANSWER: 2")  # t1 completes "A: "
    assert "\n\nCritic: VERDICT: REVISE" in t
    assert "\n\nProposer (revised): 1. (0,3) is an edge.\nANSWER: 1" in t


def test_critic_and_revision_prompts_carry_transcript():
    inst = _node_degree_instance()
    turns = [{"role": "proposer", "raw": "1. foo\nANSWER: 2"}]
    cp = debate.critic_prompt(inst, turns)
    assert "VERDICT: AGREE" in cp and "VERDICT: REVISE" in cp
    assert "ANSWER: 2" in cp  # the transcript (Proposer's latest) is embedded
    rp = debate.revision_prompt(inst, turns)
    assert "corrected answer" in rp and "ANSWER: <a single integer, the degree>" in rp
    assert "ANSWER: 2" in rp


# --- parsers ------------------------------------------------------------------

def test_parse_proposer_extracts_answer_and_claims():
    raw = ("1. (0,3) is an edge, so node 3 connects to node 0.\n"
           "2. No other edge in the list contains node 3.\n"
           "ANSWER: 1")
    value, ok, claims = debate.parse_proposer(raw, "node_degree")
    assert value == 1 and ok is True
    assert claims == ["(0,3) is an edge, so node 3 connects to node 0.",
                      "No other edge in the list contains node 3."]


def test_parse_proposer_falls_back_without_answer_line():
    # no ANSWER line -> parse the whole text (existing parser: last integer)
    value, ok, claims = debate.parse_proposer("the degree is 3", "node_degree")
    assert value == 3 and ok is True and claims == []


def test_parse_critic_agree_and_revise():
    assert debate.parse_critic("VERDICT: AGREE") == ("AGREE", [], True)
    verdict, problems, ok = debate.parse_critic(
        "VERDICT: REVISE\n- Claim 2: (2,3) is not an edge.\n- Node 4 was missed."
    )
    assert verdict == "REVISE" and ok is True
    assert problems == ["Claim 2: (2,3) is not an edge.", "Node 4 was missed."]


def test_parse_critic_unparseable_defaults_to_agree_but_flagged():
    verdict, problems, ok = debate.parse_critic("looks fine to me, no problems")
    assert verdict == "AGREE" and problems == [] and ok is False  # fake-consensus guard
