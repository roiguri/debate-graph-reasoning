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


def _instance(task):
    return next(i for i in build_dataset(n_graphs=4, seed=7)
                if i.task == task and i.encoding == "adjacency")


# --- drift guard: the shared scaffold must reproduce the approved wording exactly ---
# These golden literals are the wording that was approved + piloted. The prompts are now
# assembled from shared building blocks; if a refactor of those blocks changes the emitted
# text for node_degree, this fails -- that is the whole point (no silent prompt drift).

_ND_PROPOSER = (
    "Answer the question below using the graph. Build the answer as a numbered list of\n"
    "atomic claims -- each claim one verifiable fact about a single edge (that it exists,\n"
    "or that no further edge involves the node) -- then give the final answer.\n"
    "Use exactly this format and nothing else:\n"
    "1. <one atomic claim>\n"
    "2. <one atomic claim>\n"
    "(as many as needed)\n"
    "ANSWER: <a single integer, the degree>"
)
_ND_REVISION = (
    "Give your corrected answer in exactly this format and nothing else:\n"
    "1. <one atomic claim>\n"
    "2. <one atomic claim>\n"
    "(as many as needed)\n"
    "ANSWER: <a single integer, the degree>"
)


# v2's adopted wording. BOTH literals are now frozen: v1 backs results/main + seed11 +
# seed13, v2 backs the full re-run, and a version stops being editable once results
# depend on it. A failure here means someone edited a frozen prompt; the fix is to add
# v3, not to update the literal.
_ND_PROPOSER_V2 = (
    "Answer the question below using the graph. Build the answer as a numbered list of\n"
    "atomic claims -- each claim one verifiable fact about a single edge (that it exists,\n"
    "or that no further edge involves the node) -- then give the final answer.\n"
    "Number your claims 1., 2., 3., and so on, one claim per line. Then write a\n"
    "final line that begins with ANSWER: followed by a single integer, the degree. "
    "Write nothing after\nthat line."
)
_ND_REVISION_V2 = (
    "Give your corrected answer.\n"
    "Number your claims 1., 2., 3., and so on, one claim per line. Then write a\n"
    "final line that begins with ANSWER: followed by a single integer, the degree. "
    "Write nothing after\nthat line."
)


def test_node_degree_prompts_are_byte_identical_to_approved_wording():
    inst = _node_degree_instance()
    assert debate.proposer_prompt(inst) == f"{_ND_PROPOSER}\n\n{inst.question}"
    turns = [{"role": "proposer", "raw": "1. foo\nANSWER: 2"}]
    assert debate.revision_prompt(inst, turns).endswith(f"\n\n{_ND_REVISION}")


def test_v2_prompts_are_byte_identical_to_approved_wording():
    inst = _node_degree_instance()
    assert debate.proposer_prompt(inst, "v2") == f"{_ND_PROPOSER_V2}\n\n{inst.question}"
    turns = [{"role": "proposer", "raw": "1. foo\nANSWER: 2"}]
    assert debate.revision_prompt(inst, turns, "v2").endswith(f"\n\n{_ND_REVISION_V2}")


def test_v1_is_the_default_so_existing_configs_are_unaffected():
    inst = _node_degree_instance()
    assert debate.DEFAULT_PROMPT_VERSION == "v1"
    assert debate.proposer_prompt(inst) == debate.proposer_prompt(inst, "v1")
    assert debate.proposer_prompt(inst, "v2") != debate.proposer_prompt(inst, "v1")


def test_v2_drops_the_template_the_model_copied():
    # v1's "<one atomic claim>" placeholder was echoed verbatim to the token cap
    # (docs/findings.md 3d), so v2 must contain no fill-in template at all.
    for task in debate.supported_tasks("v2"):
        block = debate._format_block(task, "v2")
        assert "<" not in block and ">" not in block, task
    assert "<one atomic claim>" in debate._format_block("node_degree", "v1")


def test_v2_connected_nodes_answer_hint_names_no_label_space():
    # v1 said "node ids" regardless of encoding, so on friendship (named nodes) the
    # model answered in integers the parser could not resolve (docs/findings.md 3c).
    v1 = debate._format_block("connected_nodes", "v1")
    v2 = debate._format_block("connected_nodes", "v2")
    assert "node ids" in v1
    assert "node ids" not in v2
    assert "as the graph writes them" in v2


def test_unknown_prompt_version_is_rejected():
    inst = _node_degree_instance()
    with pytest.raises(ValueError, match="unknown prompt version"):
        debate.proposer_prompt(inst, "v99")


def test_proposer_and_revision_share_one_format_block():
    inst = _node_degree_instance()
    block = debate._format_block("node_degree")  # the single source of the claim+ANSWER format
    assert block in debate.proposer_prompt(inst)
    assert block in debate.revision_prompt(inst, [{"role": "proposer", "raw": "1. x\nANSWER: 2"}])


def test_connected_nodes_shares_scaffold_differs_only_in_answer_line():
    nd = debate.proposer_prompt(_node_degree_instance())
    cn = debate.proposer_prompt(_instance("connected_nodes"))
    # identical scaffold up to the ANSWER line; only the ANSWER tail differs
    assert nd.split("ANSWER:")[0] == cn.split("ANSWER:")[0]
    assert "ANSWER: <a single integer, the degree>" in nd
    assert "ANSWER: <a comma-separated list of node ids, or none>" in cn
    assert "connected_nodes" in debate.supported_tasks()


# --- prompt builders ----------------------------------------------------------

def test_proposer_prompt_wraps_question_with_instruction():
    inst = _node_degree_instance()
    p = debate.proposer_prompt(inst)
    assert p.endswith(inst.question)          # question verbatim at the end
    assert "numbered list of\natomic claims" in p
    assert "ANSWER: <a single integer, the degree>" in p


def test_unsupported_task_raises():
    with pytest.raises(NotImplementedError):
        debate._require_supported("path_len")  # a task with no approved prompt


def test_all_three_tasks_supported():
    for version in debate.PROMPT_VERSIONS:
        assert set(debate.supported_tasks(version)) == {
            "node_degree", "connected_nodes", "edge_existence"}


def test_incident_tasks_share_one_critic_cue():
    # node_degree and connected_nodes must use the exact same Critic cue (no drift);
    # edge_existence uses a different, pair-oriented cue.
    nd = debate.critic_prompt(_node_degree_instance(), [{"role": "proposer", "raw": "1. x\nANSWER: 2"}])
    cn = debate.critic_prompt(_instance("connected_nodes"), [{"role": "proposer", "raw": "1. x\nANSWER: 1"}])
    ee = debate.critic_prompt(_instance("edge_existence"), [{"role": "proposer", "raw": "1. x\nANSWER: No"}])
    assert debate._CRITIC_CUE_INCIDENT in nd and debate._CRITIC_CUE_INCIDENT in cn
    assert debate._CRITIC_CUE_EDGE in ee and debate._CRITIC_CUE_INCIDENT not in ee


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


# --- the answer fallback never reads the reasoning ----------------------------

def test_missing_answer_line_falls_back_to_the_last_line_only():
    # The trace mentions Susan only to DENY the edge. Scanning the whole output put her
    # in the answer; last-line reads just the stated answer. (docs/findings.md 3g)
    raw = ("1. Robert is connected to Michael.\n"
           "2. Robert is not connected to Susan.\n"
           "3: Robert, Michael")
    # Robert (1) is the queried node, so he is dropped as a never-neighbour of himself;
    # Michael is 3 and Susan is 16.
    value, ok, _ = debate.parse_proposer(
        raw, "connected_nodes", encoding="friendship", node_ids=[1])
    assert ok
    assert value == [3]                       # Michael only
    # Whole-text parsing returned [3, 16]: Susan (16) harvested from a negated claim.
    assert 16 not in value


def test_an_explicit_answer_line_still_wins_over_the_last_line():
    raw = "1. a claim\nANSWER: 4\ntrailing chatter 9"
    value, ok, _ = debate.parse_proposer(raw, "node_degree")
    assert (value, ok) == (4, True)


def test_last_line_fallback_is_used_when_there_is_no_answer_line():
    raw = "1. Node 5 is connected to node 0.\n5. The degree of node 5 is 3."
    value, ok, _ = debate.parse_proposer(raw, "node_degree")
    assert (value, ok) == (3, True)


def test_empty_output_parses_as_a_failure_not_a_crash():
    value, ok, _ = debate.parse_proposer("", "node_degree")
    assert (value, ok) == (None, False)
