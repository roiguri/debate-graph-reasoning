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

# --- drift guard: the frozen wording must be reproduced byte for byte ---------
#
# v2 is FROZEN: it backs results/v2-* (Qwen) and results/llama70b-* (Llama), and every
# number in docs/findings.md section 4. A failure here means someone edited a prompt that
# published results depend on. The fix is to add a new version key -- as v3 is -- never to
# update this literal. (v1 and a later revision of v2 were deleted; see the module comment.)

_V2_ND_PROPOSER = (
    "Answer the question below using the graph. Build the answer as a numbered list of\n"
    "atomic claims -- each claim one verifiable fact about a single edge (that it exists,\n"
    "or that no further edge involves the node) -- then give the final answer.\n"
    "Number your claims 1., 2., 3., and so on, one claim per line. Then write a\n"
    "final line that begins with ANSWER: followed by a single integer, the degree. "
    "Write nothing after\nthat line."
)
_V2_ND_REVISION = (
    "Give your corrected answer.\n"
    "Number your claims 1., 2., 3., and so on, one claim per line. Then write a\n"
    "final line that begins with ANSWER: followed by a single integer, the degree. "
    "Write nothing after\nthat line."
)


def test_v2_prompts_are_byte_identical_to_the_frozen_wording():
    inst = _node_degree_instance()
    assert debate.proposer_prompt(inst, "v2") == f"{_V2_ND_PROPOSER}\n\n{inst.question}"
    turns = [{"role": "proposer", "raw": "1. foo\nANSWER: 2"}]
    assert debate.revision_prompt(inst, turns, "v2").endswith(f"\n\n{_V2_ND_REVISION}")


def test_v2_and_v3_are_both_present_and_v2_is_the_default():
    # Manifests record the version, and the analysis filters on it, so these key names
    # are load-bearing. The default stays v2: it is what every committed result was run
    # under, and what a config omitting `prompt_version` must keep meaning.
    assert list(debate.PROMPT_VERSIONS) == ["v2", "v3"]
    assert debate.DEFAULT_PROMPT_VERSION == "v2"


def test_v3_differs_from_v2_in_every_role_it_changed():
    inst = _node_degree_instance()
    turns = [{"role": "proposer", "raw": "1. foo\nANSWER: 2"}]
    assert debate.proposer_prompt(inst, "v3") != debate.proposer_prompt(inst, "v2")
    assert debate.critic_prompt(inst, turns, "v3") != debate.critic_prompt(inst, turns, "v2")
    # the revision role was NOT changed, so only the transcript inside it may differ
    assert debate.PROMPT_VERSIONS["v3"]["revision_top"] == \
        debate.PROMPT_VERSIONS["v2"]["revision_top"]


def test_the_versions_share_no_mutable_constant():
    """Editing one version must not be able to reach the other.

    Two versions pointing at ONE object is exactly what let an in-place edit rewrite a
    published prompt, so every per-version piece is a distinct object even where the text
    is identical today.
    """
    v2, v3 = debate.PROMPT_VERSIONS["v2"], debate.PROMPT_VERSIONS["v3"]
    assert set(v2) == set(v3)
    for key in v2:
        if isinstance(v2[key], dict):  # the per-task dicts must not be the same object
            assert v2[key] is not v3[key], key


def test_the_frozen_prompt_carries_no_fill_in_template():
    # v1's "<one atomic claim>" placeholder was echoed verbatim to the token cap
    # (docs/findings.md 3d), so the kept wording must contain no fill-in block.
    for task in debate.supported_tasks():
        block = debate._format_block(task)
        assert "<" not in block and ">" not in block, task


def test_connected_nodes_answer_hint_names_no_label_space():
    # The defect v2 was adopted to fix: v1 said "node ids" regardless of encoding, so on
    # friendship (named nodes) the model answered in integers the parser could not
    # resolve -- 64 of 600 answers unparseable (docs/findings.md 3c).
    block = debate._format_block("connected_nodes")
    assert "node ids" not in block
    assert "as the graph writes them" in block


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
    assert "a single integer, the degree" in nd
    assert "written exactly as the graph writes them" in cn
    assert "connected_nodes" in debate.supported_tasks()


# --- prompt builders ----------------------------------------------------------

def test_proposer_prompt_wraps_question_with_instruction():
    inst = _node_degree_instance()
    p = debate.proposer_prompt(inst)
    assert p.endswith(inst.question)          # question verbatim at the end
    assert "numbered list of\natomic claims" in p  # v2, the default
    assert "ANSWER: followed by a single integer, the degree" in p


def test_unsupported_task_raises():
    with pytest.raises(NotImplementedError):
        debate._require_supported("path_len")  # a task with no approved prompt


def test_all_three_tasks_supported():
    assert set(debate.supported_tasks()) == {
        "node_degree", "connected_nodes", "edge_existence"}


def test_incident_tasks_share_one_critic_cue():
    # node_degree and connected_nodes must use the exact same Critic cue (no drift);
    # edge_existence uses a different, pair-oriented cue.
    cue = debate.PROMPT_VERSIONS["v2"]["critic_cue"]
    nd = debate.critic_prompt(_node_degree_instance(), [{"role": "proposer", "raw": "1. x\nANSWER: 2"}])
    ee = debate.critic_prompt(_instance("edge_existence"), [{"role": "proposer", "raw": "1. x\nANSWER: No"}])
    assert cue["node_degree"] == cue["connected_nodes"]
    assert cue["node_degree"] in nd
    assert cue["edge_existence"] in ee and cue["node_degree"] not in ee


def test_every_prompt_piece_is_carried_by_the_version():
    # Three pieces (claim kinds, Critic cues, revision framing) were once module-level
    # constants shared across versions, which made an in-place edit silently rewrite a
    # published prompt. Every piece must hang off the version spec.
    keys = {"answer_format", "format_block", "proposer_preamble", "revision_preamble",
            "critic_top", "critic_cue", "revision_top"}
    for version, spec in debate.PROMPT_VERSIONS.items():
        assert set(spec) == keys, version



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
    assert "corrected answer" in rp and "ANSWER: followed by a single integer, the degree" in rp
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
    # Robert is 1 (the queried node), Michael 3, Susan 16.
    value, ok, _ = debate.parse_proposer(
        raw, "connected_nodes", encoding="friendship", node_ids=[1])
    assert ok
    # Whole-text parsing returned [3, 16]: Susan (16) harvested from a negated claim.
    assert 16 not in value
    # The fallback line's claim number ("3:") is stripped, leaving the bare list
    # "Robert, Michael" -- which names Robert, so the answer CLAIMS a self-connection
    # and is kept (and will score wrong against a gold that never contains the source).
    assert value == [1, 3]


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


def test_bare_answer_line_is_the_empty_set_for_connected_nodes():
    # "...or none" is answered by writing ANSWER: and stopping. The old regex needed a
    # character after the colon, so this fell through to the last-line fallback, landed on
    # the string "ANSWER:", and scored a parse failure -- on instances whose gold IS empty.
    raw = "1. No edge involves Robert.\nANSWER:"
    value, ok, _ = debate.parse_proposer(
        raw, "connected_nodes", encoding="friendship", node_ids=[1])
    assert (value, ok) == ([], True)


@pytest.mark.parametrize("task", ["node_degree", "edge_existence"])
def test_bare_answer_line_is_a_failure_where_empty_is_not_an_answer(task):
    # A degree and a Yes/No are never empty, so an empty line is a non-answer, not "none".
    value, ok, _ = debate.parse_proposer("1. a claim\nANSWER:", task)
    assert (value, ok) == (None, False)


def test_answer_on_the_line_after_the_label_is_still_the_answer():
    # The old regex's `\s*` crossed the newline to find this; the line-scoped one must
    # keep doing so, or a split answer becomes a phantom empty set.
    value, ok, _ = debate.parse_proposer("1. a claim\nANSWER:\n3", "node_degree")
    assert (value, ok) == (3, True)


def test_fallback_line_does_not_read_its_claim_number_as_a_node():
    # Under adjacency/incident the labels ARE integers, so the leading "5." parsed as
    # node 5 and corrupted an otherwise correct answer.
    raw = "4. Node 3 has one edge.\n5. The nodes connected to 3 in alphabetical order are 0."
    value, ok, _ = debate.parse_proposer(
        raw, "connected_nodes", encoding="adjacency", node_ids=[3])
    assert (value, ok) == ([0], True)


