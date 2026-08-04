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


# v2's adopted wording. v1's literal above is FROZEN -- it backs results/main + seed11 +
# seed13, and a failure there means someone edited a prompt those results depend on; the
# fix is to add a version, not to update the literal.
#
# v2's literal is NOT frozen: v2 was deliberately edited in place after results/v2-* were
# produced (the symmetry fix, see the module comment), so this literal tracks the current
# text rather than the one those runs used. Anything comparing fresh v2 output against
# results/v2-* is comparing two different prompts.
_ND_PROPOSER_V2 = (
    "Answer the question below using the graph. Build the answer as a numbered list of\n"
    "atomic claims -- each claim one verifiable fact about a single edge (that it exists,\n"
    "or that no further edge involves the node); an edge involves the node whether\n"
    "the node is written first or second -- then give the final answer.\n"
    "Number your claims 1., 2., 3., and so on, one claim per line. Then write a\n"
    "final line that begins with ANSWER: followed by a single integer, the degree. "
    "Write nothing after\nthat line."
)
_ND_REVISION_V2 = (
    "Give your answer.\n"
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


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_incident_tasks_share_one_critic_cue(version):
    # node_degree and connected_nodes must use the exact same Critic cue (no drift);
    # edge_existence uses a different, pair-oriented cue. Holds in every version.
    cue = debate.PROMPT_VERSIONS[version]["critic_cue"]
    nd = debate.critic_prompt(_node_degree_instance(),
                              [{"role": "proposer", "raw": "1. x\nANSWER: 2"}], version)
    cn = debate.critic_prompt(_instance("connected_nodes"),
                              [{"role": "proposer", "raw": "1. x\nANSWER: 1"}], version)
    ee = debate.critic_prompt(_instance("edge_existence"),
                              [{"role": "proposer", "raw": "1. x\nANSWER: No"}], version)
    assert cue["node_degree"] == cue["connected_nodes"]
    assert cue["node_degree"] in nd and cue["connected_nodes"] in cn
    assert cue["edge_existence"] in ee and cue["node_degree"] not in ee


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


def test_v2_claim_kinds_state_that_an_edge_reads_both_ways():
    # The Proposer read an edge only in the direction it was written: recall of a true
    # neighbour was 0.909 with the queried node written first and 0.333 written second
    # (adjacency). The Critic was always told this; the Proposer was not.
    incident = debate.PROMPT_VERSIONS["v2"]["claim_kind"]["node_degree"]
    edge = debate.PROMPT_VERSIONS["v2"]["claim_kind"]["edge_existence"]
    assert "written first or second" in incident
    assert "in\neither order" in edge
    assert "exact edge" not in edge          # "exact" invited literal order-matching


def test_v2_connected_nodes_answers_with_the_far_endpoint():
    # Claims are about edges (pairs), the answer is nodes, and nothing used to bridge
    # them: 578 answers listed the queried node itself.
    block = debate._format_block("connected_nodes", "v2")
    assert "the node at the other end" in block


def test_v1_claim_kinds_are_untouched_by_the_v2_edit():
    # The claim kinds used to be shared across versions, so editing v2's would have
    # silently rewritten v1's prompts -- and v1 backs results/main + seed11 + seed13.
    v1 = debate.PROMPT_VERSIONS["v1"]["claim_kind"]
    assert "written first or second" not in v1["node_degree"]
    assert "exact edge" in v1["edge_existence"]
    assert v1["node_degree"] == v1["connected_nodes"]   # still one shared constant


# --- v2 Critic: the format must let the Critic AGREE with evidence ------------

@pytest.mark.parametrize("task", ["node_degree", "connected_nodes", "edge_existence"])
def test_v2_critic_gives_agree_its_own_evidence_slot(task):
    # v1 put the bullet only under REVISE, so a Critic holding the evidence for a CORRECT
    # answer had nowhere to put it but the REVISE channel: on edge_existence gold=False it
    # said REVISE 96.3% of the time, usually with "- no such edge appears" -- agreement.
    v1 = debate.PROMPT_VERSIONS["v1"]["critic_cue"][task]
    v2 = debate.PROMPT_VERSIONS["v2"]["critic_cue"][task]
    assert v1.count("VERDICT: AGREE\nor") == 1        # v1: AGREE has no bullet
    agree = v2.split("VERDICT: AGREE\n")[1]
    assert agree.lstrip().startswith("-")             # v2: AGREE takes evidence


def test_v2_critic_edge_cue_stops_treating_absence_as_a_problem():
    v2 = debate.PROMPT_VERSIONS["v2"]["critic_cue"]["edge_existence"]
    assert "absence is the evidence FOR that answer" in v2
    # the v1 phrasing filed the No-evidence under the REVISE branch
    assert "Otherwise REVISE and ground it" not in v2


def test_v2_critic_incident_cue_requires_cited_edges_to_be_relevant():
    # v1 required only that a cited edge "actually appears in the list", which an edge from
    # a different part of the graph satisfies: 12.5% of problem lines cited exactly that.
    v2 = debate.PROMPT_VERSIONS["v2"]["critic_cue"]["node_degree"]
    assert "must appear in the list and involve the queried node" in v2


def test_v2_critic_incident_cue_judges_the_answer_not_the_claims():
    # v1 gated AGREE on the claim list ("supporting edges are exactly the ones in the
    # graph") while the experiment scores the answer, so a right answer with an untidy
    # trace had to be REVISEd -- and 19 of 76 such revisions turned it wrong.
    v1 = debate.PROMPT_VERSIONS["v1"]["critic_cue"]["node_degree"]
    v2 = debate.PROMPT_VERSIONS["v2"]["critic_cue"]["node_degree"]
    assert "AGREE only if the Proposer's supporting edges are exactly" in v1
    assert "Judge the answer, not the tidiness of the claims" in v2


def test_v1_critic_cues_are_untouched_by_the_v2_edit():
    # The cues used to be shared across versions and critic_prompt ignored its `version`
    # argument, so editing them would have rewritten v1's Critic too.
    v1 = debate.PROMPT_VERSIONS["v1"]["critic_cue"]
    assert "Judge the answer" not in v1["node_degree"]
    assert "absence is the evidence" not in v1["edge_existence"]


# --- v2 revision: send the Proposer back to the graph, not to the transcript ---

def test_v2_revision_names_the_graph_as_the_source_of_truth():
    # v1's only sourcing instruction was "using the whole exchange", pointing at the
    # transcript. _CRITIC_TOP had always named the graph; the reviser never did, and it
    # showed: where the Critic asserted a queried pair was absent against a "Yes", the
    # Proposer flipped 330/482 times and 151 of those flips gave up a CORRECT answer.
    v1 = debate.PROMPT_VERSIONS["v1"]["revision_top"]
    v2 = debate.PROMPT_VERSIONS["v2"]["revision_top"]
    assert "ONLY source of truth" in v2 and "ONLY source of truth" not in v1
    assert "using the\nwhole exchange" in v1
    assert "check each objection against" in v2


def test_v2_revision_lets_a_correct_answer_stand():
    # v1 asserted the objections were real ("found problems", "verified") and asked for a
    # "corrected" answer, so nothing in the prompt allowed the answer to be kept.
    v1 = debate.PROMPT_VERSIONS["v1"]
    v2 = debate.PROMPT_VERSIONS["v2"]
    assert "found problems" in v1["revision_top"]
    assert "Keep your answer if the objection is wrong" in v2["revision_top"]
    assert v1["revision_preamble"].startswith("Give your corrected answer")
    assert v2["revision_preamble"].strip() == "Give your answer."   # presupposes nothing


def test_every_prompt_piece_is_versioned():
    # Three separate in-place edits were nearly made to prompts v1 depends on, because
    # claim kinds, Critic cues and the revision framing were all module-level constants
    # shared across versions. Each version must now carry its own copy of every piece.
    keys = {"answer_format", "format_block", "proposer_preamble", "revision_preamble",
            "claim_kind", "critic_cue", "revision_top"}
    for version, spec in debate.PROMPT_VERSIONS.items():
        assert set(spec) == keys, version
