"""Tests for `gedebate.eval.diagnostics` -- the debate-trace diagnostic tables.

The counting is the whole point of these tables (a verdict must be attributed to the
Proposer answer it actually judged, not to the final one), so the fixtures are
hand-built traces whose expected counts are obvious by inspection.
"""

from __future__ import annotations

import math

from gedebate.eval.diagnostics import (
    compliance,
    critic_confusion,
    critic_grounding,
    debate_views,
    error_shape,
    pooled_confusion,
    replay_stopping_rules,
    revision_effect,
    turn_split,
)
from gedebate.eval.stats import chi2_2x2

TASK, ENC = "node_degree", "adjacency"
CELL = (TASK, ENC)


def _proposer(answer, raw=None, gen=20, parse_ok=True):
    return {"role": "proposer", "raw": raw if raw is not None else f"1. x\nANSWER: {answer}",
            "parsed": answer, "parse_ok": parse_ok, "claims": [],
            "n_prompt_tokens": 100, "n_gen_tokens": gen}


def _critic(verdict, problems=(), parsed_ok=True, gen=10):
    return {"role": "critic", "raw": f"VERDICT: {verdict}", "verdict": verdict,
            "problems": list(problems), "critic_verdict_parsed": parsed_ok,
            "n_prompt_tokens": 200, "n_gen_tokens": gen}


def _pair(iid, turns, ground_truth, task=TASK, enc=ENC):
    """One (trace, row) pair; the row's `correct` mirrors the runner's own scoring."""
    final = [t for t in turns if t["role"] == "proposer"][-1]
    row = {"instance_id": iid, "task": task, "encoding": enc, "condition": "debate",
           "correct": final["parsed"] == ground_truth, "ground_truth": ground_truth,
           "parsed_answer": final["parsed"], "parse_ok": final["parse_ok"],
           "n_prompt_tokens": 1, "n_gen_tokens": 1, "n_responses": len(turns)}
    return {"instance_id": iid, "turns": turns}, row


def _views(pairs, **kw):
    return debate_views([t for t, _ in pairs], [r for _, r in pairs], **kw)


# --- view construction --------------------------------------------------------

def test_verdict_is_attributed_to_the_answer_it_judged_not_the_final_one():
    # P1 wrong -> critic REVISE -> P2 right -> critic AGREE. The first verdict judged a
    # wrong answer, the second a right one; scoring both against the final would hide it.
    turns = [_proposer(2), _critic("REVISE", ["(0, 1)"]), _proposer(3), _critic("AGREE")]
    (view,) = _views([_pair("7/0/node_degree/adjacency", turns, 3)])

    assert view["turn1_correct"] is False
    assert view["final_correct"] is True
    first, second = view["verdicts"]
    assert (first["judged_correct"], first["revise"]) == (False, True)
    assert (first["next_correct"], first["changed"]) == (True, True)
    assert (second["judged_correct"], second["revise"]) == (True, False)
    assert second["next_correct"] is None  # nothing followed it


def test_traces_without_a_result_row_are_dropped():
    (trace, row) = _pair("7/0/node_degree/adjacency", [_proposer(1), _critic("AGREE")], 1)
    (orphan, _) = _pair("7/9/node_degree/adjacency", [_proposer(1)], 1)
    assert len(debate_views([trace, orphan], [row])) == 1


def test_truncation_needs_the_cap_and_is_inclusive():
    turns = [_proposer(1, gen=256), _critic("AGREE", gen=256)]
    pairs = [_pair("7/0/node_degree/adjacency", turns, 1)]
    (with_cap,) = _views(pairs, max_new_tokens=256)
    (without,) = _views(pairs)
    assert with_cap["turn1_truncated"] is True
    assert with_cap["verdicts"][0]["truncated"] is True
    assert without["turn1_truncated"] is False  # no cap known -> no claim made


# --- turn split ---------------------------------------------------------------

def test_turn_split_separates_the_cot_step_from_the_loop_step():
    pairs = [
        # baseline wrong, turn 1 right, loop keeps it: CoT gained one, loop neutral
        _pair("7/0/node_degree/adjacency", [_proposer(3), _critic("AGREE")], 3),
        # turn 1 right, loop breaks it
        _pair("7/1/node_degree/adjacency",
              [_proposer(3), _critic("REVISE", ["(0, 1)"]), _proposer(2)], 3),
        # turn 1 wrong, loop fixes it
        _pair("7/2/node_degree/adjacency",
              [_proposer(1), _critic("REVISE", ["(0, 1)"]), _proposer(3)], 3),
    ]
    base = [{"instance_id": "7/0/node_degree/adjacency", "correct": False},
            {"instance_id": "7/1/node_degree/adjacency", "correct": True},
            {"instance_id": "7/2/node_degree/adjacency", "correct": False}]
    s = turn_split(_views(pairs), base)[CELL]

    assert (s["baseline_accuracy"], s["turn1_accuracy"], s["final_accuracy"]) == (1 / 3, 2 / 3, 2 / 3)
    assert s["cot_delta"] == 1 / 3 and (s["cot_c"], s["cot_b"]) == (1, 0)
    assert s["loop_delta"] == 0.0 and (s["loop_broke"], s["loop_fixed"]) == (1, 1)


def test_turn_split_without_baseline_rows_leaves_the_cot_columns_undefined():
    pairs = [_pair("7/0/node_degree/adjacency", [_proposer(3), _critic("AGREE")], 3)]
    s = turn_split(_views(pairs))[CELL]
    assert math.isnan(s["baseline_accuracy"]) and math.isnan(s["cot_delta"])
    assert s["turn1_accuracy"] == 1.0  # the loop columns still work


# --- critic confusion ---------------------------------------------------------

def test_critic_confusion_counts_every_verdict_and_orients_phi_toward_usefulness():
    # A perfect Critic: REVISE exactly when the answer it judged is wrong.
    pairs = [
        _pair("7/0/node_degree/adjacency", [_proposer(3), _critic("AGREE")], 3),
        _pair("7/1/node_degree/adjacency", [_proposer(3), _critic("AGREE")], 3),
        _pair("7/2/node_degree/adjacency",
              [_proposer(1), _critic("REVISE", ["(0, 1)"]), _proposer(3), _critic("AGREE")], 3),
    ]
    c = critic_confusion(_views(pairs))[CELL]
    assert (c["ok_agree"], c["ok_revise"], c["bad_agree"], c["bad_revise"]) == (3, 0, 0, 1)
    assert (c["false_alarm"], c["detection"]) == (0.0, 1.0)
    assert c["phi"] > 0  # positive == the Critic is pointing the right way
    assert c["revise_precision"] == 1.0


def test_critic_confusion_at_chance_has_phi_near_zero():
    # REVISE on one right and one wrong answer, AGREE on one of each: no signal.
    pairs = [
        _pair("7/0/node_degree/adjacency", [_proposer(3), _critic("AGREE")], 3),
        _pair("7/1/node_degree/adjacency", [_proposer(1), _critic("AGREE")], 3),
        _pair("7/2/node_degree/adjacency", [_proposer(3), _critic("REVISE", ["(0, 1)"])], 3),
        _pair("7/3/node_degree/adjacency", [_proposer(1), _critic("REVISE", ["(0, 1)"])], 3),
    ]
    c = critic_confusion(_views(pairs))[CELL]
    assert c["phi"] == 0.0 and c["p"] == 1.0
    assert c["revise_precision"] == c["base_rate_wrong"] == 0.5


def test_unparsed_verdicts_are_counted_but_still_scored_as_the_agree_they_became():
    turns = [_proposer(3), _critic("AGREE", parsed_ok=False)]
    c = critic_confusion(_views([_pair("7/0/node_degree/adjacency", turns, 3)]))[CELL]
    assert c["unparsed"] == 1 and c["ok_agree"] == 1


def test_pooled_confusion_sums_across_cells():
    pairs = [
        _pair("7/0/node_degree/adjacency", [_proposer(3), _critic("AGREE")], 3),
        _pair("7/1/node_degree/incident", [_proposer(3), _critic("AGREE")], 3, enc="incident"),
    ]
    views = _views(pairs)
    assert len(critic_confusion(views)) == 2
    assert pooled_confusion(views)["n_verdicts"] == 2


# --- critic grounding ---------------------------------------------------------

def test_critic_grounding_splits_real_edges_from_hallucinated_ones_and_prose():
    turns = [
        _proposer(1),
        _critic("REVISE", ["(0, 1)", "(0, 4)", "no such edge appears"]),
        _proposer(2),
    ]
    pairs = [_pair("7/0/node_degree/adjacency", turns, 2)]
    views = _views(pairs, edgelists={"7/0/node_degree/adjacency": [[0, 1], [2, 3]]})
    g = critic_grounding(views)[CELL]
    assert g["problems"] == 3
    assert (g["real_edge"], g["hallucinated"], g["no_pair"]) == (1, 1, 1)
    assert g["revise_turns"] == 1


def test_critic_grounding_resolves_friendship_names_and_ignores_edge_direction():
    cell = ("node_degree", "friendship")
    turns = [_proposer(1), _critic("REVISE", ["no edge between Robert and James"]), _proposer(2)]
    pairs = [_pair("7/0/node_degree/friendship", turns, 2, enc="friendship")]
    # James=0, Robert=1; the citation names them in the other order.
    views = _views(pairs, edgelists={"7/0/node_degree/friendship": [[0, 1]]})
    g = critic_grounding(views)[cell]
    assert g["real_edge"] == 1


def test_critic_grounding_skips_instances_with_no_edge_list():
    turns = [_proposer(1), _critic("REVISE", ["(0, 1)"]), _proposer(2)]
    g = critic_grounding(_views([_pair("7/0/node_degree/adjacency", turns, 2)]))[CELL]
    assert g["problems"] == 0 and math.isnan(g["real_rate"])


# --- revision effect ----------------------------------------------------------

def test_revision_effect_counts_transitions_and_ignores_unrevised_verdicts():
    pairs = [
        # REVISE that breaks a correct answer
        _pair("7/0/node_degree/adjacency",
              [_proposer(3), _critic("REVISE", ["(0, 1)"]), _proposer(2)], 3),
        # REVISE the Proposer did not act on (same answer back)
        _pair("7/1/node_degree/adjacency",
              [_proposer(1), _critic("REVISE", ["(0, 1)"]), _proposer(1)], 3),
        # terminal REVISE with no room left to revise: not a transition
        _pair("7/2/node_degree/adjacency", [_proposer(1), _critic("REVISE", ["(0, 1)"])], 3),
    ]
    r = revision_effect(_views(pairs))[CELL]
    assert r["revisions"] == 2  # the terminal REVISE is excluded
    assert r["changed"] == 1 and r["changed_rate"] == 0.5
    assert (r["ok_to_bad"], r["bad_to_bad"]) == (1, 1)
    assert r["net"] == -1


# --- compliance ---------------------------------------------------------------

def test_compliance_separates_a_missing_answer_line_from_a_failed_parse():
    pairs = [
        # no ANSWER: line, but the fallback parse still recovers the degree
        _pair("7/0/node_degree/adjacency",
              [_proposer(3, raw="1. edge (0,1)\nThe degree of node 0 is 3."), _critic("AGREE")], 3),
        # ran into the cap and parsed nothing
        _pair("7/1/node_degree/adjacency",
              [_proposer(None, raw="1. <one atomic claim>", gen=256, parse_ok=False),
               _critic("AGREE", parsed_ok=False)], 3),
    ]
    c = compliance(_views(pairs, max_new_tokens=256))[CELL]
    assert c["n"] == 2
    assert c["turn1_no_answer_line"] == 2  # both lack the line
    assert c["turn1_unparsed"] == 1        # only one actually failed to parse
    assert c["turn1_truncated"] == 1
    assert (c["proposer_turns"], c["critic_turns"]) == (2, 2)
    assert c["critic_no_verdict"] == 1


# --- counterfactual stopping rules --------------------------------------------

def _replay(pairs, edgelists=None):
    return replay_stopping_rules(_views(pairs, edgelists=edgelists))[CELL]


def test_turn1_only_rule_recovers_an_answer_the_loop_broke():
    # P1 right -> REVISE -> P2 wrong. The real run scores 0; stopping at turn 1 scores 1.
    turns = [_proposer(3), _critic("REVISE", ["(0, 1)"]), _proposer(2)]
    r = _replay([_pair("7/0/node_degree/adjacency", turns, 3)])
    assert r["actual"]["accuracy"] == 0.0
    assert r["turn1_only"]["accuracy"] == 1.0
    assert r["turn1_only"]["delta"] == 1.0
    assert (r["turn1_only"]["b"], r["turn1_only"]["c"]) == (0, 1)


def test_at_most_one_revision_stops_after_the_first_revision():
    # right -> REVISE -> wrong -> REVISE -> right. One revision only lands on wrong;
    # the real run runs on and recovers.
    turns = [_proposer(3), _critic("REVISE", ["(0, 1)"]), _proposer(2),
             _critic("REVISE", ["(0, 1)"]), _proposer(3)]
    r = _replay([_pair("7/0/node_degree/adjacency", turns, 3)])
    assert r["actual"]["accuracy"] == 1.0
    assert r["at_most_one_revision"]["accuracy"] == 0.0


def test_at_most_one_revision_is_a_no_op_when_the_critic_agreed():
    turns = [_proposer(3), _critic("AGREE")]
    r = _replay([_pair("7/0/node_degree/adjacency", turns, 3)])
    assert r["at_most_one_revision"]["accuracy"] == r["actual"]["accuracy"] == 1.0


def test_hallucination_gate_vetoes_a_revise_citing_a_pair_that_is_not_an_edge():
    # The Critic cites (0, 4), which is not in the graph, and the revision breaks a
    # correct answer. Gating on hallucinated evidence keeps the correct answer.
    turns = [_proposer(3), _critic("REVISE", ["(0, 4)"]), _proposer(2)]
    edges = {"7/0/node_degree/adjacency": [[0, 1], [2, 3]]}
    r = _replay([_pair("7/0/node_degree/adjacency", turns, 3)], edges)
    assert r["actual"]["accuracy"] == 0.0
    assert r["gate_hallucinated"]["accuracy"] == 1.0


def test_hallucination_gate_lets_a_real_citation_through():
    turns = [_proposer(2), _critic("REVISE", ["(0, 1)"]), _proposer(3)]
    edges = {"7/0/node_degree/adjacency": [[0, 1], [2, 3]]}
    r = _replay([_pair("7/0/node_degree/adjacency", turns, 3)], edges)
    assert r["gate_hallucinated"]["accuracy"] == r["actual"]["accuracy"] == 1.0


def test_the_two_gates_differ_only_on_prose_only_critiques():
    # "no such edge appears" cites nothing: permissive lets it through, strict vetoes it.
    turns = [_proposer(3), _critic("REVISE", ["no such edge appears"]), _proposer(2)]
    edges = {"7/0/node_degree/adjacency": [[0, 1]]}
    r = _replay([_pair("7/0/node_degree/adjacency", turns, 3)], edges)
    assert r["gate_hallucinated"]["accuracy"] == 0.0  # not vetoed, revision stands
    assert r["gate_must_cite"]["accuracy"] == 1.0     # vetoed, turn 1 stands


def test_gates_collapse_onto_the_real_run_when_no_graph_was_supplied():
    # Evidence is unknown, not ungrounded: a rule must not veto on missing data.
    turns = [_proposer(3), _critic("REVISE", ["(0, 4)"]), _proposer(2)]
    r = _replay([_pair("7/0/node_degree/adjacency", turns, 3)])
    assert r["gate_hallucinated"]["accuracy"] == r["actual"]["accuracy"] == 0.0
    assert r["gate_must_cite"]["accuracy"] == 0.0


def test_a_rule_that_stops_at_a_later_verdict_uses_the_answer_standing_there():
    # grounded REVISE (passes) -> wrong answer -> hallucinated REVISE (vetoed).
    # The gate stops at the second verdict, so the wrong middle answer is final.
    turns = [_proposer(1), _critic("REVISE", ["(0, 1)"]), _proposer(2),
             _critic("REVISE", ["(0, 4)"]), _proposer(3)]
    edges = {"7/0/node_degree/adjacency": [[0, 1], [2, 3]]}
    r = _replay([_pair("7/0/node_degree/adjacency", turns, 3)], edges)
    assert r["actual"]["accuracy"] == 1.0
    assert r["gate_hallucinated"]["accuracy"] == 0.0


# --- error shape --------------------------------------------------------------

def test_error_shape_signs_a_degree_error_toward_over_or_undercounting():
    pairs = [
        _pair("7/0/node_degree/adjacency", [_proposer(5)], 3),  # +2, overcount
        _pair("7/1/node_degree/adjacency", [_proposer(2)], 3),  # -1, undercount, off by one
        _pair("7/2/node_degree/adjacency", [_proposer(3)], 3),  # exact
    ]
    m = error_shape(_views(pairs))[CELL]["metrics"]
    assert abs(m["mean_signed_error"] - 1 / 3) < 1e-12
    assert (m["overcount_rate"], m["undercount_rate"]) == (1 / 3, 1 / 3)
    assert m["off_by_one_rate"] == 1 / 3


def test_error_shape_separates_extra_neighbours_from_missing_ones():
    cell = ("connected_nodes", "friendship")
    pairs = [
        # gold {1,2}: predicted {1,2,3} has an extra and nothing missing
        _pair("7/0/connected_nodes/friendship", [_proposer([1, 2, 3])], [1, 2],
              task="connected_nodes", enc="friendship"),
        # predicted {1} is missing one and has no extra
        _pair("7/1/connected_nodes/friendship", [_proposer([1])], [1, 2],
              task="connected_nodes", enc="friendship"),
    ]
    m = error_shape(_views(pairs))[cell]["metrics"]
    assert (m["has_extra_rate"], m["has_missing_rate"]) == (0.5, 0.5)
    assert abs(m["mean_jaccard"] - (2 / 3 + 1 / 2) / 2) < 1e-12


def test_error_shape_excludes_unparsed_answers_but_still_counts_them():
    pairs = [
        _pair("7/0/node_degree/adjacency", [_proposer(3)], 3),
        _pair("7/1/node_degree/adjacency",
              [_proposer(None, raw="1. <one atomic claim>", parse_ok=False)], 3),
    ]
    s = error_shape(_views(pairs))[CELL]
    assert (s["n"], s["n_parsed"], s["n_unparsed"]) == (2, 1, 1)
    assert s["metrics"]["mean_signed_error"] == 0.0  # the parsed one was exact


# --- the underlying association test ------------------------------------------

def test_chi2_2x2_matches_the_textbook_value_and_degenerates_safely():
    s = chi2_2x2(10, 20, 30, 40)
    assert abs(s["phi"] - (10 * 40 - 20 * 30) / math.sqrt(30 * 70 * 40 * 60)) < 1e-12
    assert abs(s["chi2"] - 100 * s["phi"] ** 2) < 1e-12
    assert abs(s["odds_ratio"] - (10 * 40) / (20 * 30)) < 1e-12

    empty = chi2_2x2(0, 0, 5, 5)  # an empty margin: no association is defined
    assert empty["p"] == 1.0 and math.isnan(empty["phi"])
