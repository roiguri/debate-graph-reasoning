"""Tests for the majority-vote report view (`summarize_votes` + formatters).

Accuracy here is per *instance* (vote of the N draws), distinct from the per-draw
accuracy that baseline-style `summarize` reports.
"""

from __future__ import annotations

from gedebate.eval.report import (
    compare_baseline_vote,
    comparison_to_csv,
    format_vote_summary,
    summarize_votes,
    vote_summary_to_csv,
)


def _row(iid, task, enc, gt, parsed, si, gen=5):
    """One majority-vote sample row (lean schema subset the report reads)."""
    return {
        "instance_id": iid, "task": task, "encoding": enc,
        "ground_truth": gt, "parsed_answer": parsed,
        "correct": parsed == gt, "sample_index": si, "n_gen_tokens": gen,
    }


def test_vote_beats_single_draw_when_majority_is_right():
    # One instance, 3 draws: majority "True" is correct though one draw was wrong.
    rows = [
        _row("7/0/edge_existence/adjacency", "edge_existence", "adjacency", True, True, 0, gen=4),
        _row("7/0/edge_existence/adjacency", "edge_existence", "adjacency", True, False, 1, gen=6),
        _row("7/0/edge_existence/adjacency", "edge_existence", "adjacency", True, True, 2, gen=5),
    ]
    s = summarize_votes(rows)
    cell = s[("edge_existence", "adjacency")]
    assert cell["n"] == 1
    assert cell["voted_accuracy"] == 1.0           # vote is correct
    assert abs(cell["per_sample_accuracy"] - 2 / 3) < 1e-9  # 2 of 3 draws correct
    assert cell["total_gen_tokens"] == 15          # summed across draws (N x cost)
    assert cell["n_samples"] == [3]


def test_vote_wrong_majority_scores_incorrect():
    rows = [
        _row("7/1/node_degree/incident", "node_degree", "incident", 3, 2, 0),
        _row("7/1/node_degree/incident", "node_degree", "incident", 3, 2, 1),
        _row("7/1/node_degree/incident", "node_degree", "incident", 3, 3, 2),
    ]
    s = summarize_votes(rows)
    cell = s[("node_degree", "incident")]
    assert cell["voted_accuracy"] == 0.0           # majority voted 2, gt is 3
    assert abs(cell["per_sample_accuracy"] - 1 / 3) < 1e-9


def test_all_parse_fail_is_incorrect_and_low_parse_rate():
    rows = [
        _row("7/2/edge_existence/adjacency", "edge_existence", "adjacency", True, None, 0),
        _row("7/2/edge_existence/adjacency", "edge_existence", "adjacency", True, None, 1),
    ]
    cell = summarize_votes(rows)[("edge_existence", "adjacency")]
    assert cell["voted_accuracy"] == 0.0 and cell["parse_ok_rate"] == 0.0


def test_two_instances_aggregate_per_cell():
    rows = [
        _row("7/0/edge_existence/adjacency", "edge_existence", "adjacency", True, True, 0),
        _row("7/0/edge_existence/adjacency", "edge_existence", "adjacency", True, True, 1),
        _row("7/3/edge_existence/adjacency", "edge_existence", "adjacency", False, True, 0),
        _row("7/3/edge_existence/adjacency", "edge_existence", "adjacency", False, True, 1),
    ]
    cell = summarize_votes(rows)[("edge_existence", "adjacency")]
    assert cell["n"] == 2 and cell["voted_accuracy"] == 0.5  # one right, one wrong


def _brow(iid, task, enc, correct, gen):
    """One baseline-shaped row (what report.summarize + the McNemar pairing consume)."""
    return {"instance_id": iid, "task": task, "encoding": enc, "correct": correct,
            "parse_ok": True, "n_gen_tokens": gen}


def test_compare_baseline_vote_delta_token_mult_and_mcnemar():
    # baseline: 2 instances, both correct -> acc 1.0, 4 gen tokens total.
    base = [
        _brow("7/0/edge_existence/adjacency", "edge_existence", "adjacency", True, 2),
        _brow("7/1/edge_existence/adjacency", "edge_existence", "adjacency", True, 2),
    ]
    # majority vote: instance 7/0 votes right, 7/1 votes wrong -> vote_acc 0.5.
    # 4 rows x 3 tokens = 12 gen tokens -> token_mult 12/4 = 3.0.
    mv = [
        _row("7/0/edge_existence/adjacency", "edge_existence", "adjacency", True, True, 0, gen=3),
        _row("7/0/edge_existence/adjacency", "edge_existence", "adjacency", True, True, 1, gen=3),
        _row("7/1/edge_existence/adjacency", "edge_existence", "adjacency", True, False, 0, gen=3),
        _row("7/1/edge_existence/adjacency", "edge_existence", "adjacency", True, False, 1, gen=3),
    ]
    cmp = compare_baseline_vote(base, mv)
    c = cmp[("edge_existence", "adjacency")]
    assert c["baseline_accuracy"] == 1.0 and c["voted_accuracy"] == 0.5
    assert abs(c["delta"] - (-0.5)) < 1e-9
    # compute per instance: baseline 1 response / 2 tok; vote 2 responses / 6 tok
    assert c["baseline_responses"] == 1.0 and c["vote_responses"] == 2.0
    assert abs(c["response_mult"] - 2.0) < 1e-9
    assert c["baseline_tokens"] == 2.0 and c["vote_tokens"] == 6.0
    assert abs(c["token_mult"] - 3.0) < 1e-9
    # paired McNemar: instance 7/1 is baseline-right / vote-wrong -> b=1, c=0
    assert c["b"] == 1 and c["c"] == 0 and c["discordant"] == 1
    assert c["mcnemar_p"] == 1.0  # a single discordant pair -> exact binomial p=1.0

    csv = comparison_to_csv(cmp)
    assert csv.startswith("task,encoding,baseline_acc,vote_acc,delta")
    assert "response_mult" in csv and "mcnemar_p" in csv
    assert "edge_existence,adjacency,1.0000,0.5000,-0.5000,1,2,2.00,2.0,6.0,3.00,1,0,1,1" in csv


def test_compare_only_cells_in_both_conditions():
    base = [_brow("7/0/node_degree/incident", "node_degree", "incident", True, 5)]
    mv = [  # a different cell -> no overlap, empty comparison
        _row("7/0/edge_existence/adjacency", "edge_existence", "adjacency", True, True, 0),
    ]
    assert compare_baseline_vote(base, mv) == {}


def test_format_and_csv_render():
    rows = [
        _row("7/0/node_degree/incident", "node_degree", "incident", 3, 3, 0),
        _row("7/0/node_degree/incident", "node_degree", "incident", 3, 3, 1),
    ]
    s = summarize_votes(rows)
    out = format_vote_summary(s)
    assert "vote_acc" in out and "node_degree" in out
    csv = vote_summary_to_csv(s)
    assert csv.startswith("task,encoding,n,n_samples,voted_accuracy")
    assert "node_degree,incident,1,2,1.0000" in csv
