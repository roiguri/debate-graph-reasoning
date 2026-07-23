"""Tests for `gedebate.eval.stats` -- paired significance for the fragility claim.

The tests build rows by hand with known paired structure so the discordant counts
(McNemar's b/c) and Cochran's Q are checkable against closed-form values.
"""

from __future__ import annotations

import math

from gedebate.eval.stats import (
    cochran_q,
    mcnemar,
    task_significance,
    wilson_ci,
)


def _row(task, enc, gidx, correct, seed=7):
    """One attempt row; instance_id is 'seed/graph_index/task/encoding'."""
    return {
        "task": task,
        "encoding": enc,
        "correct": correct,
        "instance_id": f"{seed}/{gidx}/{task}/{enc}",
    }


# --- wilson_ci ----------------------------------------------------------------

def test_wilson_ci_brackets_point_estimate():
    lo, hi = wilson_ci(70, 100)
    assert lo < 0.70 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_ci_degenerate_counts():
    assert wilson_ci(0, 0) == (0.0, 1.0)      # no data -> maximally wide
    lo, hi = wilson_ci(0, 50)                  # all wrong: interval stays in [0,1]
    assert lo == 0.0 and 0.0 < hi < 0.2
    lo, hi = wilson_ci(50, 50)                 # all right
    assert hi == 1.0 and 0.8 < lo < 1.0


# --- mcnemar (paired, two encodings) ------------------------------------------

def test_mcnemar_counts_discordant_pairs():
    # graph 0: A right, B wrong (b); graph 1: A wrong, B right (c);
    # graph 2: both right (concordant, ignored); graph 3: A right, B wrong (b).
    rows = [
        _row("t", "A", 0, True), _row("t", "B", 0, False),
        _row("t", "A", 1, False), _row("t", "B", 1, True),
        _row("t", "A", 2, True), _row("t", "B", 2, True),
        _row("t", "A", 3, True), _row("t", "B", 3, False),
    ]
    m = mcnemar(rows, "A", "B")
    assert m["b"] == 2 and m["c"] == 1  # discordant only


def test_mcnemar_no_discordance_is_p1():
    rows = [_row("t", "A", g, True) for g in range(3)]
    rows += [_row("t", "B", g, True) for g in range(3)]
    assert mcnemar(rows, "A", "B")["p"] == 1.0


def test_mcnemar_large_lopsided_gap_is_significant():
    # 40 graphs A-right/B-wrong, 2 the other way: overwhelming, p should be tiny.
    rows = []
    for g in range(40):
        rows += [_row("t", "A", g, True), _row("t", "B", g, False)]
    for g in range(40, 42):
        rows += [_row("t", "A", g, False), _row("t", "B", g, True)]
    m = mcnemar(rows, "A", "B")
    assert m["b"] == 40 and m["c"] == 2
    assert m["p"] < 1e-6


# --- cochran_q (paired, k encodings) ------------------------------------------

def test_cochran_q_all_equal_is_no_evidence():
    # every graph identical across encodings -> no discordance -> Q=0, p=1.
    rows = []
    for g in range(5):
        for e in ("A", "B", "C"):
            rows.append(_row("t", e, g, True))
    q = cochran_q(rows, ["A", "B", "C"])
    assert q["q"] == 0.0 and q["p"] == 1.0 and q["df"] == 2


def test_cochran_q_matches_closed_form():
    # Hand-built: C always right, A always wrong, B right on half the graphs.
    rows = []
    for g in range(10):
        rows += [_row("t", "A", g, False), _row("t", "C", g, True)]
        rows.append(_row("t", "B", g, g < 5))
    q = cochran_q(rows, ["A", "B", "C"])
    # df=2 survival is exp(-Q/2); just assert it's a strong, consistent signal.
    assert q["df"] == 2 and q["n"] == 10
    assert math.isclose(q["p"], math.exp(-q["q"] / 2), rel_tol=1e-12)
    assert q["q"] > 6  # clearly discordant


# --- task_significance (integration over rows) --------------------------------

def test_task_significance_picks_best_worst_and_reports_both_tests():
    rows = []
    for g in range(20):
        rows.append(_row("node_degree", "incident", g, True))       # best: all right
        rows.append(_row("node_degree", "adjacency", g, g < 5))     # worst: mostly wrong
        rows.append(_row("node_degree", "friendship", g, g < 12))
    sig = task_significance(rows)["node_degree"]
    assert sig["best"] == "incident" and sig["worst"] == "adjacency"
    assert sig["omnibus"]["df"] == 2
    assert sig["gap"]["enc_a"] == "incident" and sig["gap"]["enc_b"] == "adjacency"
    assert sig["gap"]["p"] < 0.05
