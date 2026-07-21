"""Tests for `gedebate.eval.report` -- per (task, encoding) summary."""

from __future__ import annotations

from gedebate.eval.report import format_summary, summarize


def _row(task, enc, correct, parse_ok=True, gen=5):
    return {"task": task, "encoding": enc, "correct": correct,
            "parse_ok": parse_ok, "n_gen_tokens": gen}


def test_summarize_accuracy_parse_and_tokens():
    rows = [
        _row("edge_existence", "adjacency", True, gen=4),
        _row("edge_existence", "adjacency", False, parse_ok=False, gen=6),
        _row("edge_existence", "adjacency", True, gen=5),
        _row("node_degree", "incident", False, gen=10),
    ]
    s = summarize(rows)
    ee = s[("edge_existence", "adjacency")]
    assert ee["n"] == 3
    assert abs(ee["accuracy"] - 2 / 3) < 1e-9
    assert abs(ee["parse_ok_rate"] - 2 / 3) < 1e-9
    assert ee["total_gen_tokens"] == 15
    nd = s[("node_degree", "incident")]
    assert nd["n"] == 1 and nd["accuracy"] == 0.0


def test_format_summary_lists_each_group():
    s = summarize([_row("edge_existence", "adjacency", True)])
    out = format_summary(s)
    assert "edge_existence" in out and "adjacency" in out
    assert "acc" in out  # header present
