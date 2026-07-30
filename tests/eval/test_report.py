"""Tests for `gedebate.eval.report` -- per (task, encoding) summary + fragility."""

from __future__ import annotations

from gedebate.eval.report import (
    format_fragility,
    format_summary,
    fragility,
    summarize,
)


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


def test_summarize_total_tokens_and_per_instance():
    # total_tokens = prompt + generated; tables report the per-instance mean.
    rows = [
        {"task": "edge_existence", "encoding": "adjacency", "correct": True,
         "parse_ok": True, "n_gen_tokens": 2, "n_prompt_tokens": 100},
        {"task": "edge_existence", "encoding": "adjacency", "correct": True,
         "parse_ok": True, "n_gen_tokens": 4, "n_prompt_tokens": 100},
    ]
    s = summarize(rows)[("edge_existence", "adjacency")]
    assert s["total_gen_tokens"] == 6           # generated only (reference)
    assert s["total_tokens"] == 206             # (100+2) + (100+4)
    assert s["n_responses"] == 2 and s["responses_per_instance"] == 1.0
    assert s["tokens_per_instance"] == 103.0    # 206 / 2 instances


def test_format_summary_lists_each_group():
    s = summarize([_row("edge_existence", "adjacency", True)])
    out = format_summary(s)
    assert "edge_existence" in out and "adjacency" in out
    assert "acc" in out  # header present


# --- fragility ----------------------------------------------------------------

def _summary_from(acc_by_cell):
    """Build a summarize()-shaped dict from {(task,enc): accuracy}."""
    return {k: {"n": 10, "accuracy": a, "parse_ok_rate": 1.0, "total_gen_tokens": 0}
            for k, a in acc_by_cell.items()}


def test_fragility_spread_stats():
    summary = _summary_from({
        ("connected_nodes", "adjacency"): 0.2,
        ("connected_nodes", "incident"): 0.5,
        ("connected_nodes", "friendship"): 0.1,
    })
    f = fragility(summary)["connected_nodes"]
    assert f["best"] == "incident" and f["worst"] == "friendship"
    assert abs(f["max_min"] - 0.4) < 1e-9
    assert abs(f["mean"] - (0.2 + 0.5 + 0.1) / 3) < 1e-9
    assert f["std"] > 0.0
    assert set(f["per_encoding"]) == {"adjacency", "incident", "friendship"}


def test_fragility_no_spread_is_zero():
    summary = _summary_from({
        ("edge_existence", "adjacency"): 0.8,
        ("edge_existence", "incident"): 0.8,
        ("edge_existence", "friendship"): 0.8,
    })
    f = fragility(summary)["edge_existence"]
    assert f["max_min"] == 0.0 and f["std"] == 0.0


def test_format_fragility_table():
    summary = _summary_from({
        ("node_degree", "adjacency"): 0.375,
        ("node_degree", "incident"): 0.75,
        ("node_degree", "friendship"): 0.75,
    })
    out = format_fragility(fragility(summary))
    assert "node_degree" in out and "max-min" in out
