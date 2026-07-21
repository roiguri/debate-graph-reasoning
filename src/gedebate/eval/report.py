"""Summarize persisted results: per (task, encoding) accuracy, parse rate, tokens.

The primary metric is exact-match accuracy (see docs/notes.md). `parse_ok_rate` is
reported alongside it deliberately -- a low rate means "wrong" answers are really
parse failures, a confound to catch rather than a result. Token totals feed the
matched-compute comparison in later phases.
"""

from __future__ import annotations

from collections import defaultdict


def summarize(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Group rows by (task, encoding) and compute accuracy / parse rate / tokens."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["encoding"])].append(r)

    out: dict[tuple[str, str], dict] = {}
    for key in sorted(groups):
        rs = groups[key]
        n = len(rs)
        out[key] = {
            "n": n,
            "accuracy": sum(bool(r["correct"]) for r in rs) / n,
            "parse_ok_rate": sum(bool(r["parse_ok"]) for r in rs) / n,
            "total_gen_tokens": sum(int(r["n_gen_tokens"]) for r in rs),
        }
    return out


def format_summary(summary: dict[tuple[str, str], dict]) -> str:
    """A compact fixed-width table for stdout / job logs."""
    header = f"{'task':<16}{'encoding':<12}{'n':>4}{'acc':>8}{'parse_ok':>10}{'gen_tok':>10}"
    lines = [header, "-" * len(header)]
    for (task, encoding), s in summary.items():
        lines.append(
            f"{task:<16}{encoding:<12}{s['n']:>4}"
            f"{s['accuracy']:>8.3f}{s['parse_ok_rate']:>10.3f}{s['total_gen_tokens']:>10}"
        )
    return "\n".join(lines)
