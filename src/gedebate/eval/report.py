"""Summarize persisted results: per (task, encoding) accuracy, parse rate, tokens.

The primary metric is exact-match accuracy (see docs/notes.md). `parse_ok_rate` is
reported alongside it deliberately -- a low rate means "wrong" answers are really
parse failures, a confound to catch rather than a result. Token totals feed the
matched-compute comparison in later phases.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import pstdev


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


def fragility(summary: dict[tuple[str, str], dict]) -> dict[str, dict]:
    """Per-task cross-encoding spread -- the proposal's secondary metric.

    For each task, over its per-encoding accuracies: `mean`, population `std`, and
    `max_min` (max minus min = the headline fragility gap), plus which encoding is
    best/worst. This is what tells us whether encoding-fragility reproduces.
    """
    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    for (task, encoding), s in summary.items():
        by_task[task][encoding] = s["accuracy"]

    out: dict[str, dict] = {}
    for task in sorted(by_task):
        accs = by_task[task]
        vals = list(accs.values())
        best = max(accs, key=accs.get)
        worst = min(accs, key=accs.get)
        out[task] = {
            "per_encoding": dict(sorted(accs.items())),
            "mean": sum(vals) / len(vals),
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "max_min": max(vals) - min(vals),
            "best": best,
            "worst": worst,
        }
    return out


def format_fragility(frag: dict[str, dict]) -> str:
    """Fixed-width per-task fragility table: mean, std, max-min gap, best/worst."""
    header = f"{'task':<16}{'mean':>7}{'std':>7}{'max-min':>9}  {'best':>10} {'worst':>10}"
    lines = [header, "-" * len(header)]
    for task, f in frag.items():
        lines.append(
            f"{task:<16}{f['mean']:>7.3f}{f['std']:>7.3f}{f['max_min']:>9.3f}  "
            f"{f['best']:>10} {f['worst']:>10}"
        )
    return "\n".join(lines)
