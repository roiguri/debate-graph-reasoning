"""Summarize persisted results: per (task, encoding) accuracy, parse rate, tokens.

The primary metric is exact-match accuracy (see docs/notes.md). `parse_ok_rate` is
reported alongside it deliberately -- a low rate means "wrong" answers are really
parse failures, a confound to catch rather than a result. Token totals feed the
matched-compute comparison in later phases.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import pstdev

from gedebate.eval.stats import wilson_ci


def summarize(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Group rows by (task, encoding) and compute accuracy / parse rate / tokens."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["encoding"])].append(r)

    out: dict[tuple[str, str], dict] = {}
    for key in sorted(groups):
        rs = groups[key]
        n = len(rs)
        n_correct = sum(bool(r["correct"]) for r in rs)
        out[key] = {
            "n": n,
            "accuracy": n_correct / n,
            "acc_ci": wilson_ci(n_correct, n),  # 95% Wilson interval (lo, hi)
            "parse_ok_rate": sum(bool(r["parse_ok"]) for r in rs) / n,
            "total_gen_tokens": sum(int(r["n_gen_tokens"]) for r in rs),
        }
    return out


def format_summary(summary: dict[tuple[str, str], dict]) -> str:
    """A compact fixed-width table for stdout / job logs."""
    header = (f"{'task':<16}{'encoding':<12}{'n':>4}{'acc':>8}"
              f"{'95% CI':>16}{'parse_ok':>10}{'gen_tok':>10}")
    lines = [header, "-" * len(header)]
    for (task, encoding), s in summary.items():
        lo, hi = s["acc_ci"]
        ci = f"[{lo:.3f},{hi:.3f}]"
        lines.append(
            f"{task:<16}{encoding:<12}{s['n']:>4}"
            f"{s['accuracy']:>8.3f}{ci:>16}{s['parse_ok_rate']:>10.3f}"
            f"{s['total_gen_tokens']:>10}"
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


def summary_to_csv(summary: dict[tuple[str, str], dict]) -> str:
    """Per-cell summary as CSV text (for a committed analysis/ artifact)."""
    lines = ["task,encoding,n,accuracy,ci_lo,ci_hi,parse_ok_rate,total_gen_tokens"]
    for (task, encoding), s in summary.items():
        lo, hi = s["acc_ci"]
        lines.append(
            f"{task},{encoding},{s['n']},{s['accuracy']:.4f},{lo:.4f},{hi:.4f},"
            f"{s['parse_ok_rate']:.4f},{s['total_gen_tokens']}"
        )
    return "\n".join(lines) + "\n"


def fragility_to_csv(frag: dict[str, dict]) -> str:
    """Per-task fragility as CSV text (mean/std/max-min + best/worst encoding)."""
    lines = ["task,mean,std,max_min,best,worst"]
    for task, f in frag.items():
        lines.append(
            f"{task},{f['mean']:.4f},{f['std']:.4f},{f['max_min']:.4f},"
            f"{f['best']},{f['worst']}"
        )
    return "\n".join(lines) + "\n"


def _stars(p: float) -> str:
    """Conventional significance markers for a p-value."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def format_significance(sig: dict[str, dict]) -> str:
    """Per-task paired-test table: omnibus Cochran's Q + best-vs-worst McNemar gap.

    `sig` is `stats.task_significance(rows)`. Because encodings are applied to the
    same graphs, these paired tests -- not the cross-encoding std -- are what says
    the fragility is real rather than sampling noise.
    """
    header = (f"{'task':<16}{'Q(df=2)':>9}{'p':>11}{'':>4}   "
              f"best>worst gap (McNemar){'':>3}{'p':>11}{'':>4}")
    lines = [header, "-" * len(header)]
    for task, s in sig.items():
        o, g = s["omnibus"], s["gap"]
        gap = f"{s['best']}>{s['worst']}  b/c={g['b']}/{g['c']}"
        lines.append(
            f"{task:<16}{o['q']:>9.2f}{o['p']:>11.2e} {_stars(o['p']):<3}   "
            f"{gap:<28}{g['p']:>11.2e} {_stars(g['p']):<3}"
        )
    return "\n".join(lines)


def significance_to_csv(sig: dict[str, dict]) -> str:
    """Per-task significance as CSV (omnibus + best-vs-worst gap)."""
    lines = ["task,best,worst,cochran_q,cochran_df,cochran_p,"
             "mcnemar_b,mcnemar_c,mcnemar_p"]
    for task, s in sig.items():
        o, g = s["omnibus"], s["gap"]
        lines.append(
            f"{task},{s['best']},{s['worst']},{o['q']:.4f},{o['df']},{o['p']:.3e},"
            f"{g['b']},{g['c']},{g['p']:.3e}"
        )
    return "\n".join(lines) + "\n"


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
