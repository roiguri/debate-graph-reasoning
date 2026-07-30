"""Summarize persisted results: per (task, encoding) accuracy, parse rate, tokens.

The primary metric is exact-match accuracy (see docs/notes.md). `parse_ok_rate` is
reported alongside it deliberately -- a low rate means "wrong" answers are really
parse failures, a confound to catch rather than a result. Token totals feed the
matched-compute comparison in later phases.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import pstdev

from gedebate.conditions.majority_vote import vote
from gedebate.eval.scoring import score
from gedebate.eval.stats import mcnemar_from_bc, wilson_ci


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


def summarize_votes(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Vote-aggregated per-cell summary for majority-vote rows.

    Group the N sample rows by instance, take the vote of their parsed answers, and
    score that voted answer -- so accuracy is per *instance*, not per draw. Reports
    `voted_accuracy` (with Wilson CI) next to `per_sample_accuracy` (the mean
    single-draw accuracy, i.e. what baseline-style `summarize` would show) and
    `total_gen_tokens` (the N x cost that makes the matched-compute comparison
    honest). Expects rows from a single (majority_vote) condition.
    """
    per_instance: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_instance[r["instance_id"]].append(r)

    cells: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n_instances": 0, "voted_correct": 0, "voted_parse_ok": 0,
                 "sample_rows": 0, "sample_correct": 0, "total_gen_tokens": 0,
                 "n_samples": set()}
    )
    for irows in per_instance.values():
        irows = sorted(irows, key=lambda r: r.get("sample_index", 0))
        head = irows[0]
        voted, parse_ok, _support = vote([r["parsed_answer"] for r in irows])
        c = cells[(head["task"], head["encoding"])]
        c["n_instances"] += 1
        c["voted_correct"] += bool(score(voted, head["ground_truth"]))
        c["voted_parse_ok"] += bool(parse_ok)
        c["sample_rows"] += len(irows)
        c["sample_correct"] += sum(bool(r["correct"]) for r in irows)
        c["total_gen_tokens"] += sum(int(r["n_gen_tokens"]) for r in irows)
        c["n_samples"].add(len(irows))

    out: dict[tuple[str, str], dict] = {}
    for key in sorted(cells):
        c = cells[key]
        n = c["n_instances"]
        out[key] = {
            "n": n,
            "voted_accuracy": c["voted_correct"] / n,
            "voted_acc_ci": wilson_ci(c["voted_correct"], n),
            "per_sample_accuracy": c["sample_correct"] / c["sample_rows"],
            "parse_ok_rate": c["voted_parse_ok"] / n,
            "total_gen_tokens": c["total_gen_tokens"],
            "n_samples": sorted(c["n_samples"]),  # normally a single value [N]
        }
    return out


def format_vote_summary(summary: dict[tuple[str, str], dict]) -> str:
    """Fixed-width per-cell majority-vote table: voted vs per-sample accuracy + tokens."""
    header = (f"{'task':<16}{'encoding':<12}{'n':>4}{'N':>4}{'vote_acc':>9}"
              f"{'95% CI':>16}{'1samp':>8}{'parse_ok':>10}{'gen_tok':>10}")
    lines = [header, "-" * len(header)]
    for (task, encoding), s in summary.items():
        lo, hi = s["voted_acc_ci"]
        ci = f"[{lo:.3f},{hi:.3f}]"
        ns = s["n_samples"][0] if len(s["n_samples"]) == 1 else "*"
        lines.append(
            f"{task:<16}{encoding:<12}{s['n']:>4}{str(ns):>4}"
            f"{s['voted_accuracy']:>9.3f}{ci:>16}{s['per_sample_accuracy']:>8.3f}"
            f"{s['parse_ok_rate']:>10.3f}{s['total_gen_tokens']:>10}"
        )
    return "\n".join(lines)


def vote_summary_to_csv(summary: dict[tuple[str, str], dict]) -> str:
    """Per-cell majority-vote summary as CSV text (for a committed analysis artifact)."""
    lines = ["task,encoding,n,n_samples,voted_accuracy,ci_lo,ci_hi,"
             "per_sample_accuracy,parse_ok_rate,total_gen_tokens"]
    for (task, encoding), s in summary.items():
        lo, hi = s["voted_acc_ci"]
        ns = s["n_samples"][0] if len(s["n_samples"]) == 1 else "mixed"
        lines.append(
            f"{task},{encoding},{s['n']},{ns},{s['voted_accuracy']:.4f},{lo:.4f},{hi:.4f},"
            f"{s['per_sample_accuracy']:.4f},{s['parse_ok_rate']:.4f},{s['total_gen_tokens']}"
        )
    return "\n".join(lines) + "\n"


def compare_baseline_vote(
    base_rows: list[dict], mv_rows: list[dict]
) -> dict[tuple[str, str], dict]:
    """Per-cell majority-vote vs greedy-baseline comparison: accuracy delta + token cost.

    Reuses `summarize` (per-draw/greedy) and `summarize_votes` (voted-per-instance) so
    the numbers come from the same pipeline as every other table, not a one-off. Only
    cells present in *both* conditions are compared. `delta` = voted − baseline
    accuracy; `token_mult` = the vote's generated-token cost over the baseline's.

    Adds a **paired McNemar** of vote vs baseline (they run on the same instances):
    `b` = baseline right / vote wrong, `c` = vote right / baseline wrong, over the
    instances shared by both conditions. `mcnemar_p` turns "the CIs overlap" into a
    real test -- a large p means voting is statistically indistinguishable from greedy.
    """
    bs = summarize(base_rows)
    vs = summarize_votes(mv_rows)
    disc = _vote_vs_baseline_discordance(base_rows, mv_rows)
    out: dict[tuple[str, str], dict] = {}
    for key in sorted(set(bs) & set(vs)):
        b, v = bs[key], vs[key]
        bt, vt = b["total_gen_tokens"], v["total_gen_tokens"]
        bc = disc.get(key, {"b": 0, "c": 0})
        mc = mcnemar_from_bc(bc["b"], bc["c"])
        out[key] = {
            "baseline_accuracy": b["accuracy"],
            "voted_accuracy": v["voted_accuracy"],
            "delta": v["voted_accuracy"] - b["accuracy"],
            "baseline_gen_tokens": bt,
            "vote_gen_tokens": vt,
            "token_mult": (vt / bt) if bt else float("nan"),
            "b": mc["b"],  # baseline right, vote wrong
            "c": mc["c"],  # vote right, baseline wrong
            "discordant": mc["b"] + mc["c"],
            "mcnemar_p": mc["p"],
        }
    return out


def _vote_vs_baseline_discordance(
    base_rows: list[dict], mv_rows: list[dict]
) -> dict[tuple[str, str], dict]:
    """Per-cell McNemar b/c pairing baseline vs the voted answer on the same instance.

    Vote-correctness is derived (vote the N draws, then score) -- it is not a stored
    field -- so this pairs each instance's baseline `correct` against its voted result.
    """
    base_correct = {r["instance_id"]: bool(r["correct"]) for r in base_rows}
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for r in mv_rows:
        by_instance[r["instance_id"]].append(r)

    disc: dict[tuple[str, str], dict] = defaultdict(lambda: {"b": 0, "c": 0})
    for iid, irows in by_instance.items():
        if iid not in base_correct:
            continue
        irows = sorted(irows, key=lambda r: r.get("sample_index", 0))
        head = irows[0]
        voted, *_ = vote([r["parsed_answer"] for r in irows])
        vote_ok = bool(score(voted, head["ground_truth"]))
        base_ok = base_correct[iid]
        cell = disc[(head["task"], head["encoding"])]
        if base_ok and not vote_ok:
            cell["b"] += 1
        elif vote_ok and not base_ok:
            cell["c"] += 1
    return disc


def format_comparison(cmp: dict[tuple[str, str], dict]) -> str:
    """Fixed-width vote-vs-baseline table: accuracy delta, token multiplier, and the
    paired McNemar (b/c discordance + p) per cell. `_stars` is defined above."""
    header = (f"{'task':<16}{'encoding':<12}{'base_acc':>9}{'vote_acc':>9}"
              f"{'delta':>8}{'x':>6}{'b/c':>8}{'McNemar p':>11}")
    lines = [header, "-" * len(header)]
    for (task, encoding), s in cmp.items():
        bc = f"{s['b']}/{s['c']}"
        lines.append(
            f"{task:<16}{encoding:<12}{s['baseline_accuracy']:>9.3f}"
            f"{s['voted_accuracy']:>9.3f}{s['delta']:>+8.3f}{s['token_mult']:>6.1f}"
            f"{bc:>8}{s['mcnemar_p']:>9.3f} {_stars(s['mcnemar_p']):<3}"
        )
    return "\n".join(lines)


def comparison_to_csv(cmp: dict[tuple[str, str], dict]) -> str:
    """Vote-vs-baseline comparison as CSV text (the P4 headline artifact)."""
    lines = ["task,encoding,baseline_acc,vote_acc,delta,baseline_gen_tok,vote_gen_tok,"
             "token_mult,mcnemar_b,mcnemar_c,discordant,mcnemar_p"]
    for (task, encoding), s in cmp.items():
        lines.append(
            f"{task},{encoding},{s['baseline_accuracy']:.4f},{s['voted_accuracy']:.4f},"
            f"{s['delta']:+.4f},{s['baseline_gen_tokens']},{s['vote_gen_tokens']},"
            f"{s['token_mult']:.2f},{s['b']},{s['c']},{s['discordant']},{s['mcnemar_p']:.4g}"
        )
    return "\n".join(lines) + "\n"


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
