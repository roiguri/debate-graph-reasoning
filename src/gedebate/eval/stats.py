"""Significance for the fragility claim, from the *paired* structure of the data.

The three encodings are applied to the **same** graphs (one `graph_index` per
graph, shared query), so per-graph correctness across encodings is matched, not
independent. The honest test of "does encoding matter for this task?" is therefore
a paired one:

- **Cochran's Q** -- omnibus across all encodings of a task (any encoding differ?).
- **McNemar** -- pairwise, used for the headline best-vs-worst gap.

Both live at closed-form p-values here (Q has df = k-1 = 2 for our 3 encodings;
McNemar has df = 1), so this module is pure stdlib -- no scipy, importable on the
CPU-only analysis path. `wilson_ci` gives each cell's accuracy an interval for the
summary table.

Rows are the persisted attempt rows (one per instance for the baseline); each is
keyed to its graph via `instance_id` = "seed/graph_index/task/encoding".
"""

from __future__ import annotations

import math
from collections import defaultdict

Z_95 = 1.959963984540054  # standard normal two-sided 95%


def _norm_sf(x: float) -> float:
    """Upper tail of the standard normal, P(Z > x)."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def graph_index(instance_id: str) -> str:
    """Graph key that pairs encodings/seeds: the graph_index field of instance_id."""
    return instance_id.split("/")[1]


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (n==0 -> (0, 1))."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _correct_by_graph(rows: list[dict]) -> dict[str, dict[str, bool]]:
    """{graph_index: {encoding: correct}} for one task's rows."""
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in rows:
        out[graph_index(r["instance_id"])][r["encoding"]] = bool(r["correct"])
    return out


def mcnemar(rows: list[dict], enc_a: str, enc_b: str) -> dict:
    """Paired McNemar test on two encodings' per-graph correctness (task rows).

    Only graphs scored under *both* encodings count. `b` = right on A, wrong on B;
    `c` = wrong on A, right on B. Uses the continuity-corrected chi-square (df=1);
    for tiny discordant counts (b+c < 25) reports the exact binomial p instead.
    """
    by_graph = _correct_by_graph(rows)
    b = c = 0
    for encs in by_graph.values():
        if enc_a not in encs or enc_b not in encs:
            continue
        a_ok, b_ok = encs[enc_a], encs[enc_b]
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1
    n = b + c
    if n == 0:
        return {"enc_a": enc_a, "enc_b": enc_b, "b": b, "c": c, "stat": 0.0, "p": 1.0}
    if n < 25:  # exact two-sided binomial (p=0.5), small-sample regime
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
        return {"enc_a": enc_a, "enc_b": enc_b, "b": b, "c": c, "stat": float(k), "p": p}
    chi = (abs(b - c) - 1) ** 2 / n
    return {"enc_a": enc_a, "enc_b": enc_b, "b": b, "c": c,
            "stat": chi, "p": 2 * _norm_sf(math.sqrt(chi))}


def cochran_q(rows: list[dict], encodings: list[str]) -> dict:
    """Cochran's Q omnibus across encodings, on graphs scored under *all* of them.

    Q ~ chi-square(df = k-1). For our k=3 that is df=2, whose survival has the
    closed form exp(-Q/2); df=1 reduces to the normal tail. Other df raise (we do
    not ship a general incomplete-gamma).
    """
    by_graph = _correct_by_graph(rows)
    encs = list(encodings)
    k = len(encs)
    complete = [g for g, e in by_graph.items() if all(x in e for x in encs)]
    n = len(complete)
    col = [sum(by_graph[g][e] for g in complete) for e in encs]
    row = [sum(by_graph[g][e] for e in encs) for g in complete]
    total = sum(col)
    denom = k * total - sum(r * r for r in row)
    df = k - 1
    if denom == 0:  # no discordance (all-right or all-wrong graphs) -> no evidence
        return {"q": 0.0, "df": df, "p": 1.0, "n": n, "encodings": encs}
    q = (k - 1) * (k * sum(x * x for x in col) - total * total) / denom
    if df == 1:
        p = 2 * _norm_sf(math.sqrt(q))
    elif df == 2:
        p = math.exp(-q / 2)
    else:
        raise ValueError(f"cochran_q p-value supports df in (1, 2), got df={df}")
    return {"q": q, "df": df, "p": p, "n": n, "encodings": encs}


def task_significance(rows: list[dict]) -> dict[str, dict]:
    """Per-task paired significance: omnibus Q + best-vs-worst McNemar.

    Groups the given rows by task, ranks that task's encodings by accuracy to name
    best/worst, then reports Cochran's Q over all encodings and the McNemar gap
    between best and worst. Rows may pool multiple seeds -- pairing is by graph.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    out: dict[str, dict] = {}
    for task in sorted(by_task):
        trows = by_task[task]
        acc: dict[str, list[int]] = defaultdict(list)
        for r in trows:
            acc[r["encoding"]].append(bool(r["correct"]))
        rates = {e: sum(v) / len(v) for e, v in acc.items()}
        encs = sorted(rates)
        best = max(rates, key=rates.get)
        worst = min(rates, key=rates.get)
        out[task] = {
            "omnibus": cochran_q(trows, encs),
            "gap": mcnemar(trows, best, worst),
            "best": best,
            "worst": worst,
        }
    return out
