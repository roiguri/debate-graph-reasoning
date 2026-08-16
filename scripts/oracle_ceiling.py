"""Oracle stopping ceiling: the best the debate loop could do with perfect foresight.

Every counterfactual stopping rule in `eval.diagnostics` picks a turn by some computable
signal. This asks the ceiling question instead: if a rule could see the ground truth and
stop at whichever Proposer turn happened to be right, how much would that be worth? It is
an upper bound on *any* stopping rule over the transcripts that actually occurred -- it
cannot say what a different revision prompt or a stronger Critic would have produced,
only that no re-reading of these turns beats it.

The replay is exact rather than simulated: each Proposer turn stores its own parsed
answer, so scoring turn `k` is scoring the answer that stood at turn `k`.

    python scripts/oracle_ceiling.py results/v2-main results/v2-seed11 results/v2-seed13

Prints per cell: turn-1 accuracy, the accuracy the run actually got, the oracle ceiling,
and the headroom between the last two.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from gedebate.eval import results
from gedebate.eval.scoring import score


def _debate_rows(run_dirs: list[str]) -> dict[str, dict]:
    """instance_id -> debate row."""
    rows: dict[str, dict] = {}
    for run_dir in run_dirs:
        for f in results.result_files(run_dir):
            for r in results.read_rows(f):
                if r["condition"] != "debate":
                    continue
                rows[r["instance_id"]] = r
    return rows


def ceiling(run_dirs: list[str]) -> dict[tuple[str, str], dict]:
    rows = _debate_rows(run_dirs)
    cells: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "turn1": 0, "actual": 0, "oracle": 0, "proposer_turns": 0})

    for run_dir in run_dirs:
        for f in results.trace_files(run_dir):
            for trace in results.read_traces(f):
                row = rows.get(trace["instance_id"])
                if row is None:  # no matching debate row, or a torn line
                    continue
                proposer = [t for t in trace["turns"] if t["role"] == "proposer"]
                if not proposer:
                    continue
                gt = row["ground_truth"]
                c = cells[(row["task"], row["encoding"])]
                c["n"] += 1
                c["proposer_turns"] += len(proposer)
                c["turn1"] += bool(score(proposer[0].get("parsed"), gt))
                c["oracle"] += any(score(t.get("parsed"), gt) for t in proposer)
                c["actual"] += bool(row["correct"])
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="+", help="one or more debate run dirs; pooled")
    args = ap.parse_args()

    cells = ceiling(args.run_dir)
    if not cells:
        raise SystemExit("no debate traces found in " + ", ".join(args.run_dir))

    print(f"{'task/encoding':30s} {'n':>5s} {'turn1':>7s} {'actual':>7s} "
          f"{'oracle':>7s} {'headroom':>9s} {'turns/i':>8s}")
    print("-" * 78)
    total = defaultdict(int)
    for cell in sorted(cells):
        c = cells[cell]
        n = c["n"]
        print(f"{cell[0] + '/' + cell[1]:30s} {n:5d} {c['turn1']/n:7.3f} "
              f"{c['actual']/n:7.3f} {c['oracle']/n:7.3f} "
              f"{(c['oracle'] - c['actual'])/n:+9.3f} {c['proposer_turns']/n:8.2f}")
        for k, v in c.items():
            total[k] += v
    n = total["n"]
    print("-" * 78)
    print(f"{'POOLED':30s} {n:5d} {total['turn1']/n:7.3f} {total['actual']/n:7.3f} "
          f"{total['oracle']/n:7.3f} {(total['oracle'] - total['actual'])/n:+9.3f} "
          f"{total['proposer_turns']/n:8.2f}")


if __name__ == "__main__":
    main()
