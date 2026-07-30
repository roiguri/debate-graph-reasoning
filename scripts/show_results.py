"""Inspect run dir(s): summary + significance + optional raw->parsed rows.

Shell-agnostic (no heredoc needed -- the cluster login shell is tcsh):

    python scripts/show_results.py results/main
    python scripts/show_results.py results/main --fragility
    python scripts/show_results.py results/main --compare        # vote vs baseline delta
    python scripts/show_results.py results/main --compare --save analysis/mv
    python scripts/show_results.py results/main --raw --wrong-only
    python scripts/show_results.py results/main --raw \
        --task connected_nodes --encoding friendship

Pool independent seeds for a replication check by passing more than one run dir.
The pooled tables answer "is the effect real"; `--by-seed` also prints each seed's
own fragility table so you can see the *pattern* repeat on independent graphs:

    python scripts/show_results.py results/main results/seed11 results/seed13 \
        --fragility --by-seed
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from gedebate.eval import report, results, stats


def _seed_of(row: dict) -> str:
    """The dataset seed a row came from (first field of instance_id)."""
    return row["instance_id"].split("/")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="+",
                    help="one or more run dirs; multiple are pooled (replication)")
    ap.add_argument("--fragility", action="store_true",
                    help="also show per-task cross-encoding spread + paired significance")
    ap.add_argument("--by-seed", action="store_true",
                    help="with >1 seed present, also print each seed's own fragility table")
    ap.add_argument("--compare", action="store_true",
                    help="with both baseline + majority_vote present, print the "
                         "vote-vs-baseline delta table (accuracy delta + token cost)")
    ap.add_argument("--condition", default=None,
                    help="comma-separated conditions to include (default: all). "
                         "e.g. --condition baseline for a baseline-only pooled view")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="write summary/fragility/significance CSVs here (e.g. analysis/baseline)")
    ap.add_argument("--raw", action="store_true", help="print per-instance raw_output -> parsed")
    ap.add_argument("--wrong-only", action="store_true", help="with --raw, only incorrect rows")
    ap.add_argument("--task", default=None)
    ap.add_argument("--encoding", default=None)
    args = ap.parse_args()

    rows = [r for d in args.run_dir for f in results.result_files(d) for r in results.read_rows(f)]
    if args.condition:
        keep = {c.strip() for c in args.condition.split(",")}
        rows = [r for r in rows if r["condition"] in keep]
    seeds = sorted({_seed_of(r) for r in rows})
    where = ", ".join(args.run_dir)
    print(f"{len(rows)} rows in {where}  (seeds: {', '.join(seeds) or 'none'})\n")

    # A shared out_dir holds baseline/ and majority_vote/ side by side; summarize each
    # separately (per-draw accuracy for baseline, voted-per-instance for majority-vote)
    # rather than pooling them into one number. Select baseline by name, not by
    # "not majority_vote": once P5's debate/ rows share the dir, an inequality would
    # silently pool them into the baseline table.
    base_rows = [r for r in rows if r["condition"] == "baseline"]
    mv_rows = [r for r in rows if r["condition"] == "majority_vote"]

    summary = report.summarize(base_rows) if base_rows else {}
    frag = report.fragility(summary) if base_rows else {}
    sig = stats.task_significance(base_rows) if base_rows else {}
    if base_rows:
        if mv_rows:
            print("-- baseline (greedy, per instance) --")
        print(report.format_summary(summary))

    if base_rows and args.fragility:
        pooled = "pooled, all seeds" if len(seeds) > 1 else "single seed"
        print(f"\n-- fragility (accuracy spread across encodings, per task; {pooled}) --")
        print(report.format_fragility(frag))
        print("\n-- significance (paired: encodings share the same graphs) --")
        print(report.format_significance(sig))
        if args.by_seed and len(seeds) > 1:
            by_seed: dict[str, list[dict]] = defaultdict(list)
            for r in base_rows:
                by_seed[_seed_of(r)].append(r)
            for seed in seeds:
                print(f"\n-- seed {seed} only (replication view) --")
                print(report.format_fragility(report.fragility(report.summarize(by_seed[seed]))))

    vote_summary = report.summarize_votes(mv_rows) if mv_rows else {}
    if mv_rows:
        print("\n-- majority vote (voted per instance; 1samp = mean single-draw acc) --")
        print(report.format_vote_summary(vote_summary))

    # vote-vs-baseline comparison (needs both conditions): printed on --compare,
    # and always saved when --save is on, since it's the P4 headline artifact.
    comparison = report.compare_baseline_vote(base_rows, mv_rows) if (base_rows and mv_rows) else {}
    if args.compare:
        if comparison:
            print("\n-- majority vote vs baseline "
                  "(delta = vote_acc - baseline_acc; x = token multiplier) --")
            print(report.format_comparison(comparison))
        else:
            print("\n-- --compare needs both baseline and majority_vote rows in the run dir --")

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        # Condition-tagged filenames so a single --save into a scope dir is
        # unambiguous (baseline views never collide with a generic name).
        wrote = []
        if base_rows:
            (out / "baseline_summary.csv").write_text(report.summary_to_csv(summary), encoding="utf-8")
            (out / "baseline_fragility.csv").write_text(report.fragility_to_csv(frag), encoding="utf-8")
            (out / "baseline_significance.csv").write_text(report.significance_to_csv(sig), encoding="utf-8")
            wrote += ["baseline_summary.csv", "baseline_fragility.csv", "baseline_significance.csv"]
        if mv_rows:
            (out / "mv_vote_summary.csv").write_text(
                report.vote_summary_to_csv(vote_summary), encoding="utf-8")
            wrote.append("mv_vote_summary.csv")
        if comparison:
            (out / "mv_vs_baseline.csv").write_text(
                report.comparison_to_csv(comparison), encoding="utf-8")
            wrote.append("mv_vs_baseline.csv")
        print(f"\nsaved {' + '.join(wrote)} -> {out}/")

    if not args.raw:
        return
    print("\n-- per instance (raw model output -> parsed answer) --")
    for r in rows:
        if args.task and r["task"] != args.task:
            continue
        if args.encoding and r["encoding"] != args.encoding:
            continue
        if args.wrong_only and r["correct"]:
            continue
        print(
            f"{r['task']:<16}{r['encoding']:<11} correct={str(r['correct']):<5} "
            f"raw={r['raw_output']!r} -> {r['parsed_answer']}  (gt={r['ground_truth']})"
        )


if __name__ == "__main__":
    main()
