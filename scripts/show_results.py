"""Inspect a run dir's results: the summary table + optional raw->parsed rows.

Shell-agnostic (no heredoc needed -- the cluster login shell is tcsh):

    python scripts/show_results.py results/p2-pilot-matrix
    python scripts/show_results.py results/p2-pilot-matrix --raw
    python scripts/show_results.py results/p2-pilot-matrix --raw --wrong-only
    python scripts/show_results.py results/p2-pilot-matrix --raw \
        --task connected_nodes --encoding friendship
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gedebate.eval import report, results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--fragility", action="store_true",
                    help="also show per-task cross-encoding spread (mean/std/max-min)")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="write summary.csv + fragility.csv here (e.g. analysis/baseline)")
    ap.add_argument("--raw", action="store_true", help="print per-instance raw_output -> parsed")
    ap.add_argument("--wrong-only", action="store_true", help="with --raw, only incorrect rows")
    ap.add_argument("--task", default=None)
    ap.add_argument("--encoding", default=None)
    args = ap.parse_args()

    rows = [r for f in results.result_files(args.run_dir) for r in results.read_rows(f)]
    print(f"{len(rows)} rows in {args.run_dir}\n")
    summary = report.summarize(rows)
    frag = report.fragility(summary)
    print(report.format_summary(summary))

    if args.fragility:
        print("\n-- fragility (accuracy spread across encodings, per task) --")
        print(report.format_fragility(frag))

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.csv").write_text(report.summary_to_csv(summary), encoding="utf-8")
        (out / "fragility.csv").write_text(report.fragility_to_csv(frag), encoding="utf-8")
        print(f"\nsaved summary.csv + fragility.csv -> {out}/")

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
