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


# --- debate views (glue: reuse report.summarize/fragility/stats; debate is 1 voted row
# per instance, like baseline, so only its compute column differs -- it makes several calls) ---

def _apply_debate_responses(summary: dict, rows: list[dict]) -> None:
    """summarize() reports 1 response/instance (the baseline case); debate makes several,
    so overwrite its compute fields with the stored n_responses (mean per cell)."""
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        by_cell[(r["task"], r["encoding"])].append(int(r.get("n_responses", 1)))
    for key, s in summary.items():
        resp = by_cell.get(key, [1])
        s["n_responses"] = sum(resp)
        s["responses_per_instance"] = sum(resp) / len(resp)


def _bc_debate_baseline(base_rows: list[dict], debate_rows: list[dict]) -> dict:
    """Per-cell McNemar b/c pairing baseline vs debate on the same instance (both are 1
    row/instance). b = baseline right / debate wrong; c = debate right / baseline wrong."""
    base_ok = {r["instance_id"]: bool(r["correct"]) for r in base_rows}
    out: dict[tuple[str, str], dict] = defaultdict(lambda: {"b": 0, "c": 0})
    for r in debate_rows:
        if r["instance_id"] not in base_ok:
            continue
        dok, bok = bool(r["correct"]), base_ok[r["instance_id"]]
        cell = out[(r["task"], r["encoding"])]
        if bok and not dok:
            cell["b"] += 1
        elif dok and not bok:
            cell["c"] += 1
    return out


def _compare_debate(other_summary: dict, other_acc_key: str, deb_summary: dict, bc: dict) -> dict:
    """Per-cell debate-vs-other rows: accuracy delta, response/token multipliers (debate
    over other), and paired McNemar (b/c already oriented as c = debate-only right)."""
    out: dict[tuple[str, str], dict] = {}
    for key in sorted(set(other_summary) & set(deb_summary)):
        o, d = other_summary[key], deb_summary[key]
        mc = stats.mcnemar_from_bc(bc.get(key, {"b": 0})["b"], bc.get(key, {"c": 0})["c"])
        out[key] = {
            "a_acc": o[other_acc_key], "d_acc": d["accuracy"], "delta": d["accuracy"] - o[other_acc_key],
            "xresp": d["responses_per_instance"] / o["responses_per_instance"] if o["responses_per_instance"] else float("nan"),
            "xtok": d["tokens_per_instance"] / o["tokens_per_instance"] if o["tokens_per_instance"] else float("nan"),
            "b": mc["b"], "c": mc["c"], "p": mc["p"],
        }
    return out


def _format_debate_compare(cmp: dict, a_label: str) -> str:
    header = (f"{'task':<16}{'encoding':<11}{a_label:>9}{'debate':>8}{'delta':>8}"
              f"{'xresp':>6}{'xtok':>6}{'b/c':>9}{'McNemar p':>12}")
    lines = [header, "-" * len(header)]
    for (task, enc), s in cmp.items():
        bc = f"{s['b']}/{s['c']}"
        lines.append(
            f"{task:<16}{enc:<11}{s['a_acc']:>9.3f}{s['d_acc']:>8.3f}{s['delta']:>+8.3f}"
            f"{s['xresp']:>6.1f}{s['xtok']:>6.1f}{bc:>9}{s['p']:>10.3f} {report._stars(s['p']):<3}"
        )
    return "\n".join(lines)


def _compare_to_csv(cmp: dict, a_name: str) -> str:
    lines = [f"task,encoding,{a_name}_acc,debate_acc,delta,resp_mult,token_mult,mcnemar_b,mcnemar_c,mcnemar_p"]
    for (task, enc), s in cmp.items():
        lines.append(
            f"{task},{enc},{s['a_acc']:.4f},{s['d_acc']:.4f},{s['delta']:+.4f},"
            f"{s['xresp']:.2f},{s['xtok']:.2f},{s['b']},{s['c']},{s['p']:.4g}"
        )
    return "\n".join(lines) + "\n"


def _select_prompt_version(debate_rows: list[dict], requested: str | None) -> list[dict]:
    """Keep one Proposer wording's rows; refuse to silently pool two.

    Applies to every condition that prompts with the Proposer wording (debate and
    `majority_vote_cot`). Different versions live in different run dirs but share a
    condition name, so pooling them would average two experiments into one accuracy.
    Mixed input is an error, not a warning.
    """
    present = sorted({results.row_prompt_version(r) for r in debate_rows})
    if requested is None:
        if len(present) > 1:
            raise SystemExit(
                f"run dirs mix debate prompt versions {present}; these are different "
                f"experiments and must not be pooled. Pass --prompt-version."
            )
        return debate_rows
    return [r for r in debate_rows if results.row_prompt_version(r) == requested]


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
    ap.add_argument("--prompt-version", default=None,
                    help="which Proposer wording's debate rows to include (e.g. v1, v2); "
                         "required when the run dirs contain more than one")
    ap.add_argument("--condition", default=None,
                    help="comma-separated conditions to include (default: all). "
                         "e.g. --condition baseline for a baseline-only pooled view")
    ap.add_argument("--vote-condition", default="majority_vote", choices=results.VOTE_CONDITIONS,
                    help="which vote arm the majority-vote views report: the terse "
                         "baseline-prompt one (default) or the reasoned Proposer-prompt "
                         "one. They answer different questions and are never pooled.")
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
    mv_rows = [r for r in rows if r["condition"] == args.vote_condition]
    if args.vote_condition == "majority_vote_cot":
        # This arm samples the Proposer prompt, so its rows carry a prompt_version and
        # two wordings must no more be pooled here than they may be for debate.
        mv_rows = _select_prompt_version(mv_rows, args.prompt_version)
    # Tag saved filenames with the arm, so the two never overwrite each other in one dir.
    mv_tag = "mv" if args.vote_condition == "majority_vote" else "mvcot"

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
        print(f"\n-- {args.vote_condition} (voted per instance; 1samp = mean single-draw acc) --")
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
            print(f"\n-- --compare needs both baseline and {args.vote_condition} rows in the run dir --")

    # -- debate (verify-and-revise): reuse the same summarize/fragility/significance path as
    # baseline; select by name so it never pools into the baseline table.
    debate_rows = _select_prompt_version(
        [r for r in rows if r["condition"] == "debate"], args.prompt_version)
    deb_summary = report.summarize(debate_rows) if debate_rows else {}
    deb_vs_base: dict = {}
    deb_vs_mv: dict = {}
    if debate_rows:
        _apply_debate_responses(deb_summary, debate_rows)
        print("\n-- debate (verify-and-revise, per instance) --")
        print(report.format_summary(deb_summary))
        if args.fragility:
            print("\n-- debate fragility (accuracy spread across encodings, per task) --")
            print(report.format_fragility(report.fragility(deb_summary)))
            print("\n-- debate significance (paired: encodings share graphs) --")
            print(report.format_significance(stats.task_significance(debate_rows)))
        if base_rows:
            deb_vs_base = _compare_debate(report.summarize(base_rows), "accuracy",
                                          deb_summary, _bc_debate_baseline(base_rows, debate_rows))
        if mv_rows:
            disc = report._vote_vs_baseline_discordance(debate_rows, mv_rows)  # base=debate
            bc_mv = {k: {"b": v["c"], "c": v["b"]} for k, v in disc.items()}   # orient c = debate-only right
            deb_vs_mv = _compare_debate(report.summarize_votes(mv_rows), "voted_accuracy", deb_summary, bc_mv)

    if args.compare and deb_vs_base:
        print("\n-- debate vs baseline (delta = debate - baseline; b/c = base-only / debate-only right) --")
        print(_format_debate_compare(deb_vs_base, "base_acc"))
    if args.compare and deb_vs_mv:
        print("\n-- debate vs majority vote (delta = debate - vote; xresp<1 = debate is cheaper) --")
        print(_format_debate_compare(deb_vs_mv, "vote_acc"))

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
            (out / f"{mv_tag}_vote_summary.csv").write_text(
                report.vote_summary_to_csv(vote_summary), encoding="utf-8")
            wrote.append(f"{mv_tag}_vote_summary.csv")
        if comparison:
            (out / f"{mv_tag}_vs_baseline.csv").write_text(
                report.comparison_to_csv(comparison), encoding="utf-8")
            wrote.append(f"{mv_tag}_vs_baseline.csv")
        if debate_rows:
            (out / "debate_summary.csv").write_text(report.summary_to_csv(deb_summary), encoding="utf-8")
            (out / "debate_fragility.csv").write_text(
                report.fragility_to_csv(report.fragility(deb_summary)), encoding="utf-8")
            (out / "debate_significance.csv").write_text(
                report.significance_to_csv(stats.task_significance(debate_rows)), encoding="utf-8")
            wrote += ["debate_summary.csv", "debate_fragility.csv", "debate_significance.csv"]
        if deb_vs_base:
            (out / "debate_vs_baseline.csv").write_text(_compare_to_csv(deb_vs_base, "baseline"), encoding="utf-8")
            wrote.append("debate_vs_baseline.csv")
        if deb_vs_mv:
            (out / f"debate_vs_{mv_tag}.csv").write_text(_compare_to_csv(deb_vs_mv, "vote"), encoding="utf-8")
            wrote.append(f"debate_vs_{mv_tag}.csv")
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
