"""Open up the debate loop: where does the answer actually go, and is the Critic real?

`show_results.py` reports whether debate beat baseline. It did not, and the tables here
say why, from the committed trace sidecars only (no GPU time). See
docs/plan/p5-followup-diagnosis.md for the findings these regenerate.

Shell-agnostic (the cluster login shell is tcsh):

    python scripts/debate_diagnostics.py results/main
    python scripts/debate_diagnostics.py results/main --save analysis/main
    python scripts/debate_diagnostics.py results/main results/seed11 results/seed13 \
        --save analysis/pooled

Pooling several run dirs is the replication view, exactly as in `show_results.py`.
The dataset is read for the Critic-grounding audit (it needs each graph's true edge
list); it is found from each run's manifest, or pass `--dataset` to override.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gedebate.data.store import load_dataset
from gedebate.eval import diagnostics as diag
from gedebate.eval import results

# Fallback generation cap when a run's manifest does not record one (pre-v2 manifests).
# Only affects the `truncated` flags, never accuracy.
DEFAULT_MAX_NEW_TOKENS = 256


def _max_new_tokens(run_dirs: list[str]) -> int | None:
    """The debate generation cap recorded by the runs, or None if they disagree.

    Disagreeing caps make a pooled truncation count meaningless (a turn truncated
    under a 128-cap is not comparable to one under 256), so we drop the flags rather
    than report a number built from two different rulers.
    """
    caps = set()
    for d in run_dirs:
        manifest = results.read_manifest(d) or {}
        cond = manifest.get("conditions", {}).get("debate", {})
        caps.add(cond.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    if len(caps) == 1:
        return caps.pop()
    print(f"note: run dirs disagree on max_new_tokens {sorted(caps)}; "
          "truncation counts suppressed")
    return None


def _edgelists(run_dirs: list[str], override: str | None) -> dict[str, list]:
    """instance_id -> edge list, from each run's own dataset (deduped across runs)."""
    paths = {override} if override else {
        (results.read_manifest(d) or {}).get("dataset") for d in run_dirs
    }
    out: dict[str, list] = {}
    for p in sorted(x for x in paths if x):
        if not Path(p).exists():
            print(f"note: dataset {p} not found; Critic grounding will skip its instances")
            continue
        for inst in load_dataset(p):
            out[inst.instance_id] = inst.graph_edgelist
    return out


def dir_prompt_version(run_dir: str) -> str | None:
    """The Proposer wording a run dir's debate rows were produced under (None if no
    debate rows). Reads the manifest, falling back to the rows for dirs that predate
    the manifest field."""
    man = results.read_manifest(run_dir) or {}
    pv = man.get("conditions", {}).get("debate", {}).get("prompt_version")
    if pv:
        return pv
    versions = {results.row_prompt_version(r)
                for f in results.result_files(run_dir) for r in results.read_rows(f)
                if r["condition"] == "debate"}
    return versions.pop() if len(versions) == 1 else None


def select_debate_dirs(run_dirs: list[str], requested: str | None) -> list[str]:
    """Which run dirs' debate output to analyse, refusing to silently pool two versions.

    Selection is per **run dir**, not per row: v1 and v2 runs cover the SAME instance
    ids, so filtering traces by instance id would keep both. Pooling two wordings would
    average two different experiments into one accuracy, which is a wrong number rather
    than a noisy one, so a mixed set is an error unless `--prompt-version` picks one.
    Dirs with no debate rows (a baseline-only dir) are always kept: they contribute the
    baseline the CoT column needs.
    """
    versions = {d: dir_prompt_version(d) for d in run_dirs}
    present = sorted({v for v in versions.values() if v})
    if requested is None:
        if len(present) > 1:
            raise SystemExit(
                f"run dirs mix debate prompt versions {present}; these are different "
                f"experiments and must not be pooled. Re-run with "
                f"--prompt-version {present[0]} (or {present[1]})."
            )
        return list(run_dirs)
    if requested not in present:
        raise SystemExit(f"no debate rows with prompt_version={requested!r}; present: {present}")
    kept = [d for d in run_dirs if versions[d] in (requested, None)]
    dropped = [d for d in run_dirs if d not in kept]
    if dropped:
        print(f"note: prompt_version={requested}; ignoring debate output from "
              f"{', '.join(dropped)}")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="+",
                    help="one or more run dirs; multiple are pooled (replication)")
    ap.add_argument("--prompt-version", default=None,
                    help="which Proposer wording's debate rows to analyse (e.g. v1, v2). "
                         "Required when the given run dirs contain more than one.")
    ap.add_argument("--dataset", default=None,
                    help="dataset JSONL for the Critic-grounding audit "
                         "(default: each run's manifest `dataset`)")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="write the diagnostic CSVs here (e.g. analysis/main)")
    args = ap.parse_args()

    # Baseline is version-independent, so it is read from every dir given; debate output
    # comes only from the dirs matching the selected prompt version.
    base_rows = [r for d in args.run_dir for f in results.result_files(d)
                 for r in results.read_rows(f) if r["condition"] == "baseline"]
    debate_dirs = select_debate_dirs(args.run_dir, args.prompt_version)
    debate_rows = [r for d in debate_dirs for f in results.result_files(d)
                   for r in results.read_rows(f) if r["condition"] == "debate"]
    traces = [t for d in debate_dirs for f in results.trace_files(d)
              for t in results.read_traces(f)]

    if not debate_rows or not traces:
        raise SystemExit(
            f"no debate rows ({len(debate_rows)}) or traces ({len(traces)}) in "
            f"{', '.join(args.run_dir)} -- diagnostics need the trace sidecars"
        )

    edgelists = _edgelists(args.run_dir, args.dataset)
    views = diag.debate_views(traces, debate_rows, edgelists=edgelists,
                              max_new_tokens=_max_new_tokens(args.run_dir))
    seeds = sorted({v["instance_id"].split("/")[0] for v in views})
    print(f"{len(views)} debate instances in {', '.join(args.run_dir)}  "
          f"(seeds: {', '.join(seeds)}; {len(base_rows)} baseline rows)\n")

    split = diag.turn_split(views, base_rows)
    conf = diag.critic_confusion(views)
    pooled = diag.pooled_confusion(views)
    rev = diag.revision_effect(views)
    comp = diag.compliance(views)
    shape = diag.error_shape(views)
    replay = diag.replay_stopping_rules(views)
    ground = diag.critic_grounding(views) if edgelists else {}

    print("-- turn split: baseline -> turn 1 (CoT effect) -> final (loop effect) --")
    print(diag.format_turn_split(split))

    print("\n-- critic verdict vs whether the answer it judged was correct --")
    print("   (FA = P(REVISE | correct), det = P(REVISE | wrong); gap ~ 0 means no signal)")
    print(diag.format_critic_confusion(conf, pooled))
    print(f"\n   pooled: a REVISE moves P(the answer is wrong) from "
          f"{pooled['base_rate_wrong']:.3f} to {pooled['revise_precision']:.3f} "
          f"(chi2={pooled['chi2']:.2f}, phi={pooled['phi']:+.3f}, OR={pooled['odds_ratio']:.2f})")

    if ground:
        print("\n-- critic evidence: is the cited pair actually an edge of the graph? --")
        print(diag.format_critic_grounding(ground))

    print("\n-- what a REVISE does to the answer (net = bad>ok minus ok>bad) --")
    print(diag.format_revision_effect(rev))

    print("\n-- format compliance (wrong for a non-reasoning reason) --")
    print(diag.format_compliance(comp))

    print("\n-- error shape: how a turn-1 answer is wrong, in that task's own terms --")
    print(diag.format_error_shape(shape))

    print("\n-- counterfactual stopping rules, replayed on these traces --")
    print("   (each rule stops the loop earlier; delta is against the run as it happened)")
    print(diag.format_stopping_rules(replay))

    if not args.save:
        return
    out = Path(args.save)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "debate_turn_split.csv": diag.turn_split_to_csv(split),
        "debate_critic_confusion.csv": diag.critic_confusion_to_csv(conf, pooled),
        "debate_revision_effect.csv": diag.revision_effect_to_csv(rev),
        "debate_compliance.csv": diag.compliance_to_csv(comp),
        "debate_error_shape.csv": diag.error_shape_to_csv(shape),
        "debate_stopping_rules.csv": diag.stopping_rules_to_csv(replay),
    }
    if ground:
        artifacts["debate_critic_grounding.csv"] = diag.critic_grounding_to_csv(ground)
    for name, text in artifacts.items():
        (out / name).write_text(text, encoding="utf-8")
    print(f"\nsaved {' + '.join(sorted(artifacts))} -> {out}/")


if __name__ == "__main__":
    main()
