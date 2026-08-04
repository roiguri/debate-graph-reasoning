"""Re-score persisted debate output under the CURRENT answer parser.

The parser is a scoring rule, not a generation rule, so unlike the prompts it is **not
versioned**: scoring one run under an old rule and another under a new one would make the
conditions incomparable. When the rule changes, every affected run is re-scored instead.
That costs no GPU because the raw model output is persisted in both the rows and the
trace sidecars, so correctness is re-derivable exactly.

**Every condition is re-scored**, because the shared `scoring.parse` can change and not
just the debate-only `parse_proposer`. Scoring one condition under a new rule and leaving
another on the old one would silently bias the comparison between them -- the exact
unequal-standards trap docs/findings.md 3g warns about -- so this walks every row it
finds. Majority vote needs no special handling: the vote is derived from the parsed
answers at report time, never stored, so re-scoring its sample rows re-derives it.

Dry-run by default -- it prints what would change and writes nothing:

    python scripts/rescore.py results/main
    python scripts/rescore.py results/main results/seed11 results/seed13 --write

`--write` rewrites `parsed_answer` / `parse_ok` / `correct` in the rows and `parsed` /
`parse_ok` in the traces, and records the re-scoring in the run manifest. It is
idempotent: re-running it against an already-current run reports zero changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gedebate
from gedebate.data.store import load_dataset
from gedebate.eval import results
from gedebate.eval.scoring import parse as parse_answer
from gedebate.eval.scoring import score
from gedebate.prompts.debate import parse_proposer


def _node_ids(run_dir: str, override: str | None) -> dict[str, list]:
    """instance_id -> node_ids, needed to parse connected_nodes answers."""
    path = override or (results.read_manifest(run_dir) or {}).get("dataset")
    if not path or not Path(path).exists():
        raise SystemExit(
            f"{run_dir}: dataset {path!r} not found; connected_nodes cannot be re-scored "
            f"without it (pass --dataset)"
        )
    return {i.instance_id: i.node_ids for i in load_dataset(path)}


def _reparse(raw: str, row: dict, node_ids: list | None):
    """Re-derive (value, parse_ok, correct) under the parser this row's condition uses.

    Debate output carries the claim + `ANSWER:` scaffold, so it goes through
    `parse_proposer`; baseline and majority-vote output is already a bare answer and goes
    straight to `scoring.parse`, which is what those conditions called when they ran.
    """
    if row["condition"] == "debate":
        value, ok, _ = parse_proposer(
            raw, row["task"], encoding=row["encoding"], node_ids=node_ids)
    else:
        value, ok = parse_answer(
            row["task"], raw, encoding=row["encoding"], node_ids=node_ids)
    return value, ok, score(value, row["ground_truth"])


def rescore_dir(run_dir: str, dataset: str | None, write: bool) -> dict:
    """Re-derive every row (all conditions) + every debate trace turn from persisted raw text."""
    node_ids = _node_ids(run_dir, dataset)
    rows_by_id: dict[str, dict] = {}
    for f in results.result_files(run_dir):
        for r in results.read_rows(f):
            if r["condition"] == "debate":
                rows_by_id[r["instance_id"]] = r

    stat = {"rows": 0, "rows_changed": 0, "flips": 0, "turns": 0, "turns_changed": 0,
            "conditions": set()}

    # traces first: every turn, so turn-level diagnostics match the rows
    for f in results.trace_files(run_dir):
        traces = results.read_traces(f)
        dirty = False
        for t in traces:
            row = rows_by_id.get(t["instance_id"])
            if row is None:
                continue
            ids = node_ids.get(t["instance_id"])
            for turn in t["turns"]:
                if turn.get("role") != "proposer":
                    continue
                stat["turns"] += 1
                value, ok, _ = _reparse(turn["raw"], row, ids)
                if turn.get("parsed") != value or bool(turn.get("parse_ok")) != ok:
                    turn["parsed"], turn["parse_ok"] = value, ok
                    stat["turns_changed"] += 1
                    dirty = True
        if dirty and write:
            Path(f).write_text("".join(json.dumps(t) + "\n" for t in traces), encoding="utf-8")

    # rows: the FINAL proposer answer is what a row records
    final_raw = {}
    for f in results.trace_files(run_dir):
        for t in results.read_traces(f):
            props = [x for x in t["turns"] if x.get("role") == "proposer"]
            if props:
                final_raw[t["instance_id"]] = props[-1]["raw"]

    for f in results.result_files(run_dir):
        rows = results.read_rows(f)
        dirty = False
        for r in rows:
            stat["rows"] += 1
            stat["conditions"].add(r["condition"])
            # A debate row records the FINAL proposer turn, which lives in the trace;
            # every other condition already stores its own answer.
            raw = (final_raw.get(r["instance_id"], r["raw_output"])
                   if r["condition"] == "debate" else r["raw_output"])
            value, ok, correct = _reparse(raw, r, node_ids.get(r["instance_id"]))
            if (r["parsed_answer"] != value or bool(r["parse_ok"]) != ok
                    or bool(r["correct"]) != correct):
                stat["flips"] += bool(r["correct"]) != correct
                r["parsed_answer"], r["parse_ok"], r["correct"] = value, ok, correct
                stat["rows_changed"] += 1
                dirty = True
        if dirty and write:
            Path(f).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    if write and stat["rows_changed"]:
        _stamp_manifest(run_dir, sorted(stat["conditions"]))
    return stat


def _stamp_manifest(run_dir: str, conditions: list[str]) -> None:
    """Record which of this run's conditions were re-scored, and under what code version.

    Stamps each condition that was actually walked, not just debate: a run whose baseline
    was re-scored under a new shared parser must say so, or its provenance claims a
    scoring rule it no longer uses.
    """
    path = results.manifest_path(run_dir)
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name in conditions:
        cond = manifest.setdefault("conditions", {}).setdefault(name, {})
        cond["rescored_by"] = f"scripts/rescore.py (gedebate {gedebate.__version__})"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="+")
    ap.add_argument("--dataset", default=None,
                    help="dataset JSONL (default: each run's manifest `dataset`)")
    ap.add_argument("--write", action="store_true",
                    help="apply the changes (default: dry run, writes nothing)")
    args = ap.parse_args()

    if not args.write:
        print("DRY RUN -- nothing is written. Re-run with --write to apply.\n")
    for d in args.run_dir:
        s = rescore_dir(d, args.dataset, args.write)
        verb = "changed" if args.write else "would change"
        print(f"{d}: {s['rows_changed']}/{s['rows']} rows {verb} "
              f"({s['flips']} correctness flips), "
              f"{s['turns_changed']}/{s['turns']} proposer turns {verb}"
              f"  [conditions: {', '.join(sorted(s['conditions']))}]")


if __name__ == "__main__":
    main()
