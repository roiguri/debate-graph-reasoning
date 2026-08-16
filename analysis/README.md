# analysis/ - derived tables (how to use + reproduce)

Everything in this folder is **derived** from `results/` and fully regenerable, so
the CSVs are gitignored (this README is not). They are the tables the writeup cites.
For the dataset -> run pipeline see the repo README; this file covers the
**analysis step**.

## Entrypoint: `scripts/show_results.py`

Reads one or more run dirs, splits rows by condition, prints tables, and (with
`--save`) writes CSVs. Torch-free, runs locally after `scripts/pull_results.sh`.

```
python scripts/show_results.py <run_dir> [<run_dir> ...] [flags]
```

Flags:
- `--fragility` - per-task cross-encoding spread (mean/std/max-min) + paired significance.
- `--compare` - vote-vs-baseline delta table (needs both conditions in the dir).
- `--condition C[,C2]` - restrict to these conditions (e.g. `baseline` for a pooled view).
- `--by-seed` - with >1 seed pooled, also print each seed's own fragility table.
- `--save DIR` - write CSVs (condition-tagged filenames) into DIR.
- `--raw [--wrong-only] [--task T] [--encoding E]` - per-instance raw output -> parsed.

Passing several run dirs **pools** them (replication); pairing is by `seed/graph_index`.

## Layout

One subfolder per **analysis scope**; every file is tagged by the condition it describes.

```
analysis/
  main/      # seed 7 (data/main.jsonl) - the matched-compute anchor: baseline + majority-vote
    baseline_summary.csv        baseline_fragility.csv    baseline_significance.csv
    mv_vote_summary.csv         mv_vs_baseline.csv
  pooled/    # seeds 7 + 11 + 13 - the fragility replication (baseline only; MV ran seed 7 only)
    baseline_summary.csv        baseline_fragility.csv    baseline_significance.csv
```

## Regenerate (the canonical commands)

Prerequisite: `results/` pulled locally (`main`, `seed11`, `seed13`).

```bash
# seed 7 anchor: baseline + majority-vote + the vote-vs-baseline comparison
python scripts/show_results.py results/main --compare --save analysis/main

# pooled fragility across the three seeds (baseline only)
python scripts/show_results.py results/main results/seed11 results/seed13 \
    --fragility --condition baseline --save analysis/pooled
```

## What each file holds

- **baseline_summary.csv** - per (task, encoding): `accuracy`, 95% Wilson CI, `parse_ok_rate`
  (confound guard: low means "wrong" is really a parse failure), `total_gen_tokens`.
- **baseline_fragility.csv** - per task: `mean`/`std`/`max_min` of accuracy across the 3
  encodings + `best`/`worst`. The encoding-fragility metric.
- **baseline_significance.csv** - per task, **paired** tests (the 3 encodings share graphs):
  Cochran's Q (omnibus) + McNemar (best-vs-worst gap).
- **mv_vote_summary.csv** - per cell: `voted_accuracy` (mode of the N draws, per instance) +
  CI, `per_sample_accuracy` (mean single draw), `n_samples`, `total_gen_tokens`.
- **mv_vs_baseline.csv** - per cell: baseline vs voted accuracy, `delta`, and `token_mult`
  (the majority-vote compute cost over baseline, ~10x at N=10).

## Metrics in one line each
- **Accuracy** = exact match vs normalized ground truth (bool / int / sorted set).
- **Wilson CI** = 95% interval on a cell's accuracy.
- **Fragility** = cross-encoding accuracy spread per task (std, max-min).
- **Paired significance** = Cochran's Q + McNemar, valid because encodings share graphs.
- **Matched compute** = total generated tokens (the currency for comparing conditions).

## Note
The CSVs here are regenerable, not source. Committed derived artifacts (final tables
+ figures) land in P6.
