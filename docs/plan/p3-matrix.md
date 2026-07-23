# P3 — Baseline across the full 3×3 matrix + fragility analysis

**Status: complete — premise confirmed and replicated, GO for P4.** N=200 baseline
over the full 3×3 (1800 instances, parse_ok≈1.0) reproduces encoding-fragility:
incident best for node_degree (0.75) and connected_nodes (0.345), edge_existence
encoding-insensitive (0.02 spread). Then **replicated on two independent seeds
(11, 13)**: pooled to 600 graphs/cell the effect is paired-significant (node_degree
p=3e-44, connected_nodes p=2e-11) and edge_existence stays a clean null (p=0.77).
Numbers + the go decision + the replication result in
[notes.md](../notes.md#p3-result-encoding-fragility-reproduces--fact--go-decision-p34).

Goal: run the baseline over the **full task × encoding matrix at real N**, shard it
across the cluster, and produce the **encoding-fragility result** — a per-encoding
accuracy table plus the cross-encoding spread per task. This is the phase that
**checkpoints the premise**: does encoding-fragility actually reproduce on our 3B
model? If not, the whole debate question is moot, so we gate P4/P5 on it. See
[overview.md](overview.md); P2 built the machinery this reuses unchanged.

**Done when:** `results/main/baseline/` holds baseline attempts for all 9 cells at
N=200, a fragility table (mean/std/max−min per task) is produced from it, the
results are pulled to the local repo, and we have a **go/no-go on the premise**
recorded in [notes.md](../notes.md).

## Decisions locked

- **N = 200 graphs/cell, seed = 7, fixed up front.** 1800 baseline instances;
  95% CI ≈ ±7%, enough to resolve the 10–30pt fragility spreads. **N is not
  extensible** (the generator re-samples the whole sequence when N changes — see
  [notes.md](../notes.md#graph-generation-is-not-n-extensible--fact-p3)), so a
  different N means a fresh `out_dir` and a baseline rerun. Tighter CIs come from
  additional **seeds** (pooled), not a bigger N in place. **Done in P3.3:** seeds
  11 and 13 pooled to 600 graphs/cell.
- **Model:** Qwen2.5-3B-Instruct (validated in P2.5; parse_ok=1.0, off the floor).
- **Sharding:** a **SLURM job array**, conservatively throttled (`%3`) so it is
  safe under an unknown per-student concurrent-job quota — extra shards pend, they
  don't error. Confirm the real limit with `sacctmgr`/`scontrol` and raise the
  throttle if higher.
- **Matched-compute contract:** P4/P5 reuse these exact instances (same seed, same
  N). If they become compute-bound, they subsample by `graph_index` consistently
  (a stable prefix), never by re-generating.

## What's reused vs new

P3 is deliberately thin — it is P2's machinery run **at scale**, plus a small
analysis layer. Reused **unchanged**: `build_dataset`, the runner + sharding
(`select_shard`/`parse_shard`), persistence + resume, `report.summarize`,
`show_results.py`. The only new code is a one-line manifest guard, a config, a
slurm array file, a per-task fragility stat, and two ops shell helpers. So: two
slices, not five.

## Vertical slices

### P3.1 — Run the full matrix (reuse P2 at scale)
Launch the whole 3×3 at N=200 and let it complete, resumably, across the cluster.
- `configs/matrix.toml`: full 3×3, `n_graphs = 200`, `dataset_seed = 7`,
  `out_dir = results/main` (condition in a `baseline/` subfolder), `max_new_tokens = 128`.
- **The one code change:** harden `results.ensure_manifest` to guard `dataset_seed`
  **and** `n_graphs` (it already records them). `instance_id` omits N, so resuming
  an `out_dir` with a different N would skip-by-id onto *different* graphs — the
  guard makes that a hard error, not silent corruption. Add a test.
- `slurm/matrix.slurm`: P2 node pins; shard from the array env
  (`--shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}`); logs → `results/logs/`.
  Submit `sbatch --array=0-7%3 slurm/matrix.slurm` (8 shards, ≤3 concurrent —
  safe under unknown quota). Killed shards rerun and skip done work; the union of
  shard files is the full run.
- Done: 1800 rows in `results/main/baseline/`, `parse_ok` ≈ 1.0 at real N.

### P3.2 — Retrieve, analyze, checkpoint the premise
Get the results local, compute fragility, and decide go/no-go for P4/P5.
- **Ops (get results off the cluster).** `slurm/clean_logs.sh` clears the P0/P2
  clutter (`results/*.out`, `results/*.err`, `smoke.*`) while **keeping** every run
  data dir + `manifest.json`; routes future logs to `results/logs/` (one-time
  `mkdir -p results/logs`). `scripts/pull_results.sh` rsyncs results **cluster →
  local** for analysis (report/results need no torch, so analysis runs locally):
  ```bash
  rsync -av slurm-client:/home/yandex/MLWG2026/‹user›/graph-encodings-with-debate/results/ ./results/
  ```
- **Fragility stat.** Extend `eval/report.py` with per-task cross-encoding **mean /
  std / max−min** of accuracy (the proposal's secondary metric), surfaced as a
  `--fragility` view in `scripts/show_results.py`.
- **Premise checkpoint (the gate).** Read the table and decide **go/no-go for
  P4/P5**, recording it in [notes.md](../notes.md) (this IS the writeup's "we
  reproduced fragility" claim):
  - **Go** if a real spread exists per task, worst encoding clearly below best, and
    cells are neither all-ceiling nor all-floor.
  - **Reconsider** if edge_existence (~0.85–0.9 in the pilot) saturates to a
    ceiling or a task hits the floor — decide task/prompt/model *before* spending
    P4/P5 compute.
  - `results/` stays gitignored; only the **derived** table/figures get committed
    later under `analysis/` (P6).

### P3.3 — Replication + significance (done)
Confirm the fragility is not a seed-7 artifact and put numbers behind it.
- **Sibling datasets.** `build_dataset.py` grew `--name`/`--seed` so independent
  seeds build as **sibling artifacts** (frozen `main` untouched): `data/seed11.jsonl`,
  `data/seed13.jsonl`, matched to `main` on N/tasks/encodings. Own `configs/seed*.toml`
  and `results/seed*/` (own manifest each). `slurm/matrix.slurm` selects the config at
  submit time via `CONFIG` (default `configs/matrix.toml`), so one array job drives
  every seed.
- **Paired significance.** New `gedebate.eval.stats`: Wilson CI per cell, McNemar
  (best-vs-worst), Cochran's Q (omnibus). Paired because encodings share graphs.
  `show_results.py` pools multiple run dirs and prints per-seed tables with `--by-seed`.
- **Result.** node_degree replicates cleanly (incident best / adjacency worst in all
  three seeds, pooled p=3e-44); connected_nodes' incident-advantage replicates
  (p=2e-11) though the worst encoding wobbles; edge_existence is a confirmed null.
  Full writeup in [notes.md](../notes.md).

## Notes for later phases
- P4/P5 point at the **same instances** (`configs/*` sharing seed=7, N=200); the
  persistence schema and resume are unchanged from P2. The replication seeds (11,
  13) are available if a later phase wants them, but the matched-compute comparison
  runs on seed 7 to keep budgets aligned.
- The array-throttle pattern (`%k`) is how every later GPU phase respects the quota;
  raise `k` once the `sacctmgr` probe confirms the real limit.
- Tighter CIs come from pooling more **seeds** (done: 11, 13 as sibling artifacts),
  never from growing N in place (non-extensible).
