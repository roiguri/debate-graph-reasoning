# P4 — Majority-vote condition (self-consistency)

**Status: planned.** Second condition after baseline: draw N sampled answers per
instance, vote, and report accuracy + total generated tokens so it compares to
baseline at a known compute cost. See [overview.md](overview.md); P2/P3 built the
machinery this reuses, and the persistence schema already reserves the fields P4
needs (`sample_index`, `temperature`, `seed`).

Goal: add the majority-vote condition end-to-end (N-sample generation, vote
aggregation, token accounting, resumable run) over the **same frozen instances**
the baseline scored, and produce a vote-vs-baseline accuracy table per cell with
the token multiplier made explicit. This is the first half of the debate question:
does spending N times the compute on independent samples buy back the accuracy the
fragile encodings lose?

**Done when:** `results/main/majority_vote/` holds N rows per instance for all 9
cells, `report` produces a vote-aggregated accuracy table (voted answer per
instance) alongside per-sample accuracy and total tokens, resume/top-up after a
kill is tested, and the vote-vs-baseline delta per cell is recorded in
[notes.md](../notes.md).

## What's reused vs new

P4 is thin by design. The schema, resume, sharding, manifest, model sampling path,
and per-cell reporting all exist. New code is one condition module, a few config
fields, a runner dispatch, a vote-aware report view, a config + slurm reuse, and
tests.

Reused **unchanged**: `data.store` (load), `select_shard`/`parse_shard`,
`results.*` (schema, `append_row`, `expected_attempts`, `missing_samples`,
`ensure_manifest`), `Model.generate(temperature=, seed=)`, `scoring.parse/score`,
`prompts.build_prompt`, `slurm/matrix.slurm` (config via `CONFIG` env).

## Decisions locked

- **Voting is derived, not stored.** Persist N lean attempt rows per instance (one
  per `sample_index`), exactly as the schema docstring already specifies. The voted
  answer is recomputed at report time from the N rows, so the vote rule stays
  re-derivable and rows never carry aggregate state. No schema change.
- **Vote rule:** mode over the N **parsed** answers, parse-failures excluded from
  the tally (self-consistency convention). If every sample fails to parse, the voted
  answer is a parse-failure (scored incorrect). Deterministic tie-break: among tied
  answers, pick the one whose first supporting sample has the lowest `sample_index`
  (stable, reproducible). Ties are minimized by choosing an odd N.
- **Per-sample determinism:** seed each draw from `(instance_id, sample_index)` so a
  killed shard tops up exactly the missing samples (`results.missing_samples`) and a
  rerun reproduces the same draws. Same greedy-fp16 caveat as baseline: we verify
  parsed answers, not raw bytes.
- **Same prompt, same `max_new_tokens` (128) as baseline.** Only decoding changes
  (sampling vs greedy), so a per-sample row is directly comparable to a baseline row.
- **Matched compute is reported, not enforced here.** Majority-vote spends about N
  times the baseline's generated tokens; P4 makes that multiplier explicit (voted
  accuracy next to total tokens). The equal-budget comparison across all three
  conditions is P6's job.
- **Sampling budget: N = 10, temperature = 0.7.** Standard self-consistency
  decoding at the stronger end of the usual N range, to resolve the small
  vote-vs-baseline deltas on the fragile encodings. About 10 times the baseline's
  generated tokens per instance; the array is throttled like P3 (`%3`) to respect
  the quota. Even N, so the deterministic tie-break above does real work.
- **Scope: main (seed 7) first.** Run majority-vote on `data/main.jsonl` only,
  keeping the budget aligned with the baseline anchor and P6. Seeds 11/13 are an
  **optional** follow-on (P4.4) for pooled significance, run only if the main result
  warrants confirming.
- **Pilot before the full run.** A tiny sampled pass (few instances, 1 cell) proves
  sampling + vote + top-up on the cluster before spending real matrix compute,
  mirroring every prior GPU phase.

## Vertical slices

### P4.1 — Condition module + vote (torch-free, unit-tested)
- `conditions/majority_vote.py`:
  - `run_sample(model, instance, *, sample_index, temperature, max_new_tokens)` —
    baseline's prompt → generate → parse → score, but sampled with a seed derived
    from `(instance_id, sample_index)`; returns one attempt record (same shape as
    baseline's, so `results.make_row` is unchanged).
  - `vote(parsed_answers) -> (voted_answer, parse_ok, support)` — mode with the
    tie-break above; handles bool / int / sorted-list answer shapes (lists keyed by
    tuple).
  - `CONDITION = "majority_vote"`.
- Tests: vote on bool/int/list answers, all-fail case, tie-break determinism,
  seed-derivation reproducibility. All CPU/stub, no GPU.

### P4.2 — Config + runner dispatch + resume
- `config.py`: add `"majority_vote"` to `KNOWN_CONDITIONS`; add `n_samples: int`
  (default 5) and `temperature: float` (default 0.7) fields; validate `n_samples >= 1`
  and `temperature > 0` (even N is fine, the tie-break handles it).
- `runner.py`: dispatch on `cfg.condition`. For majority-vote, per instance run
  `results.missing_samples(...)` and persist one row per sample_index with
  `sample_index`/`temperature`/`seed` set. `manifest_record` sets
  `decoding = f"temperature={T}"` and records `n_samples`. Baseline path untouched.
- Tests: runner over a stub model persists N rows/instance; a simulated kill
  (partial rows) resumes to exactly N; manifest guard still holds.

### P4.3 — Vote-aware report + pilot + full run + checkpoint
- `report.py`: a vote-aggregated view — group the N rows per instance, compute the
  voted answer, score it, and report per-cell **voted accuracy** (with Wilson CI)
  next to **per-sample accuracy** and **total_gen_tokens** (the N× cost). Keep the
  existing per-row summary for baseline. Surface via `show_results.py`.
- `configs/mv.toml` (+ any seed variants per the scope decision); reuse
  `slurm/matrix.slurm` via `CONFIG=configs/mv.toml sbatch --array=... matrix.slurm`.
- **Pilot** one cell at small N locally/on one shard; confirm vote + top-up; then
  launch the array, resumable, throttled like P3.
- Pull results, compute the vote-vs-baseline delta per cell, and record it in
  [notes.md](../notes.md): does voting lift the worst encodings, and at what token
  cost? This is the P4 result the writeup cites and P6's matched-compute analysis
  builds on.

### P4.4 — Replication on seeds 11/13 (optional)
Run only if the main result warrants confirming. Reuse the P3 sibling-dataset
pattern: `configs/mv-seed11.toml` / `mv-seed13.toml` pointing at `data/seed11.jsonl`
/ `data/seed13.jsonl`, submit via the same `CONFIG` env, pool with the P3 paired
tests (`stats`, `show_results.py --by-seed`) to check the vote effect is not a
seed-7 artifact. No new code, just configs + a run.

## Notes for later phases
- P5 (debate) and P6 (matched compute) read these same rows; the schema and resume
  are unchanged, so debate persists 1 summed-token row per instance under a sibling
  `debate/` folder and joins cleanly.
- The N× token multiplier measured here is the yardstick P6 uses to give debate an
  equal budget.
