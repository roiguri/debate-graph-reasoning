# P4 review record (independent review of the majority-vote phase)

An independent reviewer audited P4 before P5. This is the record of every finding and
its resolution (all closed). Kept because the decisions here (Framing A, D1, and M3
deferred to P5) carry forward; `notes.md`'s P4 section links here for the rationale.

Status key: [ ] open · [~] in progress · [x] done · [-] won't fix (with reason)

**Framing decision: MV is a compute-baseline CONTROL (Framing A).** Its job is to show
"raw sampling compute doesn't fix fragility." So MV ≈ greedy is the *desired* result;
we make it defensible, not competitive. The strongest reason for the null is the
single-token-voting → argmax argument (multi-token connected_nodes aside), which holds
regardless of sampling diversity. CoT (M3) is deferred to P5.

## Methodology (affect whether the null is trustworthy)

- [x] **M1 (code) — Sampling top_p/top_k now explicit + config-level.** `top_p`/`top_k`
  are RunConfig fields, threaded run_sample -> model.generate, recorded in the MV
  manifest. Defaults 0.8/20 (Qwen's) reproduce the committed run exactly, so this is
  behavior-preserving (no rerun needed for consistency). Commit 96e83c8. **D1 still
  open:** keep 0.8/20 (lean on single-token theory, no rerun) vs flip to 1.0/0 + rerun
  to confirm the null under wider diversity.

- [x] **M2 — paired McNemar of vote vs baseline → DONE.** `compare_baseline_vote` now
  pairs each instance's baseline-correct vs voted-correct (b/c), and `--compare` prints
  b/c + McNemar p (shared math in `stats.mcnemar_from_bc`). Result: every cell
  non-significant (discordance 1-9/200, p ≥ 0.69, 8/9 at p=1.0) = statistically
  indistinguishable from greedy. In `analysis/main/mv_vs_baseline.csv` + notes. No
  separate p4-analysis doc needed (narrative lives in notes.md).

- [-] **M3 — No chain-of-thought / low reasoning diversity.** Deferred to P5: under
  Framing A, MV stays direct-answer; the CoT-vs-no-CoT fairness question is resolved
  where debate's design forces it. Not a P4 task.

- [x] **M4 — Reframe the null's cause in notes → DONE.** notes.md P4 now leads with the
  structural single-token-voting → argmax argument (vote of a ~single-token answer
  converges to greedy; self-consistency needs diverse reasoning paths, which direct
  answers lack), with the systematic-error point secondary. Folds in the M2 result +
  the explicit-sampling note.

## Reproducibility / correctness

- [x] **R1 — Manifest provenance muddled → FIXED (manifest v2, per-condition).**
  `ensure_manifest` now records shared invariants (model/dataset/hash) at top level and
  per-condition provenance under `conditions[...]`, written once, guard unchanged
  (model+dataset_sha256). `scripts/upgrade_manifest.py` migrated `results/main` to v2:
  baseline (reconstructed=true, greedy/matrix.toml) + majority_vote (verbatim) now both
  recorded. Verified every result row byte-identical (digest 3913fe12 unchanged) and the
  guard still passes/rejects correctly. +2 manifest tests. NOT yet committed (holding
  per request). Note: MV `decoding` is the verbatim "temperature=0.7" the run wrote; the
  implicit Qwen top_p=0.8/top_k=20 lives in the M1 notes, not retro-edited here.

- [x] **R2 — `show_results.py` latent pooling bug.** Fixed: `base_rows` now selects
  `condition == "baseline"` (not `!= "majority_vote"`), so P5 debate rows can't pool
  into the baseline table. Data was never affected (display-only). Done.

## Minor

- [x] **N1 — `n_samples` default is 5.** Fixed: default is now 10 (the locked budget)
  in both the dataclass and `from_dict`. Done.

- [x] **N2 — No tie-break test at even N=10.** Added tests at N=10 (binary 5-5 + a
  multi-class top-tie). Decided NOT to switch to odd N: ties are ~5% of instances
  (92/1800), but 88 of them are multi-class (node_degree/connected_nodes) that odd N
  can't prevent, so the deterministic tie-break stays; N=10 kept (no rerun). Done.

- [-] **N3 — `skipped` log counter → no change needed.** The semantics are correct and
  documented: `written` = rows appended, `skipped` = fully-complete instances. A
  partially-resumed instance correctly contributes its new rows to `written` and is not
  "skipped" (work was done). No undercount of any meaningful quantity.

- [x] **N4 — `--save` naming + analysis/ layout → FIXED.** `--save` now writes
  condition-tagged files (`baseline_*.csv`, `mv_*.csv`); added `--condition` filter for
  baseline-only pooled views. Reorganized `analysis/` into scope dirs: `main/` (seed 7,
  baseline+MV) and `pooled/` (seeds 7/11/13, baseline-only), regenerated from two
  canonical commands. Added `analysis/README.md` (committed; CSVs stay gitignored) +
  integration tests. Done.

## Then decide
- [x] **D1 — Rerun vs proceed → OPTION A (no rerun).** Keep Qwen's recommended 0.8/20
  (now explicit); defend the null with the single-token-voting theory (M4) + a paired
  test (M2). No wider-diversity rerun. Vendor-recommended sampling is the more
  defensible control config than an artificially-widened one.
