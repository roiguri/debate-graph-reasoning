# Refactor — Freeze the dataset as an artifact + reproducible results

**Branch:** `refactor/dataset` (main stays untouched until proven). Planned before
P4 because P4/P5/P6 compare conditions **per-instance at matched compute**, which is
only valid if every run sees byte-identical instances.

## Motivation
Today the runner **regenerates** graphs/instances from `seed + N` on every run
(`build_dataset`). That leans on implicit determinism — the generator code,
networkx/numpy RNG behavior, and N must stay identical forever, or an `instance_id`
silently remaps to a different graph. It also made N non-extensible (see
[notes.md](../notes.md#graph-generation-is-not-n-extensible--fact-p3)), which we
patched symptomatically. Best practice for an experiment is a **frozen dataset
artifact** every run loads. This refactor makes that change and, on the way, makes
both the dataset and the results **reproducible + verifiable**.

## Decisions
- **Commit the dataset:** `data/main.jsonl` (one `Instance` per line) + a
  `data/main.meta.json` provenance sidecar. Small (~1–2 MB), citable, drift-proof.
- **Configs name a dataset path;** an explicit build step materializes it; the
  runner only **loads** (no regeneration in the run path).
- **Provenance:** the dataset meta records its build spec + `sha256` (verify by
  rebuild); the run manifest becomes a full **reproduction record**.

## Safety (don't spoil the baseline)
`results/` is gitignored, so the existing `results/main/baseline/` lives in the
working tree and **survives branch switches** — this branch can't delete it. The
only real risk is *logical*: the frozen dataset must match the graphs the baseline
was scored against. So the refactor is **gated on a proof of equivalence** (R1); we
merge to main only once it's green. If it fails, main + baseline are untouched.

## Slices

### R1 — Materialize + provenance + equivalence gate (no runner change)
Prove the artifact equals today's generation and is consistent with existing results.
- `gedebate.data`: `dump_dataset(instances, path)` + `load_dataset(path) -> list[Instance]`
  (reconstruct from JSONL; `build_dataset` stays as the generator of truth).
- `scripts/build_dataset.py`: build → write `data/main.jsonl` + `data/main.meta.json`
  = `{spec: {algorithm, dataset_seed, n_graphs, tasks, encodings}, n_instances,
  sha256, gedebate_version, git_commit, created}`.
  - `--verify`: rebuild in-memory, recompute the hash, assert it matches the meta.
    **Deterministic, no GPU** — this is how anyone reproduces the data.
- **Equivalence tests (the gate):**
  1. `load_dataset(data/main.jsonl)` equals `build_dataset(seed=7, N=200)`
     field-by-field (ids, edgelists, ground_truth, questions, answers).
  2. Cross-check the **existing** `results/main/baseline/` rows: every `instance_id`
     is in the dataset and its `ground_truth` + `graph_edgelist` match. (Proves the
     baseline stays valid under the frozen dataset.)

### R2 — Switch runner/config to load-mode + reproducible results
- `RunConfig`: replace `n_graphs`/`dataset_seed` with `dataset` (path); `tasks`/
  `encodings` become **filters** over the loaded set; keep `model`/`condition`/
  `out_dir`/`max_new_tokens`. Runner: `load_dataset` → filter → shard (downstream
  unchanged).
- **Run manifest = reproduction record:** `model`, `dataset` (path + `sha256`),
  decoding (`greedy`/`temperature`), `max_new_tokens`, `gedebate_version`,
  `git_commit`, `config` path, `created`. `ensure_manifest` now guards
  `model` + `dataset_sha256` (supersedes the `n_graphs`/`dataset_seed` guard — the
  hash pins the exact instance set).
- **Reproduce + verify the results:**
  - The manifest carries the exact command/inputs to re-run; document it (README/notes).
  - `runner --verify-sample K`: re-run K persisted instances and assert the **parsed
    answer** matches the stored row (semantic reproduction). Note: greedy fp16 on GPU
    is not guaranteed byte-identical across hardware, so we verify parsed answers, not
    raw text.
- **Migration (local, one-off):** rewrite the existing `results/main/manifest.json`
  to the new record (add `dataset_sha256`, drop `dataset_seed`/`n_graphs`) so resume
  of the proven-equivalent baseline keeps working — **no baseline rerun**.
- Update `configs/*` (→ `dataset = data/main.jsonl`) and docs (p3-matrix.md, notes).

## Done when
`data/main.jsonl` + meta are committed and hash-verify; `load_dataset` proves
equal to today's generation and consistent with the existing baseline; the runner
loads the artifact; the run manifest reproduces a run and `--verify-sample` passes;
then merge to main. The N-extensibility footgun is gone (grow the dataset by
appending a new seed's instances; existing ones and their `instance_id`s never move).
