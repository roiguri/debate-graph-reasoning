# P2 — Baseline, one task × one encoding, end-to-end

Goal: the first **runnable, scoreable** condition — take a data-layer `Instance`,
prompt the model, parse the answer, score it against ground truth, and persist a
result — proven end-to-end on a single (task, encoding) slice. **GPU** (a tiny
CPU run works for wiring; real numbers need the cluster). See
[overview.md](overview.md) for where this fits.

**P2 is the keystone.** It fixes two contracts every later condition reuses:
the **answer-parsing/scoring** contract and the **results-file schema** (with
resume). P4 (majority-vote) and P5 (debate) add sampling and a Critic loop *on
top of* these — they must not have to reopen them. So the parser and schema are
designed for **all three** answer shapes (bool / int / sorted-list) from the
start, even though only one slice is *run* here.

**Done when:** a config-driven command runs `edge_existence × adjacency` over N ER
graphs with Qwen2.5-3B on the cluster, writes one JSONL record per instance
(raw output, parsed answer, `parse_ok`, `correct`, prompt/gen token counts),
**skips already-done instances on restart**, and reports per-encoding accuracy —
with the parser/scorer/schema already general enough that P3's full 3×3 matrix is
a config change, not new code.

## Decisions locked (see notes.md)

- **Model:** Qwen2.5-3B-Instruct, fp16, **config-owned** (single source of truth,
  swappable to 4-bit 7B later) — [notes.md](../notes.md#baseline-model-qwen25-3b-instruct-config-owned--decision-p2).
- **Prompt:** GraphQA **zero-shot, direct answer, no CoT** — question verbatim +
  a minimal terse-format instruction for the chat model —
  [notes.md](../notes.md#baseline-prompt--graphqa-zero-shot-direct-answer--decision-p2).
- **Scoring:** **exact match**, set-equality for connected_nodes, no partial
  credit; `parse_ok` tracked separately —
  [notes.md](../notes.md#scoring--exact-match-incl-connected_nodes-full-set--fact--decision-p2).
- **Results:** **JSONL, one attempt-level row per line, one file per shard**,
  written atomically at attempt end; resume by counting rows per instance. See
  the persistence contract in P2.2 below.
- **Config:** **TOML** via stdlib `tomllib` (Python 3.11 — no new dep).

## Vertical slices

Each slice is end-to-end runnable and adds exactly one capability. Helpers are
built when a slice needs them, not upfront (per the workplan's principles).

### P2.1 — Walking skeleton (one instance, printed)
The narrowest complete path, no persistence. Proves the whole pipe on a single
`edge_existence × adjacency` instance.
- `prompts/`: an edge-existence template = GraphQA `question` verbatim + a minimal
  instruction ("Answer with exactly `Yes` or `No`.").
- `conditions/baseline.py`: `run_instance(model, instance) -> record dict` —
  build prompt → `model.generate` (greedy) → parse → score.
- `eval/scoring.py`: `parse(task, text) -> (value, parse_ok)` and
  `score(parsed, ground_truth) -> correct`. Bool path only for now; the function
  signatures already take `task` so the other shapes slot in at P2.4.
- Entry: `python -m gedebate.eval.runner` on `build_dataset` instance[0], **prints**
  `raw / parsed / parse_ok / correct / n_prompt_tokens / n_gen_tokens`.
- Runs on CPU with a tiny model for wiring; correctness of the *pipe*, not the
  answer, is the bar here.

### P2.2 — Persist + resume (the keystone contract)
Same single-instance run, now writes a durable record and skips it on re-run.
This slice fixes the persistence contract; the design decisions and their
rationale (killable partition + token accounting) are in
[notes.md](../notes.md#persistence-contract--decision-p2).

- **Atomic unit = one completed *attempt*, written once at attempt end.** Baseline
  = 1 row/instance; majority-vote (P4) = N rows (one per sample); debate (P5) = 1
  row carrying *summed* tokens. Kill-safe (a killed attempt leaves no row → clean
  re-run, no double-counted tokens); matched-compute is a stored field, not a
  recomputed groupby.
- Add a deterministic **instance id** to the data layer: `dataset_seed` +
  `graph_index` on `Instance` → `instance_id = "{seed}/{gi}/{task}/{encoding}"`.
  (Small `instance.py` + test change — the resume key, built now because this
  slice needs it.)
- **Uniform row schema — identical shape for all three conditions** (the P4/P5
  contract): `schema_version, instance_id, task, encoding, condition, model,
  sample_index, temperature, seed, raw_output, parsed_answer, parse_ok, correct,
  ground_truth, n_prompt_tokens, n_gen_tokens`. Baseline fills `sample_index=0`,
  `temperature` greedy; MV varies `sample_index`; debate sums the token fields.
  Baseline's row is therefore literally the final schema — nothing reopened later.
- **Verbose debate trace goes to a sidecar**, not the main log:
  `results/{run}/traces/{instance_id}.json` (per-round proposer/critic generations
  + token counts). Keeps the main JSONL lean and uniform; analysis never needs it
  for primary numbers.
- `eval/results.py`:
  - one JSONL file **per shard**; `load_done(run_dir)` reads the **union of all
    `*.jsonl` under the run** and returns per-instance row counts, so changing
    shard count between runs never redoes work.
  - **resume predicate:** instance done under condition C when it has
    `≥ expected_attempts(C)` rows (1 for baseline/debate, N for MV); for MV, resume
    tops up the *missing* `sample_index`es (per-sample seed = f(instance_id,
    sample_index)).
  - **kill tolerance:** append + flush per row (lose at most the in-flight
    attempt); reader parses line-by-line and drops an unparseable trailing line.
- **Run identity + guard:** a run = one `out_dir` with a `manifest.json` (config
  snapshot, model, git commit, host/gpu — fine, `results/` is gitignored). On
  resume, assert the manifest model matches — catches mixing two models' rows into
  one accuracy.
- Done: run twice → second run writes nothing new; one valid JSONL line exists;
  a truncated trailing line is tolerated on reload.

### P2.3 — Config-driven runner over a slice (first real number)
Scale the single instance to a whole (task, encoding) over N graphs, from a file.
- `configs/baseline.toml`: `model`, `tasks`, `encodings`, `n_graphs`,
  `dataset_seed`, `max_new_tokens`, `out_path`. **Model id lives here only.**
- `eval/runner.py`: load config → `build_dataset` → filter to `not in load_done`
  → `run_instance` each → append. CLI `--config`, `--shard i/n` (shard by instance
  index; only one shard exercised now but P3 fans out with it).
- Report per-encoding accuracy from the JSONL at the end.
- Done: `edge_existence × adjacency`, N≈20 graphs, produces an accuracy + token
  totals; killing mid-run and resuming loses ≤1 instance.

### P2.4 — Generalize parser/scorer to all three tasks
Make P3's full matrix a config change. No new run yet — breadth + tests.
- Prompt templates for `node_degree` ("a single integer") and `connected_nodes`
  ("comma-separated node ids, or `none`").
- `scoring.parse` handles int and set shapes with per-task fallback regex; `score`
  uses `==` (set-equality for the list). Exact match only.
- Unit tests on **canned model outputs** (well-formed, messy-but-parseable, and
  unparseable → `parse_ok=False`) for all three tasks — the parser is the keystone,
  so it's tested independently of the GPU.

### P2.5 — Cluster pilot + slurm
Real numbers + prove the harness survives the killable partition.
- `slurm/p2-baseline.slurm` (adapt `smoke.slurm`): Qwen2.5-3B, `geforce_rtx_2080`,
  modest mem/time, runs the runner on `configs/baseline.toml`.
- Pilot: `edge_existence × adjacency`, small N, on the cluster.
- **Floor-effect sanity check:** confirm accuracy isn't rock-bottom-zero
  everywhere (a floor would threaten P3's "reproduce fragility" premise) — if it
  is, note it and flag the 4-bit-7B escalation for P3.
- Confirm token accounting is populated and resume works after an induced kill.

## Notes for later phases
- **P4 (majority-vote)** reuses P2.2's schema unchanged: N attempt rows per
  instance with `sample_index` 0..N-1 and `temperature>0`; the vote + individual-
  sample-vs-vote accuracy are derived at analysis. Resume tops up missing samples.
- **P5 (debate)** writes 1 attempt row per instance with `n_gen_tokens` summed
  across all Proposer+Critic calls (matched-compute), plus a per-round trace to
  the `traces/` sidecar; the `condition` field distinguishes rows.
- **P3** turns the single slice into the full 3×3 matrix purely via
  `configs/*.toml` + `--shard`; if P2.4's parser is right, no code changes.
- The `parse_ok` rate is a **reported diagnostic**, not just a debug aid — a high
  rate means the terse-format instruction is failing and the "wrong" numbers are
  really parse failures.
