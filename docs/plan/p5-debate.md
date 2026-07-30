# P5 — Debate condition (Proposer-Critic)

**Status: planned.** The third and final condition, and the project's keystone: does a
*verifying Critic* lift encoding-fragile accuracy beyond what aggregation buys? P4 gives
the control -- majority vote (pure extra compute) is statistically indistinguishable
from greedy -- so any debate gain is the procedure, not the sampling.

Goal: add the Proposer-Critic debate condition end-to-end (structured trace, a Critic
that verifies the trace, a revision loop with a stopping rule), on the **correct compute
metric** (# model responses, with total tokens alongside), plus a **trace viewer**, then
compare to baseline and MV **at matched compute**.

**Done when:** `results/main/debate/` holds one summed row per instance for all 9 cells
with a trace sidecar; the viewer renders any debate turn-by-turn; debate's per-cell
accuracy sits against baseline and MV's accuracy-vs-budget curve on **# responses** (and
total tokens); and the matched-compute verdict (does verification beat aggregation, on
the worst encodings?) is in [notes.md](../notes.md).

## Slice checklist

- [x] **P5.0** Baseline/MV compute measurement: total tokens + # responses, per-instance (analysis-only, no rerun)
- [ ] **P5.1** Proposer prompt + structured trace + parse (torch-free) — *prompt needs approval*
- [ ] **P5.2** Critic (holistic trace verification) + turn loop + stopping rule + per-turn logging + persistence — *prompt needs approval*
- [ ] **P5.3** Trace viewer (`scripts/debate_viewer.py`, self-contained HTML)
- [ ] **P5.4** Pilot (node_degree) → full 3×3 → matched-compute checkpoint

Resolved (see Decisions): **compute = # responses primary + total tokens secondary**
(generated-only dropped); **Critic is holistic** (one call per turn, no per-edge checks,
so batching/scope are moot).

## The design

- **Turn 1 — Proposer:** emits a **structured trace** (its supporting edge claims) + a
  final answer, in one call.
- **Turn 2 — Critic:** reads the trace, **verifies its claims against the raw encoding**,
  and returns a critique (which claims are wrong or missing) or "no issues" -- one call.
- Repeat Proposer↔Critic until **consensus** (Critic verdict `AGREE`), **no progress**
  (Proposer repeats a previous answer), or the **response budget** (MV's N=10) is hit.
  Final = the Proposer's latest answer. (Full rule in Decisions.)

Every turn is **one full model response**, which is what makes the compute comparison
clean (see Compute). The Critic is *holistic* (not per-edge), but is **prompted to verify
the structured trace's claims against the encoding**, retaining the proposal's
"verify, do not re-derive" essence -- a holistic pass over the encoding + claims can
catch both wrong and missing edges in one call.

**Every task's trace is edge claims:** `node_degree(n)`/`connected_nodes(n)` -> the query
node's incident edges `{(n,x)}` (degree = count, neighbors = the set); `edge_existence` ->
the single edge. **Pilot: node_degree** (biggest fragility gap, 0.38; adjacency 0.37 vs
incident 0.75, the most room to lift the worst encoding).

## Decisions locked

- **Structured trace:** the Proposer emits its answer + the supporting edge claims (the
  neighbor set for node_degree/connected_nodes).
- **Critic:** holistic per-turn critique, prompted to verify the trace's claims against
  the raw encoding. **Greedy.** Same loaded model with a verification-role prompt (11GB
  VRAM allows one model).
- **Stopping rule** (stop on whichever fires first, final = last Proposer answer):
  - **consensus** -- the Critic emits a structured verdict `AGREE`. An *unparseable*
    verdict is treated as AGREE **but counted** (a `critic_verdict_parsed` flag per Critic
    turn; the per-cell rate is reported, so "fake consensus" from a muddled Critic is
    visible, mirroring `parse_ok`).
  - **no progress** -- the Proposer repeats any previous answer (catches no-change *and*
    oscillation A->B->A).
  - **budget** -- reaches **MV's N = 10 responses/instance** (the same budget MV gets, so
    debate can never win by making more calls). The loop ends on a Proposer answer, never a
    dangling Critic call. Config-tied to `n_samples`.
  - Debate's *actual* per-instance response count (usually < 10) places it as a point on
    MV's accuracy-vs-budget curve; the cap just bounds it to the curve's range. At matched
    # responses, debate's total tokens still exceed MV's (transcript growth) -- reported.
- **Compute (corrected).** **Primary: # model responses = # turns** (the literature
  standard: Huang 2024 "equivalent number of responses", Choi 2025 "number of agents",
  Du 2023 "3 responses"). **Secondary: total tokens (prompt + generated)** summed over
  turns. **Generated-only is dropped** -- for these tasks the prompt dominates (123 vs 2
  generated), so it measured ~2% of the work. Debate's transcript grows each turn
  (re-reads the encoding + history), so its total-tokens/response exceeds MV's; # responses
  hides that, total tokens surfaces it -- report both.
- **Reporting unit: per instance (mean).** Tables show the mean cost of *one* instance --
  responses/instance (baseline 1, MV 10, debate ~T) and total-tokens/instance -- not the
  cell grand total (which just scales with N=200). Cell totals stay in the CSV. The MV vs
  baseline multiplier is identical either way. For debate, tokens/instance vary with the
  number of turns, so the mean summarizes a spread (distribution shown later if useful).
- **Matched compute (P6, no rerun):** MV's accuracy-vs-budget curve, indexed by **#
  samples (= # responses)** and by total tokens (subsample the 10 stored draws). Debate is
  a point on that curve at its # responses / total tokens. Paired McNemar via
  `stats.mcnemar_from_bc`.
- **Persistence:** one lean row per instance (final answer, summed total tokens,
  `n_responses`, `sample_index=0`) + the verbose per-turn transcript in a **trace
  sidecar**. Resume = 1 row.
- **Prompts need approval.** The Proposer and Critic prompts are **raised for your
  sign-off before finalizing** (in P5.1 / P5.2). They are the experiment's core.

## What's reused vs new

Reused **unchanged:** results schema + resume (`expected_attempts("debate")==1`), runner
shard/dispatch, `model.generate`, `scoring`, `show_results` split-by-condition,
`stats.mcnemar_from_bc`, `slurm/matrix.slurm` (CONFIG).

New: total-token + response-count reporting (P5.0), the Proposer prompt + trace parse, the
Critic prompt + holistic verification, the turn loop + stopping rule + per-turn logging,
the trace sidecar, runner dispatch for debate, `scripts/debate_viewer.py`, and the
matched-compute (MV-curve) analysis.

## Vertical slices

### P5.0 — Baseline/MV compute measurement (analysis-only, NO rerun)
Get baseline + MV onto the correct metric before debate lands, from stored data
(`n_prompt_tokens` is populated; total tokens + # responses are recoverable):
- `report`: add `total_tokens` (prompt+gen) + `n_responses` to `summarize` /
  `summarize_votes`; derive the **per-instance means** (`tokens_per_instance`,
  `responses_per_instance`) for the tables. Keep `total_gen_tokens` + cell totals for reference.
- Console tables + `--compare` lead with **per-instance** total-tokens + responses (and
  their MV/baseline multipliers); CSVs keep both per-instance and cell-total columns.
- `notes.md`: record that prompt dominates (123 vs 2), so generated-only measured ~2% of
  compute; MV's ~10x multiplier holds in total tokens too (MV = N independent copies).
- Tests; regenerate `analysis/`. No cluster time.

### P5.1 — Proposer + structured trace + parse (torch-free; PROMPT NEEDS APPROVAL)
- **Proposer prompt** (raise for approval): answer node_degree + emit the structured trace
  (the supporting neighbor/edge claims) in a parseable format.
- **Parse:** proposer output -> `(answer, claims, raw)`.
- **Trace sidecar schema:** per instance a `turns[]`; per turn `{role, raw, parsed,
  n_prompt_tokens, n_gen_tokens}` + `dump`/`load`. Drives the viewer + the compute totals.
- Tests: parse on canned outputs; schema round-trip. Stub model.

### P5.2 — Critic + turn loop + persistence (torch-free loop; PROMPT NEEDS APPROVAL)
- **Critic prompt** (raise for approval): given the encoding + the Proposer's trace, verify
  the claims and return the critique (wrong/missing claims) or "no issues".
- **Loop:** Proposer -> Critic -> revise, to consensus / no-progress / response budget; log
  each turn's tokens + `critic_verdict_parsed`.
- `run_debate(model, instance, cfg) -> (record, trace)`: record is baseline's shape with
  `condition="debate"`, summed total tokens, and `n_responses`; trace is the transcript.
- **Persistence:** 1 row via `make_row` (+ responses/tokens); trace to the sidecar. Runner
  dispatch on `condition=="debate"`; resume by the 1-row rule.
- Tests: converges on a correct proposer (AGREE); revises on a wrong claim; stops on a
  repeated answer (no-progress); hits the response budget; unparseable verdict counts +
  stops; token + response sums. Scripted stub model.

### P5.3 — Debate viewer (self-contained static HTML)
- `scripts/debate_viewer.py`: read debate rows + trace sidecar -> one standalone HTML
  (embedded JSON + inline CSS/JS), opened locally, offline, theme-aware:
  - left: filter/search instances (task, encoding, correct/incorrect, # turns, id);
  - right: turn-by-turn -- question + gold + final (✓/✗), Proposer trace + answer, Critic
    critique, revisions, per-turn + total tokens, # responses.
- Tests: generator emits valid HTML with the embedded data for a synthetic trace.

### P5.4 — Pilot -> full run -> matched-compute checkpoint
- **Pilot** node_degree × one encoding (few graphs) on GPU: confirm loop + trace + viewer
  end to end; watch turns + tokens.
- `configs/debate.toml`; reuse `slurm/matrix.slurm` via CONFIG. Full 3×3, resumable.
- **Analysis:** debate accuracy + # responses + total tokens per cell; MV
  accuracy-vs-budget curve; is debate above MV's curve at matched # responses (and total
  tokens) on the worst encodings? paired McNemar. Record in [notes.md](../notes.md) -- the
  project's headline result.

## Notes for later phases
- P6 formalizes the matched-compute figures on # responses and total tokens; the MV-curve
  trick means no MV rerun.
- The prompt-token dominance (123:2) is why total tokens matters and generated-only was
  misleading; # responses is the literature-standard axis.
