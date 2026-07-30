# P5 — Debate condition (Proposer-Critic)

**Status: planned.** The third and final condition, and the project's keystone: does
*active verification* (not mere aggregation) lift encoding-fragile accuracy? P4 gives
the control -- majority vote (pure extra compute) is statistically indistinguishable
from greedy -- so any debate gain is the procedure, not the sampling.

Goal: add the Proposer-Critic debate condition end-to-end (structured trace, per-claim
verification reusing `edge_existence`, revision loop with a stopping rule, full token
accounting), plus a **trace viewer** for inspecting individual debates, then compare to
baseline and to MV **at matched generated-token compute**.

**Done when:** `results/main/debate/` holds one summed-token row per instance for all 9
cells with a trace sidecar; the viewer renders any debate round-by-round; debate's
per-cell accuracy + token cost sit against baseline and MV's accuracy-vs-budget curve;
and the matched-compute verdict (does verification beat aggregation, on the worst
encodings?) is recorded in [notes.md](../notes.md).

## The idea (proposal §2.1-2.2)

The Proposer answers a task and emits its **atomic edge-existence claims**; the Critic
verifies each claim *against the raw encoding* (a strictly narrower task than solving);
refuted/missing claims go back to the Proposer, which revises; the loop runs to
consensus or a cap. The bet: **`edge_existence` is the least encoding-fragile task**
(P3: ~0.02 spread), so decomposing a fragile task into edge-checks should be more
encoding-robust than solving it holistically. That is exactly the verification
asymmetry the method assumes, and graphs make it directly measurable.

**Every task decomposes to edge claims:**
- `edge_existence(u,v)` -> the answer *is* one edge claim (near-degenerate; a weak test).
- `node_degree(n)` -> claims = the query node's incident edges `{(n,x)}`; degree = count of verified ones.
- `connected_nodes(n)` -> the same incident edges, reported as a set.

So node_degree and connected_nodes share one debate machine (verify node n's
neighborhood); edge_existence is the trivial case. **Pilot: node_degree** (biggest
fragility gap, 0.38; adjacency 0.37 vs incident 0.75, the most room to lift the worst).

## Decisions locked

- **Critic = `edge_existence` reused.** Each edge-check is an `edge_existence` query
  against the instance's encoding, reusing the existing prompt + parser. **Greedy.**
- **One model, two role prompts.** The Critic is the same loaded model with a
  verification prompt (11GB VRAM allows only one model).
- **Stopping rule: consensus-primary + backstop.** Stop when the Critic refutes nothing
  (consensus), OR after **R = 2** Critic->revise rounds, OR on a **fixed point** (a
  revision that does not change the trace). **No live token-budget throttle** in P5 --
  tokens are measured; matched compute is settled in P6 (below).
- **Token accounting.** Sum *every* generation: Proposer + each Critic edge-check + each
  revision. Persist **one lean row per instance** (final answer, summed tokens,
  `sample_index=0`) via the existing schema; the verbose transcript goes to a **trace
  sidecar** (the persistence contract already reserves this). Resume = 1 row done.
- **Matched compute (P6, no rerun).** Compare debate's per-cell `(tokens, accuracy)`
  point against **MV's accuracy-vs-budget curve** -- vote the first k of MV's 10 stored
  draws (k=1..10) for accuracy at 1x..10x budget -- plus baseline (1x). Headline: is
  debate above MV's curve at matched tokens on the worst encodings? Paired McNemar
  (debate vs baseline, debate vs MV@budget) reuses `stats.mcnemar_from_bc`.

## Decision to finalize in P5.1 (flagged)

- **Critic scope for node_degree / connected_nodes:** does the Critic check only the
  Proposer's *claimed* edges (cheap ~deg checks, catches over-claims only), or **sweep
  all candidate incident edges** `(n,x)` (≤ n-1 checks, catches over- **and**
  under-claims)? The full sweep is the method's essence -- it decomposes the fragile
  task into edge-checks and is where the encoding-robustness (and most of the token
  cost) comes from -- so we lean **full sweep**, but confirm on the pilot since it drives
  debate's compute (potentially > MV's 10x).

## What's reused vs new

Reused **unchanged:** results schema + resume (`expected_attempts("debate")==1`), runner
shard/dispatch pattern, `model.generate`, `edge_existence` prompt + parser (as the
Critic), `scoring`, `show_results` split-by-condition, `slurm/matrix.slurm` (CONFIG env),
`stats.mcnemar_from_bc`.

New: the Proposer prompt + claim extraction, the Critic wrapper, the debate loop +
stopping rule, the trace sidecar (schema + writer + reader), runner dispatch for debate,
`scripts/debate_viewer.py`, and the matched-compute (MV-curve) analysis.

## Vertical slices

### P5.1 — Proposer + claim extraction + trace schema (torch-free, tested)
- **Proposer prompt:** answer node_degree AND list the incident edges counted (the
  claims), in a parseable structured format.
- **Claim extraction:** parse the Proposer output -> `(answer, [edge claims])`.
- **Trace sidecar schema:** per instance a `rounds[]`; per round `{proposer_raw, claims,
  critic_verdicts, revised, tokens}` + `dump`/`load`. Drives the viewer.
- Tests: extraction on canned outputs, schema round-trip. Stub model.

### P5.2 — Critic + debate loop + persistence (torch-free loop, stub-tested)
- **Critic:** per candidate edge, an `edge_existence` query -> present/absent; assemble
  refuted + missing vs the Proposer's claims (scope per the P5.1 decision).
- **Loop:** propose -> critic -> revise, to consensus / R / fixed-point; sum tokens.
- `run_debate(model, instance, cfg) -> (record, trace)`: record is baseline's shape with
  `condition="debate"` + summed tokens; trace is the transcript.
- **Persistence:** 1 row via `make_row`; trace appended to the sidecar. Runner dispatch
  on `condition=="debate"`; resume by the 1-row rule.
- Tests: converges on a correct proposer; revises on a wrong claim; hits the R cap;
  fixed-point halt; token sum. Scripted stub model.

### P5.3 — Debate viewer (self-contained static HTML)
- `scripts/debate_viewer.py`: read debate rows + trace sidecar -> one standalone HTML
  (embedded JSON + inline CSS/JS), opened locally, offline, theme-aware:
  - left: filter/search instances (task, encoding, correct/incorrect, #rounds, id);
  - right: round-by-round -- question + gold + final (✓/✗), Proposer answer + claims,
    Critic per-claim verdicts (refuted highlighted), the revision, per-turn + total tokens.
- Tests: generator emits valid HTML with the embedded data for a synthetic trace.

### P5.4 — Pilot -> full run -> matched-compute checkpoint
- **Pilot** node_degree × one encoding (few graphs) on GPU: confirm loop + trace + viewer
  end to end; watch rounds + tokens (confirm the Critic-scope decision).
- `configs/debate.toml`; reuse `slurm/matrix.slurm` via CONFIG. Full 3×3 (all tasks
  decompose), resumable, throttled like P3.
- **Analysis:** debate accuracy + tokens per cell; MV accuracy-vs-budget curve
  (subsample); is debate above MV's curve at matched tokens on the worst encodings?
  paired McNemar. Record in [notes.md](../notes.md) -- the project's headline result.

## Notes for later phases
- P6 formalizes the matched-compute figures; the MV-curve trick means **no MV rerun**.
- Debate's cost is variable per instance, which is why compute is measured in tokens
  (not rounds) and summed including the Critic's -- the yardstick is P4's MV spend.
