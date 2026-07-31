# P5 follow-up — why debate showed no effect, and how to attack it

**Status: Tier 0 done.** The full 3x3 debate run finished and debate is statistically
indistinguishable from baseline and from majority vote in 8 of 9 cells (the exception is
`node_degree/adjacency`, +0.055 pooled, p=0.028). This doc records the *diagnosis* built
from the committed traces, and the tiered plan that follows from it.

**Primary numbers are pooled** over seeds 7/11/13 (n=600 per cell), from
`analysis/pooled/`. Where a single seed disagrees with the pool it is called out; seed 7
alone is in `analysis/main/`. Everything regenerates with no GPU time:

```bash
python scripts/debate_diagnostics.py results/main results/seed11 results/seed13 \
    --save analysis/pooled
```

## Summary

Three separable failures, in order of how much they cost:

1. **The Critic's verdict is worth almost nothing.** Pooled, a REVISE moves P(the answer
   is wrong) from 0.520 to 0.533 (odds ratio 1.13). It is not literally independent of
   correctness, but it is close enough that the loop cannot act on it, and 14 to 32
   percent of the edges it cites as evidence do not exist in the graph.
2. **The debate loop is net-harmful**, and it sits on top of a chain-of-thought effect
   that the current condition table attributes to nothing.
3. **A prompt-format degeneration** silently scores 12 to 13 percent of some cells wrong
   for a non-reasoning reason, concentrated on `friendship` and on `connected_nodes`.

## 1. The Critic's verdict is worth almost nothing

Every Critic verdict, cross-tabbed against whether the Proposer answer *that verdict was
judging* was correct (pooled, 6,553 verdicts):

|                      | AGREE | REVISE |                            |
|----------------------|-------|--------|----------------------------|
| Proposer **correct** |  1400 |   1746 | false-alarm rate **0.555** |
| Proposer **wrong**   |  1417 |   1990 | detection rate **0.584**   |

chi2 = 5.65 (1 df, p = 0.017), phi = +0.029, odds ratio 1.13. With N in the thousands
that clears p < 0.05, and it still means nothing in practice: a REVISE moves P(wrong)
from the base rate 0.520 to 0.533. The verdict is worth about one accuracy point of
information.

**It is not uniformly at chance, though.** Three cells carry real discrimination:

| cell                      | verdicts | FA (REVISE given correct) | detection | phi   | p       |
|---------------------------|----------|---------------------------|-----------|-------|---------|
| edge_existence/incident   |      825 | 0.820                     | 0.966     | +0.21 | 1.3e-09 |
| edge_existence/friendship |      708 | 0.706                     | 0.852     | +0.16 | 2.1e-05 |
| connected_nodes/incident  |      709 | 0.336                     | 0.462     | +0.13 | 7.8e-04 |
| connected_nodes/friendship|      745 | 0.514                     | 0.506     | -0.01 | 0.86    |
| node_degree/adjacency     |      716 | 0.452                     | 0.468     | +0.02 | 0.68    |

On `edge_existence/incident` the Critic detects a wrong answer 97 percent of the time,
but it also cries REVISE on 82 percent of *correct* answers. The discrimination is real
and drowned by the bias, which is a calibration problem, not an ignorance problem. That
distinction matters for Tier 1: a threshold or a gate can recover signal that the raw
verdict throws away.

**The evidence is often fabricated.** Each REVISE is supposed to quote an edge from the
graph. Resolving the node labels in every cited problem line back to ids and checking the
implied pair against `graph_edgelist`:

| task/encoding              | REVISEs | cited pair is a real edge | does not exist | no pair cited |
|----------------------------|---------|---------------------------|----------------|---------------|
| connected_nodes/adjacency  |     482 | 0.83                      | 0.14           |            18 |
| connected_nodes/friendship |     378 | 0.69                      | 0.17           |            70 |
| connected_nodes/incident   |     292 | 0.59                      | 0.17           |            71 |
| node_degree/adjacency      |     330 | 0.79                      | 0.20           |             4 |
| node_degree/friendship     |     280 | 0.58                      | 0.20           |            71 |
| node_degree/incident       |     101 | 0.43                      | **0.32**       |            30 |
| edge_existence/adjacency   |     619 | 0.39                      | 0.26           |           214 |
| edge_existence/friendship  |     534 | 0.13                      | 0.05           |           437 |
| edge_existence/incident    |     720 | 0.03                      | 0.01           |           693 |

On `edge_existence` the Critic almost never cites a pair at all, it writes prose. On the
incident-edge tasks, a fifth to a third of its cited evidence is an edge that is not in
the graph, despite the prompt saying the graph text is the only source of truth.

**The Proposer mostly ignores it anyway.** After a REVISE the Proposer changes its answer
only 1,178 of 3,734 times (32 percent). Pooled across cells the direction is roughly a
wash (`ok_to_bad` 394 vs `bad_to_ok` 366), but it is strongly cell-dependent:
`edge_existence/adjacency` nets +36 while `edge_existence/incident` nets -35.

Representative failure (`7/70/node_degree/friendship`, gold 4): the Proposer lists four
correct edges and answers 4; the Critic replies `VERDICT: REVISE / - There is no edge
between David and Robert in the graph`, which is false; the Proposer restates the same
four edges and answers 4 again. Two extra model calls, no information.

## 2. The loop is net-harmful and sits on a CoT effect

The baseline is answer-only (2 to 10 generated tokens per instance), while the debate
Proposer writes a claim trace. So `debate vs baseline` conflates chain-of-thought with
debate. Turn 1 of the debate trace **is** a single-turn CoT answer at the same decoding
settings, so the two separate with no new runs:

| task/encoding             | baseline | turn-1 | final | CoT delta (p)       | loop delta (p)      |
|---------------------------|----------|--------|-------|---------------------|---------------------|
| connected_nodes/adjacency |    0.280 |  0.285 | 0.270 | +0.005 (0.880)      | -0.015 (0.328)      |
| connected_nodes/friendship|    0.263 |  0.192 | 0.228 | **-0.072 (0.0009)** | **+0.037 (0.0009)** |
| connected_nodes/incident  |    0.373 |  0.452 | 0.402 | **+0.078 (0.0050)** | **-0.050 (2e-05)**  |
| edge_existence/adjacency  |    0.703 |  0.663 | 0.723 | -0.040 (0.087)      | **+0.060 (0.0025)** |
| edge_existence/friendship |    0.695 |  0.705 | 0.697 | +0.010 (0.722)      | -0.008 (0.688)      |
| edge_existence/incident   |    0.690 |  0.743 | 0.685 | **+0.053 (0.0499)** | **-0.058 (0.0144)** |
| node_degree/adjacency     |    0.388 |  0.440 | 0.443 | **+0.052 (0.0426)** | +0.003 (0.905)      |
| node_degree/friendship    |    0.458 |  0.438 | 0.430 | -0.020 (0.441)      | -0.008 (0.511)      |
| node_degree/incident      |    0.750 |  0.735 | 0.728 | -0.015 (0.515)      | -0.007 (0.344)      |

p-values are paired McNemar (exact binomial under 25 discordant pairs, else
continuity-corrected chi-square), the same test the rest of the analysis uses.

Two findings, both replicated across three seeds:

- **The CoT prompt is not uniformly good.** It gains +0.078 on `connected_nodes/incident`
  and +0.052 on `node_degree/adjacency`, and it *loses* -0.072 on
  `connected_nodes/friendship`. The claim-trace scaffold helps where the encoding is
  already tractable and hurts where it is not, which is the opposite of what a robustness
  intervention should do.
- **The loop then works against the CoT step** in the two cells where CoT gained most
  (`connected_nodes/incident` -0.050, `edge_existence/incident` -0.058). The two effects
  partly cancel, which is exactly why the headline `debate vs baseline` delta looked like
  nothing.

Seed 7 alone shows `connected_nodes/friendship` CoT delta at +0.000, not -0.072. The
pooled result is the trustworthy one; this is why the primary tables are pooled.

**Debate also makes fragility worse**, the opposite of the project's hypothesis
(`analysis/pooled/*_fragility.csv`):

| task            | baseline std | debate std | baseline max-min | debate max-min |
|-----------------|--------------|------------|------------------|----------------|
| connected_nodes |       0.0484 | **0.0739** |           0.1100 |     **0.1733** |
| edge_existence  |       0.0055 | **0.0160** |           0.0133 |     **0.0383** |
| node_degree     |       0.1566 |     0.1376 |           0.3617 |         0.2983 |

## 3. Format degeneration, concentrated on friendship and connected_nodes

Turn-1 Proposer format compliance (pooled, n=600, cap = 256 new tokens):

| task/encoding              | hit token cap | no `ANSWER:` line | unparsed (auto-wrong) |
|----------------------------|---------------|-------------------|-----------------------|
| connected_nodes/adjacency  |             7 |                57 |            6 (0.01)   |
| connected_nodes/friendship |        **86** |           **196** |       **78 (0.13)**   |
| connected_nodes/incident   |            29 |               112 |           16 (0.03)   |
| edge_existence/adjacency   |             9 |                78 |       **75 (0.12)**   |
| edge_existence/friendship  |             8 |                38 |           17 (0.03)   |
| edge_existence/incident    |             1 |                99 |       **69 (0.12)**   |
| node_degree/adjacency      |             1 |               149 |            0 (0.00)   |
| node_degree/friendship     |            22 |               255 |            0 (0.00)   |
| node_degree/incident       |             0 |               256 |            0 (0.00)   |

The failure mode is that the Proposer copies the format template literally and runs to
the cap:

```
16. <one atomic claim>
17. <one atomic claim>
18. <one atomic claim>
...
```

That is 13 percent of `connected_nodes/friendship` scored wrong for a prompt-formatting
reason, not a graph-reasoning reason. `node_degree` survives its very high
no-`ANSWER`-line count only because `scoring._parse_int` takes the last integer in the
text, which happens to recover the degree. That is luck, not design: the same fallback on
`connected_nodes` produces the 78 unparsed answers above.

(`has_answer_line` requires content after `ANSWER:`, matching what the parser actually
consumes, so a turn truncated mid-answer-line counts as missing it.)

## 4. Friendship fails differently from the integer encodings

Beyond the format loss, the *shape* of the error differs (turn-1 answers, pooled):

| task/encoding              | signal                                                        |
|----------------------------|---------------------------------------------------------------|
| node_degree/adjacency      | mean signed error **-1.16**, undercounts on 45 percent        |
| node_degree/incident       | mean signed error -0.12                                       |
| node_degree/friendship     | mean signed error **+0.88**, overcounts on 37 percent         |
| connected_nodes/adjacency  | mean Jaccard 0.601, 37 percent contain a non-neighbour        |
| connected_nodes/incident   | mean Jaccard 0.770, 30 percent contain a non-neighbour        |
| connected_nodes/friendship | mean Jaccard **0.556**, **62 percent** contain a non-neighbour|
| edge_existence/incident    | answers Yes 0.595 against a gold rate of 0.480 (**+0.115**)   |
| edge_existence/adjacency   | answers Yes 0.442 against a gold rate of 0.470 (-0.029)       |

Friendship makes the model hallucinate *extra* relations; adjacency makes it miss real
ones. Two candidate explanations, not yet separated: the social framing ("X and Y are
friends") invites transitive closure, or non-integer labels are simply harder to track.

## Power is not the problem

At pooled n=600 the discordant-pair counts per cell are 150 to 250, which resolves an
effect of roughly 0.05. The observed effects are near zero, not hidden by noise. Do not
spend GPU on more seeds hoping for significance.

## Plan of attack

### Tier 0 — analysis only, no GPU  ✅

Every table above regenerates from a committed entry point.

- [x] `src/gedebate/eval/diagnostics.py` + `scripts/debate_diagnostics.py`, producing per
      cell and pooled: turn split (CoT vs loop, paired McNemar), Critic confusion
      (false-alarm/detection, chi2/phi via the new `stats.chi2_2x2`), Critic evidence
      grounding, revision transitions, format compliance, error shape, and the Tier-1a
      stopping-rule replay
- [x] Artifacts written to `analysis/main/` and `analysis/pooled/`
      (`debate_turn_split.csv`, `debate_critic_confusion.csv`,
      `debate_critic_grounding.csv`, `debate_revision_effect.csv`,
      `debate_compliance.csv`, `debate_error_shape.csv`,
      `debate_stopping_rules.csv`)
- [x] `tests/eval/test_diagnostics.py` covering the counting rules against hand-built
      traces (a verdict is attributed to the answer it judged, not the final one)
- [ ] Record the headline numbers in `docs/notes.md`

### Tier 1 — cheapest-first, and one bundled re-run

Ordering matters here: **changing the Proposer prompt invalidates the whole existing
debate run**, so the prompt fix and whatever stopping rule we adopt must ship as a single
re-run, not three.

- [x] **1a. Counterfactual stopping rules, replayed offline (no GPU).** Each candidate
      rule makes the loop stop *earlier* than it actually did, and a trace records every
      turn, so truncating it and re-reading the Proposer answer standing at that point is
      an exact replay rather than a simulation.
      (`analysis/*/debate_stopping_rules.csv`)

      **Result: evidence gating does not work.** Pooled accuracy delta against the run as
      it happened:

      | task/encoding              | turn1_only  | at_most_one | gate_hallucinated | gate_must_cite |
      |----------------------------|-------------|-------------|-------------------|----------------|
      | connected_nodes/adjacency  | +0.015      | +0.000      | +0.012 *          | +0.010         |
      | connected_nodes/friendship | **-0.037*** | +0.000      | +0.000            | -0.002         |
      | connected_nodes/incident   | **+0.050*** | +0.005      | +0.002            | +0.002         |
      | edge_existence/adjacency   | **-0.060**  | -0.002      | +0.013            | -0.007         |
      | edge_existence/friendship  | +0.008      | -0.002      | -0.010 *          | -0.010         |
      | edge_existence/incident    | **+0.058 ***| +0.000      | +0.000            | +0.055 *       |
      | node_degree/adjacency      | -0.003      | -0.003      | +0.002            | +0.002         |
      | node_degree/friendship     | +0.008      | +0.002      | +0.005            | -0.005         |
      | node_degree/incident       | +0.007      | +0.000      | +0.005            | +0.010 *       |

      Three conclusions, and they redirect the rest of Tier 1:

      1. **Neither gate is worth GPU time.** The largest gain is +0.013 (ns), and
         `gate_hallucinated` significantly *hurts* `edge_existence/friendship`. The
         loop's damage is not concentrated in critiques with fabricated citations, so
         groundedness is not the separating signal we hoped it was.
      2. **`gate_must_cite`'s one significant win is not gating, it is abstention.** On
         `edge_existence/incident` 693 of 725 cited problems name no pair at all, so the
         strict gate vetoes ~96 percent of REVISEs and collapses onto `turn1_only`
         (+0.055 vs +0.058). It is the same finding wearing a different hat.
      3. **Capping revisions at one changes essentially nothing** (max +0.005). The
         damage is done by the first revision, not accumulated over later ones.

      The only rule with real effect is *not running the loop*, and even that is
      cell-dependent: it gains +0.050 and +0.058 on the two `incident` cells and loses
      -0.037 and -0.060 on `connected_nodes/friendship` and `edge_existence/adjacency`.
      There is no fixed stopping rule that wins everywhere.

      *Limitation:* replay is only valid for rules that stop earlier. It cannot say what
      a *different* revision prompt, or a better Critic, would have produced. Those need
      GPU time, which is exactly what Tier 2 is for.
- [ ] **1b. Prompt-fix pilot (small GPU).** Drop the literal `<one atomic claim>`
      placeholder (the model copies it), give one worked example instead of a schema, add
      stop strings, set a per-task token cap, and make truncation a reported metric rather
      than a silent wrong answer. Pilot on the two worst cells only,
      `connected_nodes/friendship` and `edge_existence/incident`. Success criterion is
      mechanical, not accuracy: turn-1 unparsed drops from 12 to 13 percent toward zero.
- [ ] **1c. One bundled re-run (the real GPU cost).** Now just the prompt fix: 1a ruled
      the gating rules out, so there is nothing else to bundle. Its turn 1 *is* the CoT
      condition, so the missing control comes free.
- [ ] **CoT as a first-class arm (optional).** Debate turn-1 is greedy, same prompt, same
      model, so the numbers already exist in the traces. Run it as its own condition only
      if the writeup wants it as a named arm in the results tables; it is the cheapest run
      in the project (1 response per instance).

### Tier 2 — the actual research question

- [ ] **Asymmetric debate.** Keep the 3B Proposer, use a 7B or 14B Critic. The hypothesis
      requires verification to be easier than generation, and at 3B it is not. If a larger
      Critic is still near chance, that is a strong citable negative result against the
      debate literature's premise for structured-input tasks.
- [ ] **Per-claim verification.** Ask the Critic one binary question per atomic claim and
      aggregate in code, instead of re-solving the instance and emitting a global verdict.
      This reverses the "Critic is holistic" decision in [p5-debate.md](p5-debate.md);
      that decision was made for batching simplicity, and the null result is the reason to
      revisit it. The `edge_existence` cells are the evidence it might work: the Critic
      *does* discriminate there (detection 0.97 vs false-alarm 0.82), it just cannot turn
      that into a usable verdict.

### Tier 3 — the friendship question

- [ ] **Alphabetic-label encoding** (A, B, C with the same "connected to" phrasing) as a
      fourth encoding. If accuracy recovers, friendship's problem is non-integer labels
      and tokenization. If it stays low, the problem is the social framing inviting
      transitivity. Either answer is a result.
