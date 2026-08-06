# Findings — Llama-3.3-70B

Empirical results for the Llama-3.3-70B-Instruct arm. **This file is self-contained**: every
number in it comes from the runs in `results/llama70b-*`, and nothing here depends on, or
refers to, any other model's results. It is the results log for this arm — what the runs
showed, with the number and the test behind it.

Every number regenerates from committed code over committed run outputs:

```bash
# baseline + debate accuracy, per-encoding spread, paired significance
python scripts/show_results.py results/llama70b-main results/llama70b-seed11 \
    results/llama70b-seed13 --fragility --by-seed --save analysis/llama70b
# turn split, Critic behaviour, compliance, stopping rules
python scripts/debate_diagnostics.py results/llama70b-main results/llama70b-seed11 \
    results/llama70b-seed13 --save analysis/llama70b
```

**Setup.** `meta-llama/Llama-3.3-70B-Instruct-Turbo` served by Together.ai (`provider =
"together"`, FP8 endpoint), greedy decoding, Proposer prompt v2. Three independent
200-graph draws (seeds 7/11/13) over the frozen dataset artifacts, 3 tasks x 3 encodings,
**n=600 per cell**, 10,800 rows total. Encodings are applied to the same graphs, so all
tests are paired (McNemar, Cochran's Q). Baseline generation cap 256 tokens, debate cap
512. Configs: `configs/llama70b-{baseline,debate}-{main,seed11,seed13}.toml`.

**Conditions.** *Baseline* is one zero-shot answer-only response. *Debate* is a
Proposer–Critic loop: the Proposer emits a numbered claim trace plus a final answer, the
Critic verifies it against the encoding and returns AGREE/REVISE, the Proposer revises.
Debate turn 1 is a single-turn chain-of-thought answer at the same decoding settings,
which lets the scaffold effect and the loop effect be separated with no extra runs.

**Not yet run: majority vote.** There is no compute control in this arm. See section 7 —
this is the single largest gap in what follows.

---

## 1. Encoding fragility is large and highly significant

Baseline accuracy swings sharply with the serialization on all three tasks:

| task            | adjacency | incident  | friendship | max−min | Cochran Q (df=2) | best-vs-worst McNemar |
|-----------------|-----------|-----------|------------|---------|------------------|-----------------------|
| connected_nodes |     0.743 | **0.958** |      0.488 | **0.470** | 377.2, p=1e-82 | 293/11, p=2e-58     |
| node_degree     |     0.655 | **0.873** |      0.637 |   0.237 | 148.0, p=7e-33   | 170/28, p=1e-23     |
| edge_existence  |     0.938 | **0.980** |      0.867 |   0.113 | 80.7, p=3e-18    | 71/3, p=7e-15       |

**`incident` is best and `friendship` worst on all three tasks**, unanimously. The premise
the project rests on holds with room to spare: a `connected_nodes` question answered at
0.958 under one serialization drops to 0.488 under another, on the same graphs.

Unlike the other two, `edge_existence` starts near ceiling (0.867–0.980), so its 0.113
spread is compressed by the top of the scale. Its fragility is significant but has the
least room to move in either direction; keep that in mind reading sections 3 and 4.

## 2. Debate raises accuracy, and it replicates 3 for 3

Each seed is an independent 200-graph draw, so this separates a stable effect from a
pooled artefact:

| seed | baseline | debate | delta      |
|------|----------|--------|------------|
| 7    |    0.797 |  0.883 | **+0.086** |
| 11   |    0.790 |  0.884 | **+0.094** |
| 13   |    0.793 |  0.884 | **+0.091** |

The mean delta is **+0.090** and its three independent estimates agree to within 0.008.
This is not a pooling artefact and not a seed accident.

Per cell (paired McNemar over n=600, b = baseline right/debate wrong, c = the reverse):

| task/encoding              | baseline | debate | delta      | b/c     | p       |
|----------------------------|----------|--------|------------|---------|---------|
| connected_nodes/friendship |    0.488 |  0.787 | **+0.298** | 29/208  | 6e-31 ***|
| node_degree/friendship     |    0.637 |  0.898 | **+0.262** | 8/165   | 2e-32 ***|
| node_degree/adjacency      |    0.655 |  0.870 | **+0.215** | 8/137   | 2e-26 ***|
| connected_nodes/adjacency  |    0.743 |  0.857 | **+0.113** | 45/113  | 1e-07 ***|
| node_degree/incident       |    0.873 |  0.925 | **+0.052** | 15/46   | 1e-04 ***|
| edge_existence/incident    |    0.980 |  1.000 | +0.020     | 0/12    | 5e-04 ***|
| connected_nodes/incident   |    0.958 |  0.937 | −0.022     | 25/12   | 0.049 * |
| edge_existence/friendship  |    0.867 |  0.808 | **−0.058** | 60/25   | 2e-04 ***|
| edge_existence/adjacency   |    0.938 |  0.872 | **−0.067** | 58/18   | 8e-06 ***|

**Debate is not uniformly good.** It wins on six cells and loses on three, and the losses
are concentrated entirely in `edge_existence` — the task that was already near ceiling and
whose atomic check the Critic's own verification reduces to. The gains are concentrated in
the cells that started worst.

## 3. Debate reduces encoding fragility on both genuinely fragile tasks

This is the project's central question, and the answer differs by task.

| task            | baseline max−min | debate max−min | debate Q (df=2)  |
|-----------------|------------------|----------------|------------------|
| connected_nodes |            0.470 | **0.150**      | 84.5, p=4e-19    |
| node_degree     |            0.237 | **0.055**      | 16.0, p=3e-04    |
| edge_existence  |            0.113 | *0.192*        | 143.0, p=9e-32   |

Per seed, independently:

| task            | seed 7        | seed 11       | seed 13       |
|-----------------|---------------|---------------|---------------|
| connected_nodes | 0.495 → 0.195 | 0.490 → 0.140 | 0.425 → 0.115 |
| node_degree     | 0.225 → 0.065 | 0.260 → 0.085 | 0.225 → 0.030 |
| edge_existence  | 0.110 → 0.180 | 0.105 → 0.215 | 0.125 → 0.180 |

**The narrowing replicates 3 for 3 on `connected_nodes` and `node_degree`; the widening
replicates 3 for 3 on `edge_existence`.** Both directions are stable.

**The mechanism is that gains land where accuracy was worst.** `friendship`, the worst
encoding on every task, gains +0.298 and +0.262 on the two fragile tasks, while `incident`,
the best, moves −0.022 and +0.052. Debate pulls the floor up rather than the ceiling
higher.

**How much of this is a ceiling effect.** Some. `connected_nodes/incident` (0.958) and
`edge_existence/incident` (0.980) have almost no room to rise, so part of any gap-closing
is arithmetic rather than the intervention. Two observations argue the effect is not only
that. First, `node_degree` narrows from 0.237 to 0.055 while its best cell (0.873) is not
at ceiling and still gains +0.052. Second, the direction of the *residual* fragility
changes: under debate, `node_degree`'s worst encoding is no longer `friendship` but
`adjacency` (0.870 vs 0.898), so the ordering that held unanimously at baseline is broken
rather than merely compressed. **Separating ceiling from mechanism properly needs a harder
dataset**; it is not settled here.

**Fragility is never eliminated.** All three debate Q statistics remain highly significant.
The claim supported by these numbers is that debate *reduces* encoding sensitivity on
fragile tasks, not that it confers encoding invariance.

## 4. Both halves of the procedure contribute, in opposite directions per task

Debate turn 1 is a chain-of-thought answer under the same decoding, so the baseline →
turn-1 → final split separates the reasoning scaffold from the verify-and-revise loop:

| task/encoding              | baseline | turn-1 | final | CoT delta (p)        | loop delta (p)       |
|----------------------------|----------|--------|-------|----------------------|----------------------|
| node_degree/friendship     |    0.637 |  0.875 | 0.898 | **+0.238** (0.0000)  | **+0.023** (0.0001)  |
| connected_nodes/friendship |    0.488 |  0.717 | 0.787 | **+0.228** (0.0000)  | **+0.070** (0.0000)  |
| node_degree/adjacency      |    0.655 |  0.845 | 0.870 | **+0.190** (0.0000)  | **+0.025** (0.0119)  |
| node_degree/incident       |    0.873 |  0.938 | 0.925 | **+0.065** (0.0000)  | −0.013 (0.0386)      |
| connected_nodes/adjacency  |    0.743 |  0.807 | 0.857 | **+0.063** (0.0039)  | **+0.050** (0.0000)  |
| edge_existence/incident    |    0.980 |  1.000 | 1.000 | **+0.020** (0.0005)  | +0.000 (1.0000)      |
| connected_nodes/incident   |    0.958 |  0.940 | 0.937 | −0.018 (0.1093)      | −0.003 (0.7539)      |
| edge_existence/adjacency   |    0.938 |  0.913 | 0.872 | −0.025 (0.0499)      | **−0.042** (0.0003)  |
| edge_existence/friendship  |    0.867 |  0.833 | 0.808 | −0.033 (0.0272)      | **−0.025** (0.0026)  |

**The scaffold carries most of the effect.** The CoT step is worth +0.190 to +0.238 on the
three worst cells; the loop adds a further +0.023 to +0.070 there. Both are significant and
both point the same way on the fragile tasks.

**The loop is genuinely load-bearing, not decoration.** Replaying the traces under a
"stop at turn 1" rule costs −0.050 and −0.070 on the two `connected_nodes` cells that gain
most (both p<0.001) and −0.023 to −0.025 on the two `node_degree` gainers. Conversely, on
`edge_existence/adjacency` and `/friendship` stopping at turn 1 *gains* +0.042 and +0.025.
The loop helps where the task is hard and hurts where it is nearly solved.

**On `edge_existence`, both halves are harmful.** Every step of the procedure — writing a
claim trace, then verifying it — costs accuracy on a task the model already answers at
0.867–0.980 zero-shot. This is the clearest limit on the method found here.

## 5. The Critic carries real signal

Every verdict cross-tabbed against whether the Proposer answer *it was judging* was
correct (pooled, 5,677 verdicts):

|                      | AGREE | REVISE |                             |
|----------------------|-------|--------|-----------------------------|
| Proposer **correct** |  4618 |   243  | false-alarm rate **0.050**  |
| Proposer **wrong**   |   461 |   355  | detection rate **0.435**    |

chi2 = 1099.3 (1 df, p = 5e-241), **phi = +0.440**, odds ratio 14.6. A REVISE moves P(the
answer is wrong) from the base rate 0.144 to **0.594** — a fourfold shift in the posterior.
The verdict is a usable signal, and the loop's gains in section 4 are what that signal buys.

Per cell, the discrimination is strongest where the check is atomic:

| task/encoding              | verdicts | FA\|ok | det\|bad | phi    | unparsed |
|----------------------------|----------|--------|----------|--------|----------|
| edge_existence/adjacency   |      649 |  0.143 |    0.922 | +0.621 |        0 |
| edge_existence/friendship  |      624 |  0.107 |    0.733 | +0.589 |        0 |
| connected_nodes/adjacency  |      654 |  0.033 |    0.386 | +0.465 |       17 |
| node_degree/adjacency      |      642 |  0.023 |    0.339 | +0.448 |       15 |
| connected_nodes/friendship |      667 |  0.030 |    0.351 | +0.444 |       37 |
| node_degree/friendship     |      616 |  0.002 |    0.208 | +0.416 |       19 |
| connected_nodes/incident   |      612 |  0.032 |    0.163 | +0.169 |       62 |
| edge_existence/incident    |      601 |  0.057 |    1.000 | +0.164 |        0 |
| node_degree/incident       |      612 |  0.023 |    0.064 | +0.068 |  **169** |

The Critic is **conservative rather than trigger-happy**: false-alarm rates run 0.002–0.143.
That conservatism is why a REVISE is informative, and also why detection is only 0.435 —
it misses more than half of wrong answers rather than flagging everything.

**A REVISE improves the answer where the task is hard.** Net effect of a revision
(bad→ok minus ok→bad): `connected_nodes/friendship` **+42**, `/adjacency` **+30**,
`node_degree/adjacency` **+15**, `/friendship` **+14**. It is negative on
`edge_existence/adjacency` (**−25**) and `/friendship` (**−15**), which is the mechanism
behind those cells' negative loop deltas in section 4.

### 5a. The Critic's cited evidence is frequently fabricated

The Critic is instructed to quote an edge from the graph. Resolving every cited label pair
back to node ids:

| task/encoding              | REVISEs | real edge | hallucinated | no pair |
|----------------------------|---------|-----------|--------------|---------|
| connected_nodes/incident   |      25 | 181 (0.79)|    26 (0.11) |      21 |
| node_degree/incident       |      16 |  39 (0.72)|     2 (0.04) |      13 |
| node_degree/adjacency      |      50 |  59 (0.67)|    13 (0.15) |      16 |
| connected_nodes/adjacency  |      68 |  67 (0.48)|    28 (0.20) |      45 |
| connected_nodes/friendship |      82 | 105 (0.45)|    47 (0.20) |      80 |
| node_degree/friendship     |      17 |   7 (0.33)|     9 (0.43) |       5 |
| edge_existence/incident    |      35 |   5 (0.15)|    26 (0.76) |       3 |
| edge_existence/adjacency   |     163 |  16 (0.10)|   144 (0.89) |       1 |
| edge_existence/friendship  |     142 |   6 (0.04)|   134 (0.96) |       0 |

**On `edge_existence` the Critic is simultaneously the most accurate and the least
grounded**: detection 0.92–1.00 while 76–96 percent of its cited pairs are edges that do
not exist in the graph. Its verdicts are right; its stated reasons are not. This matters
for any claim that debate works *because* verification is grounded in the input — on this
task it demonstrably is not.

Consistent with that, evidence-gating does not help. Vetoing REVISEs whose citation is
fabricated changes nothing where debate is winning and *gains* +0.048 on
`edge_existence/adjacency` — i.e. the only thing gating achieves is partially switching off
a loop that was already harmful there.

## 6. Format compliance is good, with one cell that is not trustworthy

Turn-1 Proposer output parses in **5,387 of 5,400** instances (13 unparsed, all but one on
`edge_existence`), and Proposer truncation at the 512-token cap is at or below 1.6 percent
in every cell (worst: `edge_existence/adjacency`, 12 of 763 turns).

**The exception is Critic truncation on `node_degree/incident`: 157 of 612 turns (0.257),
producing 169 unparsed verdicts.** An unparseable verdict is defaulted to AGREE and
**terminates the loop**, so roughly a quarter of that cell's instances end in a consensus
manufactured by the token cap rather than reached by the Critic. That contaminates two
numbers, and they should not be used:

- its Critic statistics (phi = +0.068, p = 0.092), because defaulted verdicts are counted
  as AGREE in the confusion matrix;
- its loop delta (−0.013, p = 0.039), because the loop was cut short mechanically.

`connected_nodes/incident` has a milder form of the same problem (62 unparsed of 612, 0.10).
**Fixing this needs a rerun of those cells at a higher Critic cap**; it is not repairable
from the stored traces. Nothing else in this file depends on those two numbers — the
section 2 accuracy for `node_degree/incident` is unaffected, since the final answer is
scored whether or not the loop ended early.

## 7. What is missing: the compute control

Debate spends **2.16 responses and 1,663 tokens per instance against the baseline's 1.00
and 364** — roughly 4.6x the tokens. A majority-vote arm (N sampled answers aggregated by
vote) has **not been run** for this model.

Without it, the results in sections 2–4 cannot distinguish *debate* from *more compute
spent any way at all*. This is the first objection any reader will raise, and it is
currently unanswered. Self-consistency has a plausible structural reason to do nothing here
— the baseline emits a terse direct answer, so there are few diverse reasoning paths to
marginalize over — but that is an argument, not a measurement.

**This is the highest-priority remaining run.** Until it exists, the honest claim is
"debate at 4.6x compute beats a single greedy answer", not "debate beats matched compute."

## 8. Summary

1. Encoding fragility is large and significant on all three tasks; `incident` best and
   `friendship` worst, unanimously (section 1).
2. Debate improves mean accuracy by +0.090, replicating across three independent seeds
   with near-identical magnitude (section 2).
3. Debate **reduces** encoding fragility on both genuinely fragile tasks, 3 for 3, by
   lifting the worst encoding rather than the best — and **increases** it on
   `edge_existence`, 3 for 3 (section 3).
4. Most of the gain is the reasoning scaffold; the verify-and-revise loop adds a smaller
   but significant increment, and is actively harmful on `edge_existence` (section 4).
5. The Critic carries real signal (phi = +0.440) while frequently citing evidence that does
   not exist, most extremely on the task where its verdicts are most accurate (section 5).
6. One cell, `node_degree/incident`, has Critic truncation at 0.257 and its loop and Critic
   numbers must not be used (section 6).
7. No compute control has been run, which bounds what can be claimed (section 7).
