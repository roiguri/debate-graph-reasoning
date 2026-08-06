# Findings

Empirical results, in the order the project produced them. This is the **results** log:
what the runs actually showed, with the number and the test that backs it. It is kept
apart from [notes.md](notes.md), which holds reference facts and design *decisions*, and
from [plan/](plan/), which holds the steps. This file feeds the results section of the
writeup; notes.md feeds the methodology section.

Every number here regenerates from committed code over committed run outputs:

```bash
# baseline fragility + majority vote (seed 7 only) + debate, pooled over 3 seeds
python scripts/show_results.py results/main results/seed11 results/seed13 \
    results/v2-main results/v2-seed11 results/v2-seed13 \
    --prompt-version v2 --fragility --save analysis/pooled-v2
# the valid majority-vote comparison: seed 7 alone, where MV actually ran
python scripts/show_results.py results/main --compare --save analysis/main
# debate diagnostics (baseline dirs must be passed too, for the CoT split)
python scripts/debate_diagnostics.py results/main results/seed11 results/seed13 \
    results/v2-main results/v2-seed11 results/v2-seed13 \
    --prompt-version v2 --save analysis/pooled-v2
# upper bound on any stopping rule over the transcripts that occurred
python scripts/oracle_ceiling.py results/v2-main results/v2-seed11 results/v2-seed13 \
    --prompt-version v2
```

Unless a line says otherwise, numbers are **pooled over seeds 7/11/13, n=600 per cell**,
from `analysis/pooled-v2/`. Encodings are applied to the same graphs, so the tests are
paired (McNemar, Cochran's Q); see notes.md.

**There is one debate arm, Proposer prompt v2.** The earlier v1 arm and a failed later
revision were deleted with the code that produced them; see section 7 for what they were
and why their numbers are not in this file.

---

## 1. Encoding fragility reproduces (P3)

The premise the project rests on holds. Baseline accuracy swings sharply with the
encoding on two of three tasks:

| task            | adjacency | incident  | friendship | max-min | Cochran Q (df=2) | best-vs-worst McNemar |
|-----------------|-----------|-----------|------------|---------|------------------|-----------------------|
| node_degree     |     0.388 | **0.750** |      0.458 |   0.362 | 200.2, p=3e-44   | 258/41, p=8e-36       |
| connected_nodes |     0.260 | **0.343** |      0.217 |   0.127 | 65.8, p=5e-15    | 90/14, p=2e-13        |
| edge_existence  | **0.703** |     0.690 |      0.695 |   0.013 | 0.52, p=0.77 ns  | 64/56, p=0.52 ns      |

Direction replicates Fatemi et al.: **incident is best on both fragile tasks.**

**`edge_existence` is not encoding-fragile.** Its spread is 0.013, its omnibus test is
nowhere near significant, and the best/worst labels differ across seeds. Treating it as a
third fragile task would misstate the result: it is a *control* that happens to sit in the
matrix, and any intervention has almost no fragility to remove there. It is also the task
the Critic's atomic check reduces to, which matters in section 3.

## 2. Majority vote buys nothing (P4)

At 10 samples and 10x the tokens, self-consistency is statistically indistinguishable
from a single greedy answer in all 9 cells.

*Seed 7 only, n=200*: `majority_vote/` was only ever run on seed 7. Every delta is within
±0.010, discordance is 1 to 8 instances per cell, and every McNemar p is 0.625 or above.
Use `analysis/main/mv_vs_baseline.csv`, not the pooled one: MV exists only for seed 7, so
in a pooled run the delta column would compare 200 MV instances against 600 baseline ones.

The mechanism is structural, not incidental. Self-consistency's leverage comes from
marginalizing over *diverse reasoning paths* (Wang et al. 2023); the baseline emits a
direct terse answer, so there are no paths to marginalize, and a majority vote over
samples of a near-single-token answer converges to the argmax, which is exactly greedy.
This is the control the debate condition needs: extra compute alone does not move this
model on these tasks.

**Compute per instance** (responses / total tokens): baseline 1 / 374, majority vote
10 / 3,659 (10.0x), debate 2.9 / 1,805 (4.8x). Debate is strictly **cheaper** than the
vote it has to beat, so no budget-matching scheme rescues either.

## 3. Debate is bidirectional, not null (P5)

Mean accuracy over the nine cells is **0.496 for debate against 0.501 for the baseline**.
On the headline number debate does nothing, and that headline is misleading: per cell it
moves accuracy from -0.100 to +0.095 and **five of nine moves are significant**, three
losses and two gains. The near-zero mean is cancellation, not inertness.

| task/encoding              | baseline | debate | delta   | McNemar b/c | p          |
|----------------------------|----------|--------|---------|-------------|------------|
| connected_nodes/adjacency  |    0.260 |  0.160 | -0.100  | 100/40      | 6e-07 ***  |
| connected_nodes/incident   |    0.343 |  0.438 | +0.095  | 98/155      | 4e-04 ***  |
| edge_existence/adjacency   |    0.703 |  0.787 | +0.083  | 43/93       | 3e-05 ***  |
| connected_nodes/friendship |    0.217 |  0.140 | -0.077  | 95/49       | 2e-04 ***  |
| node_degree/friendship     |    0.458 |  0.412 | -0.047  | 116/88      | 0.059      |
| edge_existence/incident    |    0.690 |  0.730 | +0.040  | 74/98       | 0.079      |
| edge_existence/friendship  |    0.695 |  0.655 | -0.040  | 74/50       | 0.039 *    |
| node_degree/adjacency      |    0.388 |  0.417 | +0.028  | 87/104      | 0.247      |
| node_degree/incident       |    0.750 |  0.723 | -0.027  | 81/65       | 0.215      |

Against majority vote at seed 7 the picture is the same shape: debate wins
`connected_nodes/incident` (+0.123, p=0.010), loses `connected_nodes/adjacency` (-0.085,
p=0.045), and is non-significant in the other seven, at roughly a third of MV's responses.

### 3a. Debate makes encoding fragility worse

| task            | baseline std | debate std | baseline max-min | debate max-min | debate Q, p     |
|-----------------|--------------|------------|------------------|----------------|-----------------|
| connected_nodes |       0.0526 | **0.1362** |           0.1267 |     **0.2983** | 190.1, p=5e-42  |
| edge_existence  |       0.0055 | **0.0539** |           0.0133 |     **0.1317** | 31.5, p=1e-07   |
| node_degree     |       0.1566 |     0.1458 |           0.3617 |         0.3117 | 174.7, p=1e-38  |

`connected_nodes` spread more than doubles, because debate lifts incident by +0.095 while
dropping adjacency by -0.100 and friendship by -0.077. **The gains land in the encoding
that was already best.**

The sharpest form is `edge_existence`. At baseline it is encoding-insensitive (Q=0.52,
p=0.77). Under debate the same task becomes significantly encoding-sensitive (Q=31.5,
p=1e-07). Debate does not merely fail to remove fragility; on a task that had none, it
manufactures it.

### 3b. The Critic's verdict is worth almost nothing

Every verdict cross-tabbed against whether the Proposer answer *that verdict was judging*
was correct (pooled, 6,424 verdicts):

|                      | AGREE | REVISE |                            |
|----------------------|-------|--------|----------------------------|
| Proposer **correct** |  1274 |   1710 | false-alarm rate **0.573** |
| Proposer **wrong**   |  1391 |   2049 | detection rate **0.596**   |

chi2 = 3.36 (1 df, **p = 0.067, ns**), phi = +0.023, odds ratio 1.10. A REVISE moves
P(the answer is wrong) from the base rate 0.535 to 0.545. **Pooled, the verdict is
statistically independent of correctness.**

It is not uniformly at chance, and the per-cell pattern is the interesting part:

| cell                      | verdicts | FA (REVISE given correct) | detection | phi    | p      |
|---------------------------|----------|---------------------------|-----------|--------|--------|
| edge_existence/adjacency  |      761 | 0.705                     | 0.947     | +0.266 | 2e-13  |
| edge_existence/friendship |      790 | 0.816                     | 0.965     | +0.221 | 6e-10  |
| edge_existence/incident   |      757 | 0.807                     | 0.948     | +0.181 | 7e-07  |
| connected_nodes/incident  |      689 | 0.302                     | 0.464     | +0.161 | 3e-05  |
| node_degree/incident      |      615 | 0.138                     | 0.218     | +0.098 | 0.015  |
| node_degree/adjacency     |      681 | 0.489                     | 0.412     | **-0.076** | 0.048 |
| connected_nodes/adjacency |      741 | 0.664                     | 0.586     | -0.058 | 0.115  |

Three readings, and the third is the one that matters:

- **The failure is calibration, not ignorance.** On `edge_existence` the Critic detects
  95 to 96 percent of wrong answers, real discrimination, and simultaneously fires on 70
  to 82 percent of correct ones. It knows something and cannot express it as a decision.
- **In two cells it is worse than chance.** `node_degree/adjacency` is significantly
  *anti*-correlated with error (phi=-0.076, p=0.048, OR=0.73), and
  `connected_nodes/adjacency` trends the same way. A loop acting on those verdicts is
  actively misinformed.
- **Its flag rate is close to a per-cell constant, and it is set backwards.** Comparing how
  often it says REVISE against how often the Proposer was actually wrong:

| task/encoding              | REVISE rate | Proposer actually wrong | gap        |
|----------------------------|-------------|-------------------------|------------|
| edge_existence/incident    |       0.849 |                   0.303 | **+0.547** |
| edge_existence/adjacency   |       0.777 |                   0.297 | **+0.480** |
| edge_existence/friendship  |       0.875 |                   0.396 | **+0.478** |
| node_degree/friendship     |       0.483 |                   0.602 |     -0.119 |
| node_degree/incident       |       0.161 |                   0.283 |     -0.122 |
| node_degree/adjacency      |       0.442 |                   0.606 |     -0.164 |
| connected_nodes/incident   |       0.402 |                   0.620 |     -0.218 |
| connected_nodes/adjacency  |       0.598 |                   0.843 |     -0.246 |
| connected_nodes/friendship |       0.543 |                   0.878 | **-0.335** |

  On the three `edge_existence` cells the Proposer is wrong about 30 percent of the time
  and the Critic objects to about 80 percent of its answers. On the six others the
  Proposer is wrong 28 to 88 percent of the time and the Critic objects far less often.
  **It is most critical exactly where the Proposer is most reliable, and least critical
  where the Proposer is least reliable.** The high `edge_existence` "detection" figure in
  the table above is largely this: a near-constant REVISE catches most errors because it
  catches most of everything.

**Its evidence is often fabricated.** Each REVISE is supposed to quote an edge from the
graph. Resolving every cited pair back to node ids:

| task/encoding              | REVISEs | problems | real edge | hallucinated | no pair cited |
|----------------------------|---------|----------|-----------|--------------|---------------|
| node_degree/adjacency      |     301 |      326 |      0.88 |         0.10 |             5 |
| node_degree/friendship     |     328 |      374 |      0.75 |         0.14 |            41 |
| connected_nodes/adjacency  |     443 |      470 |      0.73 |     **0.21** |            27 |
| connected_nodes/friendship |     386 |      630 |      0.65 |         0.16 |           122 |
| connected_nodes/incident   |     277 |      306 |      0.55 |         0.20 |            76 |
| node_degree/incident       |      99 |      102 |      0.46 |         0.19 |            36 |
| edge_existence/adjacency   |     591 |      591 |      0.29 |     **0.26** |           264 |
| edge_existence/friendship  |     691 |      704 |      0.24 |         0.07 |           484 |
| edge_existence/incident    |     643 |      649 |      0.11 |         0.04 |           551 |

The one sub-task the whole design assumed was easy, checking whether a pair is in a list,
is where grounding fails most: on `edge_existence` the Critic mostly cites no pair at all
and writes prose instead (551 of 643 on incident), and where it does cite, a quarter of
the cited pairs on adjacency are edges that do not exist.

**The Proposer mostly ignores it.** After a REVISE the answer changes 15 to 44 percent of
the time, and the net effect of every revision in the study is **+74 corrections over
3,759 critiques**.

### 3c. The loop is a weak positive; the CoT scaffold carries the movement

The baseline is answer-only (about 5 generated tokens per instance) while the debate
Proposer writes a claim trace, so `debate vs baseline` conflates CoT with debate. Turn 1
of the debate trace *is* a single-turn CoT answer at the same decoding settings, which
separates them with no new runs:

| task/encoding              | baseline | turn-1 | final | CoT delta (p)       | loop delta (p)      |
|----------------------------|----------|--------|-------|---------------------|---------------------|
| connected_nodes/adjacency  |    0.260 |  0.142 | 0.160 | **-0.118 (6e-08)**  | +0.018 (0.145)      |
| connected_nodes/friendship |    0.217 |  0.127 | 0.140 | **-0.090 (2e-05)**  | +0.013 (0.216)      |
| connected_nodes/incident   |    0.343 |  0.378 | 0.438 | +0.035 (0.231)      | **+0.060 (6e-06)**  |
| edge_existence/adjacency   |    0.703 |  0.732 | 0.787 | +0.028 (0.284)      | **+0.055 (0.0092)** |
| edge_existence/friendship  |    0.695 |  0.653 | 0.655 | -0.042 (0.073)      | +0.002 (1.000)      |
| edge_existence/incident    |    0.690 |  0.767 | 0.730 | **+0.077 (0.0043)** | -0.037 (0.093)      |
| node_degree/adjacency      |    0.388 |  0.402 | 0.417 | +0.013 (0.612)      | +0.015 (0.223)      |
| node_degree/friendship     |    0.458 |  0.407 | 0.412 | **-0.052 (0.041)**  | +0.005 (0.775)      |
| node_degree/incident       |    0.750 |  0.732 | 0.723 | -0.018 (0.410)      | -0.008 (0.180)      |
| **mean**                   |  **0.501** | 0.482 | **0.496** | **-0.019**      | **+0.014**          |

- **The CoT step carries the movement, in both directions.** Four of nine are significant,
  three of them losses, and on the mean the claim-trace scaffold is a net *loss* of 0.019.
- **The loop is a weak positive**, not inert: significant in two of nine cells, both
  positive, worth +0.014 on the mean, and it never reverses the sign the scaffold set. It
  recovers part of what the scaffold costs and does not touch the spread.

So fragility amplification is a property of the **reasoning format**, not of the
verification procedure layered on top of it.

### 3d. What the scaffold actually changes is the model's answer prior

The single most explanatory result in the study. The claim-trace format does not make the
model reason better or worse; it shifts **which answer it is willing to commit to**, away
from negative and empty answers and towards positive, enumerated ones.

What each condition *answers*, regardless of correctness:

| measure (rate over all 1,800 instances)        | baseline | turn-1 (CoT) | debate final | gold  |
|------------------------------------------------|----------|--------------|--------------|-------|
| `edge_existence`: says **Yes**                 |    0.241 |    **0.562** |        0.382 | 0.508 |
| `connected_nodes`: answers the **empty list**  |    0.217 |    **0.009** |        0.037 | 0.115 |

The baseline is biased the *other* way: it says Yes at less than half the gold rate, and
over-produces the empty list (0.217 against a gold rate of 0.115). The scaffold does not correct that bias, it
overshoots past it — and on `connected_nodes` it essentially abolishes the empty answer,
from 217 in 1000 to 9 in 1000.

Accuracy follows the prior exactly. Splitting each task by whether its gold answer is the
"negative" one (turn-1 vs baseline, pooled):

| task            | bucket        |    n | baseline | turn-1 (CoT) | CoT delta  |
|-----------------|---------------|------|----------|--------------|------------|
| connected_nodes | gold **= {}** |  207 |    0.990 |    **0.077** | **-0.913** |
| connected_nodes | gold ≠ {}     | 1593 |    0.180 |        0.234 | **+0.053** |
| edge_existence  | gold **No**   |  885 |    0.963 |    **0.652** | **-0.311** |
| edge_existence  | gold Yes      |  915 |    0.438 |    **0.780** | **+0.342** |
| node_degree     | gold 0        |  168 |    0.625 |        0.583 |     -0.042 |
| node_degree     | gold >0       | 1632 |    0.523 |        0.506 |     -0.017 |

The two tasks with a clean negative answer show the same signature, large and opposite:
the scaffold is worth **+0.342** where the answer is Yes and **-0.311** where it is No,
**+0.053** on a real neighbour list and **-0.913** on an empty one. `node_degree` has no
such answer (0 is just another integer) and shows no such split.

**This means the headline accuracies are largely a coincidence of the gold prior.** The
baseline's 0.696 on `edge_existence` is not edge-existence competence: it is 0.963 on the
No half and 0.438 on the Yes half, of a set that is 50.8 percent Yes. Whether an
intervention "helps" on a cell depends mostly on which way that cell's gold prior points.

The mechanism is visible in the traces: having written claims that name nodes, the model
harvests those names into its answer. On `7/80` it reasons correctly and then contradicts
itself —

```
1. Robert is an end node.
2. There is no edge involving Robert.
3. ANSWER: James, John, Michael
```

— and on `7/56` it writes "John is not connected to any other node" and answers
"James, David".

**It also explains the graph-size dependence.** Negative answers are concentrated in small
graphs (`connected_nodes` gold is empty on 0.196 of 5-9-node graphs against 0.048 of
15-19-node ones), and that is exactly where debate loses:

| task            | 5-9 nodes            | 10-14                | 15-19                |
|-----------------|----------------------|----------------------|----------------------|
| connected_nodes | 0.422 → 0.327 (-0.095) | 0.225 → 0.232 (+0.007) | 0.143 → 0.162 (+0.020) |
| edge_existence  | 0.821 → 0.788 (-0.033) | 0.631 → 0.690 (+0.059) | 0.611 → 0.681 (+0.070) |
| node_degree     | 0.619 → 0.621 (+0.001) | 0.555 → 0.520 (-0.035) | 0.405 → 0.390 (-0.014) |

(baseline → debate). For `connected_nodes` the size effect *is* the empty-answer effect.
For `edge_existence` the No-rate is flat across sizes (0.52/0.51/0.44), so its size effect
is separate: the scaffold genuinely helps locate a pair in a long edge list.

Two consequences for how the rest of this file should be read. First, section 3a's
fragility amplification is **not** an artefact of this: on the non-empty
`connected_nodes` bucket alone the spread still widens, baseline max-min 0.141 to CoT
0.258. Second, the most promising untested intervention in the project is making the
scaffold able to answer "none" / "No" — worth 207 instances on `connected_nodes` and 885
on `edge_existence`, and it needs a prompt pilot, not more compute.

### 3e. Format compliance is good under v2, with one exception

Turn-1 Proposer compliance (cap = 256 new tokens): unparsed answers are below 2.4 percent
in eight of nine cells. The exception is **`edge_existence/friendship`: 53 of 600
truncated and 52 unparsed (0.087)**, the worst parse loss in the matrix, which plausibly
accounts for its -0.040. The token cap for that cell is the obvious suspect and is
untested.

### 3f. No stopping rule rescues the loop, and the oracle ceiling is small

Every candidate rule stops the loop earlier than it really did, so truncating the trace
and reading the answer standing at that point is an exact replay. None of the four
candidates (stop at turn 1, cap revisions at one, veto REVISEs with fabricated citations,
require a real cited edge) wins in every cell, and the two gates mostly *hurt*:
`gate_must_cite` costs -0.088 on `edge_existence/adjacency` (p=4e-07) and
`gate_hallucinated` -0.028 (p=0.010). The loop's damage is not concentrated in critiques
with fabricated evidence.

The ceiling question answers the rest. An **oracle** that stops at whichever Proposer turn
happened to be right, an upper bound on *any* rule over these transcripts:

| task/encoding              | turn-1 | actual | oracle | headroom |
|----------------------------|--------|--------|--------|----------|
| edge_existence/incident    |  0.767 |  0.730 |  0.880 |   +0.150 |
| edge_existence/friendship  |  0.653 |  0.655 |  0.795 |   +0.140 |
| edge_existence/adjacency   |  0.732 |  0.787 |  0.892 |   +0.105 |
| node_degree/friendship     |  0.407 |  0.412 |  0.450 |   +0.038 |
| connected_nodes/adjacency  |  0.142 |  0.160 |  0.192 |   +0.032 |
| connected_nodes/incident   |  0.378 |  0.438 |  0.467 |   +0.028 |
| node_degree/adjacency      |  0.402 |  0.417 |  0.447 |   +0.030 |
| connected_nodes/friendship |  0.127 |  0.140 |  0.160 |   +0.020 |
| node_degree/incident       |  0.732 |  0.723 |  0.735 |   +0.012 |
| **POOLED**                 |  0.482 |  0.496 |  0.557 |   +0.062 |

Total headroom is **6 accuracy points**, and it is distributed exactly the wrong way:
concentrated on `edge_existence`, the task that is not fragile, and worth +0.012 to +0.032
on the cells that are. It leaves `connected_nodes` spanning 0.160 to 0.467. No stopping
rule, not even a clairvoyant one, recovers the encoding gap from these transcripts.

## 4. Friendship fails differently from the integer encodings

Not just more often, differently in kind (turn-1 answers):

| task/encoding              | signal                                                        |
|----------------------------|---------------------------------------------------------------|
| node_degree/adjacency      | mean signed error **-0.96**, undercounts on 44 percent        |
| node_degree/incident       | mean signed error -0.09                                       |
| node_degree/friendship     | mean signed error **+0.15**, overcounts on 35 percent         |
| connected_nodes/adjacency  | mean Jaccard 0.502, 66 percent contain a non-neighbour        |
| connected_nodes/incident   | mean Jaccard 0.711, 45 percent contain a non-neighbour        |
| connected_nodes/friendship | mean Jaccard **0.534**, **76 percent** contain a non-neighbour|
| edge_existence/adjacency   | answers Yes 0.613 against a gold rate of 0.501 (+0.112)       |
| edge_existence/incident    | answers Yes 0.612 against a gold rate of 0.508 (+0.103)       |

Friendship makes the model hallucinate *extra* relations; adjacency makes it miss real
ones. Two candidate explanations, not yet separated: the social framing ("X and Y are
friends") invites transitive closure, or non-integer labels are simply harder to track.
The alphabetic-label encoding proposed in the plan doc's Tier 3 is the experiment that
separates them.

## 5. Power is not the limiting factor

At pooled n=600 the discordant-pair counts per cell run 124 to 253, which resolves an
effect of about 0.05. The condition effects that are near zero are near zero, not hidden
by noise. More seeds will not turn them significant.

**What replicates per seed, and what does not.** Each seed is an independent 200-graph
draw, so this separates the stable results from the pooled artefacts:

| seed | baseline mean | debate mean | delta   | connected_nodes spread, base → debate |
|------|---------------|-------------|---------|----------------------------------------|
| 7    |         0.486 |       0.498 | **+0.012** | 0.150 → 0.280                       |
| 11   |         0.501 |       0.503 | **+0.002** | 0.125 → 0.350                       |
| 13   |         0.516 |       0.487 | **-0.029** | 0.105 → 0.265                       |

- **The mean debate-vs-baseline delta does not have a stable sign.** It is +0.012, +0.002
  and -0.029 across the three seeds. The correct statement is that debate is
  **indistinguishable from the baseline on the mean**, not that it is below it; the pooled
  -0.005 is inside the seed-to-seed spread and must not be read as a small negative effect.
- **The fragility amplification replicates 3 for 3.** `connected_nodes` spread widens in
  every seed independently, and by a factor of 1.9 to 2.8. This is the result to lean on.

The contrast is the point: pooling is what makes the *per-cell* effects significant, but
the headline mean is a number that moves with the seed, while the spread effect is one
that does not.

## 6. What the writeup says

`docs/paper/main.tex` is the ACL writeup built on exactly these numbers. Its spine: the
verification asymmetry that debate assumes (Irving et al. 2018) is **absent** in a domain
constructed so that atomic verification is a string lookup, and the interventions that
raise mean accuracy raise the encoding spread with it.

## 7. Historical note: the two deleted prompt arms and the parser corrections

Two things happened that this file no longer carries numbers for, recorded here so the
gap is explained rather than silent.

**The v1 debate arm is gone.** `results/main`, `seed11` and `seed13` originally held a
debate arm produced by an earlier Proposer wording whose `connected_nodes` answer hint
said "a comma-separated list of node ids" for *every* encoding. Under friendship the graph
labels nodes with names, so the model obeyed the hint, answered in integers, and 64 of 600
answers were unparseable — which read in the results as evidence that friendship is hard
to reason over, when it was a label-space mismatch the baseline prompt never had. v2 fixed
it. The v1 wording was deleted from `prompts/debate.py` (a81a24a) and its rows from
`results/` (591a772), so its numbers cannot be regenerated and are not reported. A third
wording, meant to strengthen all three roles, measured significantly worse on five of nine
cells and drove Proposer capitulation from 0.63 to 0.96; it was deleted as a failed
iteration.

**Three parser defects, all in the `connected_nodes` path, were corrected after the first
version of this file was written** (73ba8e0), and every run dir was re-scored. This is why
`connected_nodes` numbers here differ from that earlier version while `edge_existence` and
`node_degree` are unchanged:

1. **A bare `ANSWER:` line was discarded.** The regex required a character after the
   colon, so the model complying with "…or none" fell through to the last-line fallback
   and scored a parse failure. This is what produced the earlier claim that "the model
   never answers none" — it answered none 55 times in v2 and the parser threw them away.
   Section 3d is the corrected finding, and it is the opposite: the *baseline* answers
   none nearly perfectly and the *scaffold* is what breaks.
2. **Echoing the queried node was silently forgiven, and not evenly.** The source was
   dropped from every answer, so "Robert, James" for a query about Robert scored as
   `[James]`. Gold contains the queried node in **0 of 1,800** instances (no self-loops),
   so those answers must score wrong. Debate echoes the source in 32.3 percent of
   `connected_nodes` answers against the baseline's 23.1 percent — the Critic quotes edges
   as pairs and the Proposer copies both endpoints — so the leniency was a **subsidy paid
   to the condition under test**, worth about 3.2 points of `connected_nodes` accuracy to
   the baseline and 7.2 to debate.
3. **Claim numbers were parsed as node ids.** Under adjacency/incident the labels are
   integers, so "5. The nodes connected to 3 are 0." parsed as `[0, 5]`.

Defects 1 and 3 raised accuracy when fixed; defect 2 lowered it and dominated. The net
effect was to remove an advantage debate was receiving from the scorer rather than from
the condition. Pooled, debate now reads 0.496 against the baseline's 0.501 — but see
section 5: that gap is smaller than the seed-to-seed spread, so the honest statement is
that the two are indistinguishable on the mean, not that debate is behind. **The parser is deliberately not versioned:** prompts are
versioned because they change what the model generates, whereas the parser only changes
how stored text is scored, so one rule applies uniformly and affected runs are re-scored
with `scripts/rescore.py`. That costs no GPU because raw output is persisted.
