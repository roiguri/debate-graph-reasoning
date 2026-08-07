# Findings — Llama-3.3-70B

Empirical results for the Llama-3.3-70B-Instruct arm. **This file is self-contained**: every
number in it comes from the runs in `results/llama70b-*`, and nothing here depends on, or
refers to, any other model's results.

**The result of this arm is the Proposer prompt v3 run.** Sections 1–7 are v3 throughout.
An earlier prompt, v2, was also run over the identical instances; it is reported separately
in section 8 as a prompt ablation, and nothing in sections 1–7 depends on it.

Every number regenerates from committed code over committed run outputs:

```bash
# baseline + debate accuracy, per-encoding spread, paired significance
python scripts/show_results.py results/llama70b-main results/llama70b-seed11 \
    results/llama70b-seed13 results/llama70b-v3-main results/llama70b-v3-seed11 \
    results/llama70b-v3-seed13 --prompt-version v3 --fragility --by-seed \
    --save analysis/llama70b-v3
# turn split, Critic behaviour, compliance, stopping rules
python scripts/debate_diagnostics.py results/llama70b-main results/llama70b-seed11 \
    results/llama70b-seed13 results/llama70b-v3-main results/llama70b-v3-seed11 \
    results/llama70b-v3-seed13 --prompt-version v3 --save analysis/llama70b-v3
# the matched-compute vote arm (section 7)
python scripts/show_results.py results/llama70b-main results/llama70b-seed11 \
    results/llama70b-seed13 results/llama70b-v3-main results/llama70b-v3-seed11 \
    results/llama70b-v3-seed13 --prompt-version v3 \
    --vote-condition majority_vote_cot --compare --save analysis/llama70b-v3
```

`--prompt-version v3` is **required**, not optional: the run dirs hold debate rows under two
prompt versions and the analysis refuses to pool them (see section 9).

**Setup.** `meta-llama/Llama-3.3-70B-Instruct-Turbo` served by Together.ai (`provider =
"together"`, FP8 endpoint), greedy decoding, Proposer prompt **v3**. Three independent
200-graph draws (seeds 7/11/13) over the frozen dataset artifacts, 3 tasks x 3 encodings,
**n=600 per cell** — 5,400 baseline instances, 5,400 debate instances, and 16,200
majority-vote draws (3 per instance). Encodings are applied to the same graphs, so all
tests are paired (McNemar, Cochran's Q). Baseline generation cap 256 tokens, debate and
vote cap 512. Configs: `configs/llama70b-baseline-{main,seed11,seed13}.toml`,
`configs/llama70b-debate-v3-{main,seed11,seed13}.toml` and
`configs/llama70b-mvcot-{main,seed11,seed13}.toml`.

**Conditions.** *Baseline* is one zero-shot answer-only response. *Debate* is a
Proposer–Critic loop: the Proposer emits a numbered claim trace plus a final answer, the
Critic reviews it against the encoding and returns AGREE/REVISE, the Proposer revises.
*Majority vote* draws the **same Proposer prompt** N=3 times independently (temperature 0.6,
top_p 0.9 — the model's shipped defaults) and takes the mode of the parsed answers; it is
the matched-compute control, differing from debate only in whether the attempts see each
other. Debate turn 1 is a single-turn chain-of-thought answer at the same decoding settings,
which lets the scaffold effect and the loop effect be separated with no extra runs. The
baseline prompt is untouched by the prompt versioning, so the same baseline rows serve every
section.

**A note on `edge_existence`.** It is a single-pair lookup: the Proposer's trace reduces to
one atomic claim, so a claim-by-claim critique has nothing to work on. It also starts near
ceiling (0.867–0.980 at baseline). It behaves as a control rather than as a third fragile
task, and sections 2–7 report both the full 9-cell scope and the 6-cell scope excluding it.
Both are given everywhere; the exclusion is never applied silently.

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

Unlike the other two, `edge_existence` starts near ceiling, so its 0.113 spread is
compressed by the top of the scale.

## 2. Debate raises accuracy, and it replicates 3 for 3

Each seed is an independent 200-graph draw, so this separates a stable effect from a pooled
artefact:

| seed | baseline | debate | delta      | 6-cell baseline | 6-cell debate | 6-cell delta |
|------|----------|--------|------------|-----------------|---------------|--------------|
| 7    |    0.797 |  0.854 | +0.057     |           0.732 |         0.843 | **+0.112**   |
| 11   |    0.790 |  0.847 | +0.057     |           0.714 |         0.854 | **+0.140**   |
| 13   |    0.793 |  0.828 | +0.035     |           0.732 |         0.831 | **+0.099**   |

Pooled: **0.793 → 0.843, +0.050** over all nine cells (b/c = 351/619, p = 1e-17), and
**0.726 → 0.843, +0.117** over the six cells excluding `edge_existence` (b/c = 155/576,
p = 2e-54). The sign replicates 3 for 3 on both scopes; the magnitude varies more on the
9-cell scope (0.035–0.057) than the 6-cell one (0.099–0.140) because `edge_existence`'s
losses vary by seed.

Per cell (paired McNemar over n=600, b = baseline right/debate wrong, c = the reverse):

| task/encoding              | baseline | debate | delta      | b/c     | p        |
|----------------------------|----------|--------|------------|---------|----------|
| connected_nodes/friendship |    0.488 |  0.687 | **+0.198** | 38/157  | 1e-17 ***|
| node_degree/adjacency      |    0.655 |  0.850 | **+0.195** | 19/136  | 3e-21 ***|
| node_degree/friendship     |    0.637 |  0.828 | **+0.192** | 16/131  | 2e-21 ***|
| connected_nodes/adjacency  |    0.743 |  0.817 | **+0.073** | 43/87   | 2e-04 ***|
| node_degree/incident       |    0.873 |  0.935 | **+0.062** | 13/50   | 8e-07 ***|
| edge_existence/incident    |    0.980 |  0.975 | −0.005     | 13/10   | 0.678 ns |
| connected_nodes/incident   |    0.958 |  0.940 | −0.018     | 26/15   | 0.118 ns |
| edge_existence/friendship  |    0.867 |  0.752 | **−0.115** | 84/15   | 3e-12 ***|
| edge_existence/adjacency   |    0.938 |  0.803 | **−0.135** | 99/18   | 4e-14 ***|

**Debate is not uniformly good.** Five cells gain significantly, two lose significantly, two
are null — and **every significant loss is `edge_existence`**. The gains are concentrated in
the cells that started worst: the three largest gains are on the three lowest baseline cells.

## 3. Debate reduces encoding fragility on the fragile tasks

This is the project's central question. Reporting the spread at each stage separates the
scaffold's contribution from the loop's:

| task            | stage    | adjacency | incident | friendship | max−min | vs baseline |
|-----------------|----------|-----------|----------|------------|---------|-------------|
| connected_nodes | baseline |     0.743 |    0.958 |      0.488 |   0.470 |             |
|                 | turn 1   |     0.835 |    0.947 |      0.655 |   0.292 | −0.178      |
|                 | final    |     0.817 |    0.940 |      0.687 | **0.253** | **−0.217** |
| node_degree     | baseline |     0.655 |    0.873 |      0.637 |   0.237 |             |
|                 | turn 1   |     0.868 |    0.942 |      0.828 |   0.113 | −0.123      |
|                 | final    |     0.850 |    0.935 |      0.828 | **0.107** | **−0.130** |
| edge_existence  | baseline |     0.938 |    0.980 |      0.867 |   0.113 |             |
|                 | turn 1   |     0.922 |    0.978 |      0.800 |   0.178 | +0.065      |
|                 | final    |     0.803 |    0.975 |      0.752 | *0.223* | *+0.110*    |

**The narrowing is large on both genuinely fragile tasks and the widening is confined to
`edge_existence`**, which had the least fragility to remove and is not a debate-shaped task.

**The mechanism is that gains land where accuracy was worst.** On `connected_nodes` the
spread closes almost entirely by lifting `friendship` (0.488 → 0.687) while `incident`
drifts slightly down (0.958 → 0.940); on `node_degree`, `friendship` rises 0.637 → 0.828
against `incident`'s 0.873 → 0.935. Debate pulls the floor up rather than the ceiling
higher.

**Most of the narrowing is already present at turn 1** — 0.178 of 0.217 on
`connected_nodes` and 0.123 of 0.130 on `node_degree`, i.e. 82% and 95%. The loop adds a
further 0.039 and 0.006. Unlike accuracy (section 4), the loop's contribution to fragility
is at least consistently signed in the right direction on both tasks, but it is small.

Per seed, independently (baseline spread → debate spread):

| task            | seed 7        | seed 11       | seed 13       |
|-----------------|---------------|---------------|---------------|
| connected_nodes | 0.495 → 0.255 | 0.490 → 0.270 | 0.425 → 0.235 |
| node_degree     | 0.225 → 0.110 | 0.260 → 0.110 | 0.225 → 0.100 |
| edge_existence  | 0.110 → 0.155 | 0.105 → 0.260 | 0.125 → 0.255 |

**The narrowing replicates 3 for 3 on both fragile tasks, and the widening replicates 3 for
3 on `edge_existence`.** Both directions are stable, not seed artefacts.

**Fragility is never eliminated.** Under debate all three Cochran Q statistics remain highly
significant (164.3, 165.9, 46.1; all p < 1e-10) and `incident` > `friendship` still holds
unanimously. The supported claim is that debate *reduces* encoding sensitivity on fragile
tasks, not that it confers encoding invariance.

**How much is a ceiling effect.** Some. `connected_nodes/incident` (0.958) and
`edge_existence/incident` (0.980) have almost no room to rise, so part of any gap-closing is
arithmetic. `node_degree` argues against it being only that: it narrows 0.237 → 0.107 while
its best cell (0.873) is not at ceiling and still gains +0.062. **Separating ceiling from
mechanism properly needs a harder dataset**; it is not settled here.

## 4. The reasoning scaffold does the work; the loop does not

Debate turn 1 is a chain-of-thought answer under the same decoding, so the baseline →
turn-1 → final split separates the reasoning scaffold from the verify-and-revise loop:

| task/encoding              | baseline | turn-1 | final | CoT delta (p)        | loop delta (p)       | turns |
|----------------------------|----------|--------|-------|----------------------|----------------------|-------|
| node_degree/adjacency      |    0.655 |  0.868 | 0.850 | **+0.213** (0.0000)  | −0.018 (0.0543)      | 2.22  |
| node_degree/friendship     |    0.637 |  0.828 | 0.828 | **+0.192** (0.0000)  | +0.000 (1.0000)      | 2.14  |
| connected_nodes/friendship |    0.488 |  0.655 | 0.687 | **+0.167** (0.0000)  | **+0.032** (0.0017)  | 2.29  |
| connected_nodes/adjacency  |    0.743 |  0.835 | 0.817 | **+0.092** (0.0000)  | −0.018 (0.1696)      | 2.43  |
| node_degree/incident       |    0.873 |  0.942 | 0.935 | **+0.068** (0.0000)  | −0.007 (0.4240)      | 2.12  |
| edge_existence/incident    |    0.980 |  0.978 | 0.975 | −0.002 (1.0000)      | −0.003 (0.6875)      | 2.15  |
| connected_nodes/incident   |    0.958 |  0.947 | 0.940 | −0.012 (0.3105)      | −0.007 (0.5235)      | 2.17  |
| edge_existence/adjacency   |    0.938 |  0.922 | 0.803 | −0.017 (0.1649)      | **−0.118** (0.0000)  | 2.66  |
| edge_existence/friendship  |    0.867 |  0.800 | 0.752 | **−0.067** (0.0000)  | **−0.048** (0.0000)  | 2.46  |

Aggregated:

| scope            | baseline | turn 1 | final | total | **CoT** | **loop** |
|------------------|----------|--------|-------|-------|---------|----------|
| all 9 cells      |    0.793 |  0.864 | 0.843 | +0.050| **+0.071** | **−0.021** |
| 6 cells, no `edge_existence` | 0.726 | 0.846 | 0.843 | +0.117| **+0.120** | **−0.003** |

**This is the central negative result of the arm.** The entire improvement over the baseline
is the reasoning scaffold. The verify-and-revise loop contributes −0.021 across all nine
cells and −0.003 across the six that exclude `edge_existence`. Excluding `edge_existence`
does not rescue the loop; it moves it from *harmful* to *inert*.

Per cell, the loop is significantly positive **once** (`connected_nodes/friendship`, +0.032)
and significantly negative twice, both on `edge_existence`. The remaining six are null.

**On `edge_existence` the loop is destructive.** It costs −0.118 on adjacency, turning a
0.938 baseline into 0.803, while the scaffold there is roughly neutral (−0.017, ns). This is
the predicted behaviour for a task with nothing to debate: the Critic's only available move
is to talk a correct Proposer out of a correct answer.

The honest one-line statement of this arm is therefore **"asking the model to reason helps;
having a Critic argue about the reasoning does not"** — not "debate helps".

## 5. The Critic carries signal, but its revisions cost accuracy

Every verdict cross-tabbed against whether the Proposer answer *it was judging* was correct
(pooled, 5,825 verdicts):

|                      | AGREE | REVISE |                             |
|----------------------|-------|--------|-----------------------------|
| Proposer **correct** |  4141 |   627  | false-alarm rate **0.132**  |
| Proposer **wrong**   |   526 |   531  | detection rate **0.502**    |

chi2 = 747.1 (1 df, p ≈ 0), **phi = +0.358**, odds ratio 6.67. A REVISE moves P(the answer
is wrong) from the base rate 0.181 to **0.459**. The verdict is a real signal — it is not
noise, and it is not rubber-stamping.

Per cell:

| task/encoding              | verdicts | FA\|ok | det\|bad | phi    | unparsed |
|----------------------------|----------|--------|----------|--------|----------|
| edge_existence/friendship  |      647 |  0.161 |    0.926 | +0.694 |        0 |
| edge_existence/adjacency   |      701 |  0.290 |    0.957 | +0.537 |        0 |
| edge_existence/incident    |      611 |  0.105 |    0.857 | +0.406 |        0 |
| connected_nodes/friendship |      662 |  0.085 |    0.308 | +0.288 |        0 |
| node_degree/adjacency      |      643 |  0.102 |    0.298 | +0.218 |        0 |
| connected_nodes/adjacency  |      685 |  0.201 |    0.423 | +0.217 |        0 |
| node_degree/friendship     |      629 |  0.059 |    0.208 | +0.208 |        0 |
| connected_nodes/incident   |      627 |  0.102 |    0.327 | +0.186 |        0 |
| node_degree/incident       |      620 |  0.070 |    0.260 | +0.185 |        0 |

Discrimination is by far the strongest on `edge_existence` (phi +0.406 to +0.694), where the
check is atomic — which makes the loop's failure there more striking, not less: **the Critic
knows when the answer is wrong on that task and still makes it worse.**

**The resolution is what a REVISE *does*.** Net effect of a revision (bad→ok minus ok→bad):

| task/encoding              | revisions | ok→bad | bad→ok | **net** |
|----------------------------|-----------|--------|--------|---------|
| connected_nodes/friendship |       110 |     11 |     30 | **+19** |
| node_degree/friendship     |        55 |     12 |     12 |     ±0  |
| edge_existence/incident    |        80 |      8 |      6 |    −2   |
| connected_nodes/incident   |        75 |     16 |     12 |    −4   |
| node_degree/incident       |        53 |     10 |      6 |    −4   |
| connected_nodes/adjacency  |       173 |     37 |     26 |   −11   |
| node_degree/adjacency      |        88 |     22 |     11 |   −11   |
| edge_existence/friendship  |       228 |     38 |      9 |   −29   |
| edge_existence/adjacency   |       295 |     89 |     18 | **−71** |

**Net −113 revisions across the arm, positive in exactly one cell.** The Critic detects
errors at 0.502 but revises correct answers at 0.132, and because correct answers vastly
outnumber wrong ones, that false-alarm rate converts into more damage than the detections
repair. This is the mechanism behind section 4's loop deltas, and it is a **precision**
problem, not a detection problem.

### 5a. The grounding audit does not measure what its column names claim

The Critic is instructed to quote an edge from the graph. `diagnostics._classify_evidence`
resolves every cited label pair back to node ids and bins it as *real* (the pair is an edge),
*hallucinated* (it is not), or *no pair*:

| task/encoding              | REVISEs | real edge  | hallucinated | no pair |
|----------------------------|---------|------------|--------------|---------|
| node_degree/adjacency      |      88 | 140 (0.68) |    47 (0.23) |      18 |
| connected_nodes/adjacency  |     174 | 231 (0.64) |    96 (0.27) |      32 |
| connected_nodes/incident   |      75 |  59 (0.62) |    25 (0.26) |      11 |
| node_degree/incident       |      53 |  53 (0.61) |    30 (0.34) |       4 |
| node_degree/friendship     |      55 |  44 (0.49) |    34 (0.38) |      12 |
| connected_nodes/friendship |     110 | 108 (0.44) |   107 (0.44) |      28 |
| edge_existence/incident    |      80 |  11 (0.14) |    64 (0.80) |       5 |
| edge_existence/friendship  |     228 |  24 (0.11) |   202 (0.89) |       2 |
| edge_existence/adjacency   |     295 |  28 (0.09) |   267 (0.91) |       0 |

**The `hallucinated` column does not mean fabrication, and must not be read as it.** The
classifier asks only whether a cited pair is an edge; it never asks whether the Critic
claimed it *was* one. A critique asserting a **non**-relationship — "no such edge as (5,13)
appears in the list" — cites a pair that is correctly not in the edge set, and is binned as
hallucinated. That is valid negative evidence being scored as invention, and it is why
`edge_existence` tops the column: a correct "this edge is absent" critique can cite nothing
else.

**This audit is a null, not a finding.** It cannot separate invention from correct negative
evidence, so it says nothing either way about whether the Critic fabricates. Measuring
fabrication properly requires parsing the polarity of each cited claim (asserted present vs
asserted absent), which the current classifier does not do.

## 6. Format compliance is high and no consensus is manufactured

| metric | v3 |
|---|---|
| turn-1 Proposer output parses | 0.995 (5,375 of 5,400) |
| final answer parses | 0.997 |
| Critic verdicts emitted | 5,825 |
| **unparseable verdicts** | **0 (0.000)** |
| Critic turns hitting the 512 cap | 161 (0.028) |
| Proposer turns hitting the cap | 54 (0.009) |

**No verdict in the entire arm failed to parse.** This matters because the loop defaults an
unparseable verdict to AGREE and terminates, so any such verdict is a consensus manufactured
by the token cap rather than reached by the Critic — it would inflate the Critic's apparent
precision and cut the loop short mechanically.

Truncation still occurs on 161 Critic turns, but it no longer destroys the verdict: in every
one of those turns the `VERDICT:` line was emitted before the cap was reached. The cap's
rationale and the open questions around it are recorded in
[llama70b-cap-decision.md](llama70b-cap-decision.md); the choice of 512 is not principled and
should not be defended as such.

The remaining parse failures are concentrated in `edge_existence/friendship` (turn-1 0.977)
and `edge_existence/adjacency` (0.990); every other cell is at or above 0.990.

## 7. The compute control: matched-compute majority vote beats debate

A reasoned majority-vote arm (`condition = "majority_vote_cot"`) samples the **same v3
Proposer prompt** N=3 times independently at the model's shipped decoding (temperature 0.6,
top_p 0.9, no top_k) and takes the mode of the parsed answers. It differs from debate in
exactly one thing: whether the attempts see each other. Configs
`configs/llama70b-mvcot-{main,seed11,seed13}.toml`, 16,200 draws, same instances.

**Compute is matched, not approximated.** N=3 was derived from debate's own token spend:
1,395–1,659 tokens per instance against a turn-1 cost of 543–567, i.e. 2.57–2.93
turn-1-equivalents. Realized cost is **1,664 tokens per instance for the vote against 1,554
for debate — a ratio of 1.07**, with the vote spending 3 responses to debate's 2.29.

| task/encoding              | baseline | turn 1 | **MV(3)** | debate | MV − debate | b/c   | p        |
|----------------------------|----------|--------|-----------|--------|-------------|-------|----------|
| edge_existence/adjacency   |    0.938 |  0.922 | **0.950** |  0.803 | **+0.147**  | 5/93  | 4e-20 ***|
| edge_existence/friendship  |    0.867 |  0.800 | **0.822** |  0.752 | **+0.070**  | 9/51  | 8e-08 ***|
| node_degree/adjacency      |    0.655 |  0.868 | **0.875** |  0.850 | **+0.025**  | 14/29 | 0.033 *  |
| connected_nodes/incident   |    0.958 |  0.947 | **0.952** |  0.940 | +0.012      | 10/17 | 0.248 ns |
| edge_existence/incident    |    0.980 |  0.978 | **0.985** |  0.975 | +0.010      | 2/8   | 0.109 ns |
| node_degree/incident       |    0.873 |  0.942 | **0.943** |  0.935 | +0.008      | 5/10  | 0.302 ns |
| node_degree/friendship     |    0.637 |  0.828 | **0.837** |  0.828 | +0.008      | 20/25 | 0.551 ns |
| connected_nodes/adjacency  |    0.743 |  0.835 | **0.822** |  0.817 | +0.005      | 31/34 | 0.804 ns |
| connected_nodes/friendship |    0.488 |  0.655 | **0.687** |  0.687 | +0.000      | 33/33 | 0.902 ns |

| scope            | baseline | turn 1 | **MV(3)** | debate | MV − debate | p       |
|------------------|----------|--------|-----------|--------|-------------|---------|
| all 9 cells      |    0.793 |  0.864 | **0.875** |  0.843 | **+0.032**  | 2e-16 ***|
| 6 cells, no `edge_existence` | 0.726 | 0.846 | **0.853** | 0.843 | **+0.010** | 0.035 * |

**The compute control beats the treatment at matched cost.** MV is at least as accurate as
debate in **all nine cells** — it never loses one — and significantly better in three. On
the full matrix it is +0.032 (p=2e-16); on the six cells excluding `edge_existence`, where
section 4 showed the loop to be merely inert rather than harmful, it is still +0.010
(p=0.035). Debate's 2.29 interacting responses are worth less than 3 independent ones.

Per seed, independently:

| seed | baseline | debate | MV(3) | MV − debate | 6-cell MV − debate |
|------|----------|--------|-------|-------------|--------------------|
| 7    |    0.797 |  0.854 | 0.882 | **+0.028**  | +0.017             |
| 11   |    0.790 |  0.847 | 0.879 | **+0.033**  | +0.005             |
| 13   |    0.793 |  0.828 | 0.863 | **+0.034**  | +0.008             |

The vote beats debate in all three seeds on both scopes; the 9-cell margin is stable
(+0.028 to +0.034) while the 6-cell margin is smaller and varies more (+0.005 to +0.017).

**Voting also beats a single reasoned draw**, 0.875 against turn 1's 0.864 (9 cells) and
0.853 against 0.846 (6 cells). So the ordering across the arm is

> baseline **0.793** < debate **0.843** < single reasoned answer **0.864** < voted reasoned
> answers **0.875**

and debate is the *worst* of the three reasoning conditions despite costing more than a
single draw.

**Where the vote's advantage comes from.** Almost entirely `edge_existence`, where debate is
actively destructive: +0.147 and +0.070 on adjacency and friendship. That is the expected
shape — an independent redraw cannot be argued out of a correct answer, whereas a Critic
can. On the six other cells the vote's edge is small but consistently signed (five of six
positive, none negative).

**Diversity, reported rather than tuned.** The decoding is the model's shipped default,
chosen before seeing any result; realized diversity is a diagnostic, not a knob.

| cell                       | unanimous | distinct answers / 3 draws |
|----------------------------|-----------|----------------------------|
| node_degree/incident       |     0.975 | 1.03 |
| edge_existence/incident    |     0.968 | 1.02 |
| connected_nodes/incident   |     0.947 | 1.06 |
| edge_existence/adjacency   |     0.910 | 1.06 |
| node_degree/adjacency      |     0.908 | 1.10 |
| connected_nodes/adjacency  |     0.867 | 1.15 |
| edge_existence/friendship  |     0.858 | 1.11 |
| node_degree/friendship     |     0.853 | 1.15 |
| connected_nodes/friendship |     0.740 | 1.30 |

The draws agree unanimously 74–98% of the time, so the vote rarely has much to arbitrate —
which makes it notable that it still beats debate. The gain is not the vote resolving
disagreement; it is that three independent attempts cannot degrade each other.

**Caveat on reproducibility.** Together documents no determinism guarantee, so the recorded
per-draw seeds identify the request rather than reproduce it. This arm's exact draws will
not replay; its statistics will.

## 8. Additional analysis: an earlier Proposer prompt (v2)

A second prompt version, **v2**, was run over the identical instances and is retained as an
ablation. Its rows are in `results/llama70b-{main,seed11,seed13}`, tagged
`prompt_version: "v2"`; configs `configs/llama70b-debate-{main,seed11,seed13}.toml`.

**What differs.** v3 (a) consolidates v2's two per-task claim-kind wordings into one generic
statement, (b) strips from the Critic's framing four instructions that both task cues already
restated, and (c) removes from the incident cue the instruction to *derive the answer
independently before checking*, so the Critic reviews the Proposer's claims rather than
working the problem out first. The revision role is byte-identical in both versions.

**v2 scores higher.** Paired over all 5,400 instances:

| task/encoding              |    v2 |    v3 | delta   | b/c   | p        |
|----------------------------|-------|-------|---------|-------|----------|
| connected_nodes/incident   | 0.937 | 0.940 | +0.003  | 19/21 | 0.874 ns |
| node_degree/incident       | 0.925 | 0.935 | +0.010  | 6/12  | 0.238 ns |
| node_degree/adjacency      | 0.870 | 0.850 | −0.020  | 40/28 | 0.182 ns |
| edge_existence/incident    | 1.000 | 0.975 | −0.025  | 15/0  | 1e-04 ***|
| connected_nodes/adjacency  | 0.857 | 0.817 | −0.040  | 61/37 | 0.020 *  |
| edge_existence/friendship  | 0.808 | 0.752 | −0.057  | 52/18 | 1e-04 ***|
| edge_existence/adjacency   | 0.872 | 0.803 | −0.068  | 71/30 | 1e-04 ***|
| node_degree/friendship     | 0.898 | 0.828 | −0.070  | 56/14 | 4e-06 ***|
| connected_nodes/friendship | 0.787 | 0.687 | −0.100  | 82/22 | 2e-09 ***|
| **MEAN**                   | **0.884** | **0.843** | **−0.041** | | |

Per seed, v2 is 0.883/0.884/0.884 against v3's 0.854/0.847/0.828 — stable in both arms, so
this is not a seed artefact. On the 6-cell scope v2 is 0.879 against v3's 0.843.

**Why v3 is worse, mechanically.** v2's Critic was markedly more conservative: false-alarm
rate **0.050** against v3's 0.132, detection 0.435 against 0.502, phi **+0.440** against
+0.358. Removing the "derive the answer yourself first" instruction made the Critic flag
more answers in both directions, and since correct answers outnumber wrong ones, the extra
false alarms cost more than the extra detections gain. v2's revisions are net **positive**
on the four fragile cells (+42, +30, +15, +14) where v3's are net −11 to +19.

**What v3 fixed.** v2 produced **319 unparseable verdicts of 5,677 (0.056)**, concentrated
in `node_degree/incident` (169 of 612, 0.257) — each one a cap-manufactured AGREE that ended
the loop early and contaminated that cell's Critic statistics. v3 produces **zero**, because
its cue no longer asks the Critic to enumerate the full edge list before deciding, so it
never runs out of tokens before writing `VERDICT:`.

**The trade is therefore real in both directions**: v2 is more accurate, v3 is
better-behaved. A version keeping v2's derivation instruction while avoiding its truncation
— a larger cap, or requiring `VERDICT:` before the supporting detail — is untested and would
plausibly beat both. That is a one-cell experiment, not a full matrix.

Note that v2's loop is *also* not a clear win over its own turn 1; the difference is one of
degree. The section 4 conclusion is not an artefact of v3.

## 9. Provenance: how the two prompt versions are kept apart

Both versions are defined in `src/gedebate/prompts/debate.py` as fully separate constant
sets — no piece is shared between them even where the text is identical — so an edit to one
cannot alter the other. Every debate row carries its `prompt_version`, and
`scripts/show_results.py` and `scripts/debate_diagnostics.py` **refuse to run** on a mix of
versions unless one is named explicitly.

**A superseded run exists and must not be used.** `results/llama70b-v3prop-v2crit-*` holds
5,400 rows tagged `prompt_version: "v3"` that are in fact a hybrid: v3 Proposer and revision
prompts with a **v2 Critic** prompt. The cause was `conditions/debate.py` calling
`critic_prompt()` without threading `prompt_version`, so it fell back to the module default
(`"v2"`); fixed in commit `e9746444`, which also made the argument mandatory on all three
prompt builders. The diagnosis was confirmed three ways: the code at the run's recorded
commit, re-executing that code against a recording stub, and re-tokenizing stored prompts
against the served model (proposer 20/20 match v3, critic 20/20 match v2).

Those rows remain valid as a clean isolation of the **Proposer preamble change alone**
(v3 preamble + v2 Critic scores 0.879 against v2-throughout's 0.884, n=5,400 — i.e. the
preamble change is a wash, and section 8's −0.041 is attributable to the Critic changes).
Their `prompt_version` field cannot be trusted; the directory name and the `note` field in
their manifest record what they actually are.

## 10. Summary

1. Encoding fragility is large and significant on all three tasks; `incident` best and
   `friendship` worst, unanimously (section 1).
2. Debate improves mean accuracy by **+0.050** over nine cells and **+0.117** over the six
   excluding `edge_existence`, replicating in sign across three independent seeds
   (section 2).
3. Debate **reduces** encoding fragility on both genuinely fragile tasks — 0.470 → 0.253 and
   0.237 → 0.107 — by lifting the worst encoding rather than the best, and **increases** it
   on `edge_existence` (section 3).
4. **The gain is the reasoning scaffold, not the debate.** Turn 1 alone accounts for +0.071
   of the +0.050 (9 cells) and +0.120 of the +0.117 (6 cells); the verify-and-revise loop
   contributes −0.021 and −0.003 respectively. Excluding `edge_existence` moves the loop
   from harmful to inert, not to useful (section 4).
5. The Critic carries real signal (phi = +0.358, a REVISE moves P(wrong) from 0.181 to
   0.459), but its revisions are net **−113** across the arm and positive in one cell of
   nine. The failure is precision, not detection (section 5). The grounding audit's
   `hallucinated` column does **not** show fabrication and is a null (section 5a).
6. Format compliance is high and **no verdict failed to parse**, so no consensus in this arm
   was manufactured by the token cap (section 6).
7. **A matched-compute majority vote beats debate.** Three independent draws of the same
   Proposer prompt, at 1.07x debate's tokens, score **0.875 against debate's 0.843** over
   nine cells (p=2e-16) and 0.853 against 0.843 over six (p=0.035), losing no cell. The
   ordering is baseline 0.793 < debate 0.843 < one reasoned draw 0.864 < voted draws 0.875
   (section 7).
8. An earlier prompt, v2, scores **+0.041 higher** but produced 319 unparseable verdicts
   against v3's zero — more accurate, worse behaved (section 8).
