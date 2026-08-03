# Findings

Empirical results, in the order the project produced them. This is the **results** log:
what the runs actually showed, with the number and the test that backs it. It is kept
apart from [notes.md](notes.md), which holds reference facts and design *decisions*, and
from [plan/](plan/), which holds the steps. This file feeds the results section of the
writeup; notes.md feeds the methodology section.

Every number here regenerates from committed code over committed run outputs:

```bash
python scripts/show_results.py results/main results/seed11 results/seed13 \
    --fragility --compare --save analysis/pooled
python scripts/debate_diagnostics.py results/main results/seed11 results/seed13 \
    --save analysis/pooled
# the v2 arm (rows carry prompt_version; the analysis refuses to pool the two)
python scripts/debate_diagnostics.py results/main results/seed11 results/seed13 \
    results/v2-main results/v2-seed11 results/v2-seed13 \
    --prompt-version v2 --save analysis/pooled-v2
```

Unless a line says otherwise, numbers are **pooled over seeds 7/11/13, n=600 per cell**,
from `analysis/pooled/`. Seed 7 alone is in `analysis/main/`. All debate numbers are
scored under the current last-line answer parser (3g); the v1 runs were re-scored from
their persisted raw output with `scripts/rescore.py`, which moved 52 of 5,400 rows. Encodings are applied to the
same graphs, so the tests are paired (McNemar, Cochran's Q); see notes.md.

---

## 1. Encoding fragility reproduces (P3)

The premise the project rests on holds. Baseline accuracy swings sharply with the
encoding on two of three tasks:

| task            | best      | worst     | gap    | Cochran Q (df=2) | best-vs-worst McNemar |
|-----------------|-----------|-----------|--------|------------------|-----------------------|
| node_degree     | incident  | adjacency | 0.3617 | 200.2, p=3e-44   | 258/41, p=8e-36       |
| connected_nodes | incident  | friendship| 0.1100 | 49.6, p=2e-11    | 90/24, p=1e-09        |
| edge_existence  | adjacency | incident  | 0.0133 | 0.52, p=0.77     | 64/56, p=0.52         |

Per-cell accuracy: `node_degree` runs 0.388 (adjacency) to 0.750 (incident);
`connected_nodes` 0.263 (friendship) to 0.373 (incident); `edge_existence` is flat at
0.690 to 0.703.

**`edge_existence` is not encoding-fragile.** Its spread is 0.013 and its omnibus test is
nowhere near significant. Treating it as a third fragile task in the writeup would
misstate the result: it is a *control* that happens to sit in the matrix, and any
intervention has almost no fragility to remove there.

## 2. Majority vote buys nothing (P4)

At 10 samples and 10x the tokens, self-consistency is statistically indistinguishable
from a single greedy answer in all 9 cells.

*Seed 7 only, n=200*: `majority_vote/` was only ever run on seed 7. Every delta is within
±0.01 and every McNemar p is 0.69 or above, on 1 to 9 discordant pairs out of 200.

The mechanism is visible in the numbers: `per_sample_accuracy` is within a point of
`voted_accuracy` in every cell (e.g. `node_degree/incident` 0.746 vs 0.755). The 10 draws
at temperature 0.7 are nearly identical, so there is no disagreement for a vote to
resolve. This is the control the debate condition needs: extra compute alone does not move
this model on these tasks, so any debate gain would be the procedure rather than the
sampling.

> **Caveat on `analysis/pooled/mv_vs_baseline.csv`: do not use it.** MV exists only for
> seed 7, but `report.compare_baseline_vote` summarizes each condition over all of its own
> rows, so in the pooled run the delta column compares 200 MV instances against 600
> baseline instances while the McNemar column is correctly paired over just the 200 shared
> ones. The two columns describe different populations. The valid comparison is
> `analysis/main/mv_vs_baseline.csv`. Either run MV on seeds 11/13, or restrict
> `compare_baseline_vote` to instances present in both conditions.

## 3. Debate buys nothing either, and the diagnosis says why (P5)

Debate is statistically indistinguishable from baseline in 8 of 9 cells. The one exception
is `node_degree/adjacency`, +0.055 (p=0.028), and the turn split below shows most of that
is the chain-of-thought prompt rather than the debate.

**Debate makes encoding fragility worse** on two of three tasks, which is the opposite of
the hypothesis:

| task            | baseline std | debate std | baseline max-min | debate max-min |
|-----------------|--------------|------------|------------------|----------------|
| connected_nodes |       0.0484 | **0.0739** |           0.1100 |     **0.1733** |
| edge_existence  |       0.0055 | **0.0160** |           0.0133 |     **0.0383** |
| node_degree     |       0.1566 |     0.1376 |           0.3617 |         0.2983 |

Full diagnosis in [plan/p5-followup-diagnosis.md](plan/p5-followup-diagnosis.md). The
headline numbers:

### 3a. The Critic's verdict is worth almost nothing

Over 6,553 verdicts, cross-tabbed against whether the Proposer answer *that verdict was
judging* was correct: false-alarm rate 0.555, detection rate 0.584. A REVISE moves P(the
answer is wrong) from the base rate 0.520 to 0.533 (chi2 = 5.72, p = 0.017, phi = +0.030,
odds ratio 1.13). Significant at N in the thousands, and worth about one accuracy point.

It is not uniformly at chance. `edge_existence/incident` detects a wrong answer 97 percent
of the time but also cries REVISE on 82 percent of correct ones (phi = +0.21,
p = 1e-09). The discrimination is real and drowned by the bias, which is a calibration
failure rather than an ignorance one.

**Its evidence is often fabricated.** The Critic is told to quote an edge from the graph.
Resolving the labels in every cited problem back to node ids: 14 to 32 percent of cited
pairs are edges that are not in the graph (worst: `node_degree/incident` at 0.32). On
`edge_existence` it mostly cites no pair at all (693 of 725 problems on incident).

**The Proposer mostly ignores it**: after a REVISE the answer changes 1,180 of 3,734 times
(32 percent), and the direction is near a wash (ok→bad 397, bad→ok 356).

### 3b. The loop is net-harmful and hides a chain-of-thought effect

The baseline is answer-only (2 to 10 generated tokens per instance) while the debate
Proposer writes a claim trace, so `debate vs baseline` conflates CoT with debate. Debate
turn 1 *is* a single-turn CoT answer at the same decoding settings, which separates them
with no new runs:

| task/encoding             | baseline | turn-1 | final | CoT delta (p)       | loop delta (p)      |
|---------------------------|----------|--------|-------|---------------------|---------------------|
| connected_nodes/friendship|    0.263 |  0.188 | 0.207 | **-0.075 (0.0005)** | +0.018 (0.091)      |
| connected_nodes/incident  |    0.373 |  0.462 | 0.415 | **+0.088 (0.0015)** | **-0.047 (0.0001)** |
| edge_existence/adjacency  |    0.703 |  0.663 | 0.723 | -0.040 (0.087)      | **+0.060 (0.0025)** |
| edge_existence/incident   |    0.690 |  0.740 | 0.685 | +0.050 (0.068)      | **-0.055 (0.0219)** |
| node_degree/adjacency     |    0.388 |  0.440 | 0.443 | **+0.052 (0.0426)** | +0.003 (0.905)      |

(Four cells with no significant movement in either step are omitted; full table in the
plan doc and `analysis/pooled/debate_turn_split.csv`.)

The claim-trace prompt gains where the encoding is already tractable and **loses 0.075 on
`connected_nodes/friendship`**, then the loop works against it in the two cells where it
gained most. The two effects partly cancel, which is why the headline delta looked like
nothing.

### 3c. Root cause of the friendship collapse: the prompt asks for the wrong label space

The debate Proposer's `ANSWER:` hint is keyed on the **task** only:

```python
"connected_nodes": "a comma-separated list of node ids, or none"
```

but `connected_nodes`'s answer lives in the **encoding's** label space: integers under
adjacency and incident, *names* under friendship. So on a friendship graph the prompt
tells the model to answer with node ids while the graph is labelled James, Robert, John.
The model complies, emits integers, and the parser cannot map them back to nodes.

The traces confirm it. Of 600 `connected_nodes/friendship` turn-1 answers, 64 contain
integers and **no name at all**, and **all 64 are unparsed** (of 78 unparsed total). The
degenerate outputs are literally `1. 1`, `2. 3`, `3. 4` counting upward to 40 on a
five-node named graph.

This explains the single largest anomaly in the results. The zero-shot **baseline** prompt
does not have the bug (`"Answer with the connected nodes as a comma-separated list"`, no
"node ids"), which is exactly why the claim-trace prompt *loses* 0.075 on
`connected_nodes/friendship` while gaining 0.078 on `connected_nodes/incident`: the
regression is the debate prompt introducing a label-space mismatch that the baseline never
had. It is a prompt bug, not a reasoning failure, and not evidence about friendship.

`node_degree` ("a single integer, the degree") and `edge_existence` ("Yes or No") are
label-space free and unaffected.

### 3d. A format degeneration costs 13 percent of one cell

The Proposer copies the format template literally (`17. <one atomic claim>`) until it hits
the token cap. On `connected_nodes/friendship` that is 86 of 600 turn-1 answers truncated
and 78 unparsed, i.e. 13 percent scored wrong for a non-reasoning reason.
`edge_existence/adjacency` and `/incident` lose 12 percent the same way. `node_degree`
escapes only because the answer parser takes the last integer in the text, which happens
to recover the degree.

### 3e. No stopping rule rescues the loop (Tier 1a, replayed offline)

Every candidate rule stops the loop earlier than it really did, so truncating the trace
and reading the answer standing at that point is an exact replay. Result: **evidence
gating does not work.** The largest gain from vetoing REVISEs with fabricated citations is
+0.013 (ns), and it significantly *hurts* `edge_existence/friendship` (-0.010, p=0.031).
Capping revisions at one changes nothing. The loop's damage is not concentrated in
critiques with fabricated evidence.

The strict "must cite a real edge" gate's one substantial win (+0.052 on
`edge_existence/incident`, p=0.029) is abstention wearing a different hat: 96 percent of
REVISEs there cite no pair, so the gate vetoes nearly all of them and collapses onto not
debating at all (+0.055).

The only rule with real effect is **not running the loop**, and even that is
cell-dependent: +0.047 and +0.055 on the two `incident` cells (p=0.0001 and p=0.022), but
-0.060 on `edge_existence/adjacency` (p=0.002). There is no fixed stopping rule that wins
everywhere.

### 3f. The numbered-list template was load-bearing (prompt v2 pilot)

A pilot of Proposer prompt v2 on the two worst-parsing cells, 200 seed-7 instances each,
paired against the v1 rows in `results/main`. The first v2 draft made two edits at once:
it fixed the label-space bug in 3c, and it replaced v1's fill-in template
(`1. <one atomic claim>`) with a prose description of the same format. The two edits had
**opposite** effects, which is what the pilot was for.

| | `connected_nodes/friendship` | `edge_existence/incident` |
|---|---|---|
| turn-1 unparsed | 25 → **1** | 27 → **0** |
| turn-1 no `ANSWER:` line | 53 → 13 | 40 → **0** |
| turn-1 truncated | 23 → 12 | 1 → 1 |
| turn-1 accuracy | 0.210 → **0.125** (-0.085) | 0.745 → **0.820** (+0.075) |
| final accuracy | 0.255 → **0.160** (-0.095) | 0.655 → 0.615 (-0.040) |
| final McNemar | b=35, c=16, **p=0.0117** | b=33, c=25, p=0.358 |

**The parse fixes worked.** Unparsed turn-1 answers went to ~0 in both cells, and the
explicit "final line beginning with ANSWER:" instruction eliminated the missing-answer
failure outright on `edge_existence/incident` (40 → 0), which alone lifted that cell's
turn-1 accuracy by 0.075.

**Removing the template backfired on the enumeration task.** Without a demonstrated list,
the model stopped numbering: 2 of 200 turn-1 answers used numbered lines, and 174 of 200
switched to a `CLAIM:` prefix the prompt never asked for. On `connected_nodes` the list
was doing real reasoning work, and its loss cost 0.085 turn-1 accuracy (p=0.0117 on the
final answer). The error shape shows how: Jaccard 0.578 → 0.526, has-extra 0.566 → 0.633,
has-missing 0.417 → 0.523. Instead of enumerating the *queried* node's edges the Proposer
enumerated the whole graph's, e.g. on `7/13` (gold: Thomas, Christopher) it emitted
`CLAIM: The edge "James and Mary" exists. CLAIM: The edge "James and Linda" exists. ...`
and answered with eight nodes. `edge_existence` was immune because a Yes/No answer
requires no enumeration.

So the fix is not "prose beats templates" but **state the numbering explicitly**. v2 was
amended to say `Number your claims 1., 2., 3., and so on` (the scaffold, without a
fill-in block to echo) and the pilot rerun confirms it:

| | v1 | v2-draft | **v2 adopted** |
|---|---|---|---|
| `connected_nodes/friendship` numbered lines | 200 | 2 | **199** |
| turn-1 unparsed | 25 | 1 | **1** |
| turn-1 truncated | 23 | 12 | **1** |
| turn-1 accuracy | 0.210 | 0.125 | 0.205 |
| final vs v1 | — | -0.095, p=0.0117 | -0.025, p=0.532 |
| `edge_existence/incident` turn-1 accuracy | 0.735 | 0.820 | **0.790** |
| final vs v1 | — | -0.040, p=0.358 | **+0.065**, p=0.137 |

Accuracy rows are scored under the adopted last-line parser (3g), which is why they differ
slightly from the figures quoted while the pilots were running; the mechanical counts are
unaffected.

**v2 beats v1 on every mechanical measure**, including truncation (23 → 1), which the
draft only half fixed. On accuracy it is a clear win on `edge_existence/incident`
(+0.065 final) and a **wash** on `connected_nodes/friendship` (-0.005 turn-1, p=1.000;
b=21 vs c=20, i.e. one instance out of 200). v2 is adopted on the strength of the
mechanical fixes and the `edge_existence` gain, not on a friendship accuracy improvement,
which it does not deliver.

**It also invalidates one of v1's findings.** In 3b the loop appeared to *gain* on
`connected_nodes/friendship`. That gain is now +0.018 and not significant (p=0.091) once
v1 is re-scored under the corrected parser (3g), and under v2 it collapses further to
+0.005 (turn-1 0.215 → final 0.220). Under v1 the
loop looked helpful there only because 25 turn-1 answers were unparseable from the
label-space bug and revisions rescued some. With turn-1 parsing correctly there is nothing
left to rescue. The loop was cleaning up our own prompt bug, which strengthens the overall
negative result rather than weakening it.

**What the fix did not buy:** repairing 24 unparsed answers moved turn-1 accuracy by
+0.005, so almost every newly-parsed answer is still wrong. Friendship is genuinely hard
for this model, not a parsing artefact.

One thing the draft did *not* cost, despite appearances: claim extraction (`^\d+\.` in
`parse_proposer`) returned nothing for 99 percent of instances, but nothing in the harness
reads the `claims` field. It is written to the trace sidecar and never consumed. It would
only become load-bearing under per-claim verification.

### 3g. The answer parser no longer reads the reasoning

`parse_proposer` used to fall back to the **whole output** when the Proposer omitted its
`ANSWER:` line. For `connected_nodes` that is not answer extraction at all:
`scoring._parse_node_list` collects every recognised label it sees, so a claim like
`"Robert is not connected to Susan"` put Susan in the answer. Verified directly: that text
parses to `[Michael, Susan]` under whole-text and `[Michael]` under last-line.

The fallback is now the **last non-empty line**. That matches the rule `_parse_int` and
`_parse_bool` already use ("the answer is the last thing stated") and the rule the
label-free baseline relies on, and it costs nothing (seed 7, turn-1):

| | no-format cases | whole-text (old) | **last-line (new)** | strict `ANSWER:` only |
|---|---|---|---|---|
| connected_nodes/friendship, v1 | 53/200 | 0.210 | **0.210** | 0.140 |
| connected_nodes/incident, v1 | 35/200 | 0.455 | **0.460** | 0.420 |
| connected_nodes/friendship, v2 | 39/200 | 0.215 | **0.205** | 0.145 |

**Why not require an explicit `ANSWER:` line?** Because the baseline has no label either.
It is told "answer with a single integer and nothing else", and its answer is read off its
last value. Holding debate to a stricter extraction rule would cost 0.07 to 0.30 on
`node_degree` (worst: `node_degree/incident` 0.728 → 0.428) purely from a formatting
requirement the baseline never faces. That would be an artefact of unequal standards, not
a real effect.

**The parser is deliberately not versioned.** Prompts are versioned because they change
what the model generates; the parser only changes how stored text is scored. Scoring v1
results under the old rule and v2 under the new one would make the conditions
incomparable, so one parser applies uniformly and results are re-scored. This is free:
raw outputs are persisted in the rows and traces, so no rerun is needed.

### 3h. The model never answers "none"

On `connected_nodes/friendship` instances whose true answer is the empty set, the model
scores **0 of 18**, under both v1 and v2. It is not a formatting failure. On `7/80` it
reasons correctly and then contradicts itself:

```
1. Robert is an end node.
2. There is no edge involving Robert.
3. ANSWER: James, John, Michael
```

Same trace-versus-answer disconnect as 3g: the reasoning reaches the right conclusion and
the answer step discards it. This is 9 percent of that cell guaranteed wrong against a
cell accuracy of 0.215, so it is the largest single remaining sink. Whether it is a
wording problem (`or none` is buried mid-sentence in the answer spec) or a model bias
against emitting an empty list is **untested**.

## 4. With a corrected prompt, debate is inert and fragility gets *worse* (Tier 1c)

The full 3x3 matrix re-run under Proposer prompt v2, all three seeds, n=600 per cell
(`results/v2-*`, `analysis/pooled-v2/`). Both arms are scored under the same last-line
parser, and the analysis refuses to pool them: rows carry `prompt_version`.

This is the cleanest statement of the project's result, because v2 removes the parsing
artefacts that muddied v1.

### 4a. The debate loop does essentially nothing

Loop delta (turn-1 to final answer) under v2 is **non-significant in 8 of 9 cells**. The
exception is `edge_existence/adjacency`, +0.055 (p=0.0092), on the task that is not
encoding-fragile to begin with. Under v1 there were three significant loop effects, all of
which are now explained as parsing artefacts or gone.

Given a Proposer whose output parses, the verify-and-revise loop moves nothing.

### 4b. A better prompt does not reduce fragility, it amplifies it

| task | baseline | debate v1 | **debate v2** |
|---|---|---|---|
| connected_nodes max-min | 0.110 | 0.208 | **0.333** |
| connected_nodes std | 0.0484 | 0.0870 | **0.1534** |
| edge_existence max-min | 0.013 | 0.038 | **0.132** |
| node_degree max-min | 0.362 | 0.298 | 0.312 |

`connected_nodes` spread **triples** against the baseline. The mechanism is in the
per-cell numbers: v2 lifts `connected_nodes/incident` from 0.415 to 0.535 while *lowering*
adjacency (0.272 to 0.218) and leaving friendship flat (0.207 to 0.202).

**The gains land in the encoding that was already best.** That is the same pattern the
oracle ceiling showed for a *perfect* Critic. Two unrelated interventions, a better prompt
and flawless verification, each raise mean accuracy and each widen the encoding gap. The
project's hypothesis was that debate would make graph reasoning robust to encoding; on
this task family the opposite holds, and it holds for reasons that have nothing to do with
Critic quality.

### 4c. v2 vs v1, paired per cell

| cell | baseline | v1 | v2 | v2-v1 | McNemar |
|---|---|---|---|---|---|
| connected_nodes/incident | 0.373 | 0.415 | **0.535** | **+0.120** | 47/119, p<1e-4 |
| edge_existence/adjacency | 0.703 | 0.723 | **0.787** | **+0.063** | 52/90, p=0.0019 |
| edge_existence/incident | 0.690 | 0.685 | 0.730 | +0.045 | 75/102, p=0.051 |
| node_degree/incident | 0.750 | 0.728 | 0.723 | -0.005 | 32/29, p=0.80 |
| connected_nodes/friendship | 0.263 | 0.207 | 0.202 | -0.005 | 62/59, p=0.86 |
| node_degree/friendship | 0.458 | 0.430 | 0.412 | -0.018 | 89/78, p=0.44 |
| node_degree/adjacency | 0.388 | 0.443 | 0.417 | -0.027 | 88/72, p=0.24 |
| edge_existence/friendship | 0.695 | 0.693 | 0.655 | -0.038 | 90/67, p=0.079 |
| connected_nodes/adjacency | 0.280 | 0.272 | **0.218** | **-0.053** | 72/40, p=0.0034 |

**v2 is not uniformly better.** It wins significantly on two cells, loses significantly on
one, and is a wash on six. The pilot (3f) tested only `connected_nodes/friendship` and
`edge_existence/incident` and could not have detected the `connected_nodes/adjacency`
regression. A two-cell pilot is not a substitute for the matrix.

### 4d. The CoT scaffold is what amplifies fragility

The turn-1 (chain-of-thought) effect under v2, against the answer-only baseline:

| cell | CoT delta | p |
|---|---|---|
| connected_nodes/incident | **+0.165** | <1e-4 |
| edge_existence/incident | **+0.077** | 0.0043 |
| connected_nodes/friendship | **-0.065** | 0.0031 |
| connected_nodes/adjacency | **-0.053** | 0.0137 |
| node_degree/friendship | **-0.052** | 0.0408 |

The claim-trace prompt is worth +0.165 on the encoding it suits and *costs* 0.05 to 0.07
on three others. Since the loop contributes nothing (4a), this scaffold is the entire
mechanism behind 4b: fragility amplification is a property of the reasoning format, not of
debate.

### 4e. The Critic is marginally better and still useless

Pooled over 6,424 v2 verdicts: false-alarm 0.563, detection 0.607, phi +0.044, odds ratio
1.20 (v1: +0.030 and 1.13). A REVISE now moves P(the answer is wrong) from 0.510 to 0.529.
Better formatting sharpens the Critic slightly and changes nothing that matters.

### 4f. v2 introduced one regression of its own

`edge_existence/friendship` turn-1 truncation went from 8 to **53** of 600, and unparsed
from 17 to **52** (0.09), the worst parse loss in the v2 matrix. That plausibly accounts
for its -0.038. v2 fixed the truncation it was designed to fix and created a smaller one
elsewhere; the cap for that cell is the obvious suspect and is untested.

## 5. Friendship fails differently from the integer encodings

Not just more often, differently in kind (turn-1 answers):

| task/encoding              | signal                                                        |
|----------------------------|---------------------------------------------------------------|
| node_degree/adjacency      | mean signed error **-1.16**, undercounts on 45 percent        |
| node_degree/friendship     | mean signed error **+0.88**, overcounts on 37 percent         |
| connected_nodes/friendship | mean Jaccard **0.551**, **61 percent** contain a non-neighbour|
| connected_nodes/incident   | mean Jaccard 0.787, 23 percent contain a non-neighbour        |
| edge_existence/incident    | answers Yes 0.595 against a gold rate of 0.480 (**+0.115**)   |

Friendship makes the model hallucinate *extra* relations; adjacency makes it miss real
ones. Two candidate explanations, not yet separated: the social framing ("X and Y are
friends") invites transitive closure, or non-integer labels are simply harder to track.
The alphabetic-label encoding proposed in the plan doc's Tier 3 is the experiment that
separates them.

## 6. Power is not the limiting factor

At pooled n=600 the discordant-pair counts per cell run 150 to 250, which resolves an
effect of about 0.05. The observed condition effects are near zero rather than hidden by
noise. More seeds will not turn them significant.
