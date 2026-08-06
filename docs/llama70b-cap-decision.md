# Open decision: the generation cap and what truncation cost us

**Status: undecided.** This is a design memo, not a result — it records what we know about
the `max_new_tokens` cap in the Llama-3.3-70B arm, what it contaminated, and the options,
so the decision can be made later with the evidence in front of us rather than
reconstructed. Results live in [findings-llama70b.md](findings-llama70b.md); this file is
the reasoning behind one number in it.

The arm as it stands ran with **baseline cap 256, debate cap 512**
(`configs/llama70b-{baseline,debate}-*.toml`).

---

## 1. Why there is a cap at all

Three reasons, in descending order of how much they still apply:

1. **Runaway insurance, and on a paid API this is the real one.** Generation stops at EOS
   *or* the cap. Without a cap a degenerate turn runs to the model's context limit —
   131,072 tokens for this model, about $0.14 a turn. This arm ran 11,675 turns; a 1
   percent runaway rate would be ~$16, more than the entire debate arm cost ($9.34). The
   cap is what makes the bill bounded and predictable.
2. **Inherited from the GPU path.** The harness was written for local RTX 2080 nodes, where
   KV cache and wall-clock scale with generation length and a SLURM array has a time limit.
   That constraint does not exist on the API path, but the cap came along with the code.
3. **There is no "unlimited".** HuggingFace `generate()` defaults to 20 new tokens, so some
   value must be chosen deliberately. On the API side `max_tokens` is likewise required.

**Honest note on how 512 was chosen.** It is not principled. The compliance pilot at 256
failed on `connected_nodes/adjacency` (Critic truncation 82/210, 74 unparsed verdicts), so
the cap was doubled and the pilot re-run. 512 is the number that made the pilot pass, not a
number derived from the output-length distribution. It should not be defended as
principled in the writeup.

## 2. What truncation actually costs

Truncation is a **silent failure that converts a reasoning outcome into a formatting
outcome.** A Critic cut off before it writes `VERDICT:` does not return "I could not
finish"; it returns unparseable text. `conditions.debate` then defaults that verdict to
**AGREE and breaks the loop**, so the instance ends in a consensus that was never reached.
The default exists so a malformed verdict cannot hang a run, which is reasonable in
isolation and dangerous when it fires often.

Across the whole debate arm: **286 of 11,675 turns truncated (2.4 percent)** — 22 Proposer,
264 Critic. Concentrated, not spread:

| task/encoding              | critic turns | median gen | p90 | % at cap |
|----------------------------|--------------|------------|-----|----------|
| **node_degree/incident**   |          612 |        332 | 512 | **25.7** |
| connected_nodes/incident   |          612 |        220 | 454 |      7.4 |
| connected_nodes/friendship |          667 |        165 | 281 |      3.3 |
| connected_nodes/adjacency  |          654 |        209 | 367 |      2.6 |
| node_degree/adjacency      |          642 |        170 | 321 |      1.9 |
| edge_existence/friendship  |          624 |         15 |  29 |      0.6 |
| edge_existence/adjacency   |          649 |         15 |  31 |      0.5 |
| node_degree/friendship     |          616 |        136 | 190 |      0.5 |
| edge_existence/incident    |          601 |          7 |  29 |      0.2 |

Note `node_degree/incident`'s p90 is *at* the cap: its true output distribution is
**censored**, so we do not know how long it wanted to be. That matters for choosing a
replacement value — see section 5.

## 3. Two different things get truncated, and the cap cannot tell them apart

**(a) Legitimate long work.** On `node_degree/incident` the Critic is not degenerating. It
systematically walks the incident list for the queried node, then checks each Proposer
claim in turn:

```
To determine the degree of node 13, I will go through the graph's edge list...
- Node 0 is connected to nodes ... 13.
- Node 1 is connected to nodes ... 13.
[...]
12. There is an edge between node 13 and node 11. (Correct)
13. There is an edge between node 13 and node 12. (Correct)     <- cut off here
```

Two costs stack in this cell: `incident` spells out every node's neighbours, so scanning it
is long, and `node_degree`'s Proposer emits one claim per neighbour, so verification is long
too. Truncation is size-dependent (16 / 37 / 27 percent for 5–9 / 10–14 / 15–19 node
graphs). **A higher cap fixes this case.**

**(b) Genuine runaway.** On `edge_existence` — a Yes/No question about *one* pair — the
Proposer transcribes the whole edge list as numbered claims:

```
1. The graph G contains an edge (0, 2) connecting nodes 0 and 2.
...
37. The graph G contains an edge (5, 10).                        <- still going at the cap
```

Here the cap is doing its job. **But even this case is not clean**: in one sampled instance
the Proposer reached `"...does not confirm a direct edge between 5 and 4"` at claim 17 and
was converging on the right answer when it was cut. So a higher cap may rescue some of
these too, rather than merely paying for more padding.

## 4. What this contaminated, and what it did not

For `node_degree/incident` (157/612 Critic turns truncated, 169 unparsed verdicts):

- **Critic phi is contaminated.** `diagnostics._count_verdicts` files a defaulted verdict
  into the AGREE cells, so 169 truncation artefacts are counted as the Critic agreeing.
- **The loop delta is contaminated.** The loop was cut short mechanically, so it measures
  the cap rather than the procedure.
- **Accuracy is NOT contaminated.** The final answer is whichever Proposer answer was
  standing when the loop stopped; an early stop means fewer revisions, not a broken answer.
  `node_degree/incident` = 0.925 is a real number, and section 2 of findings is unaffected.

`connected_nodes/incident` has a milder form (62 unparsed of 612).

### What can be recovered for free

Recomputing that cell over **only the 443 parsed verdicts**, and the loop delta over the
**431 of 600 instances never hit by an unparsed verdict**:

|                          | as reported (unparsed = AGREE) | parsed-only subset |
|--------------------------|-------------------------------|--------------------|
| false alarm              | 0.023                         | 0.031              |
| detection                | 0.064                         | **0.115**          |
| phi                      | +0.068 (p=0.092, ns)          | **+0.106 (p=0.026)** |
| loop delta               | −0.013                        | **+0.030 (p=0.061)** |

Both move the way you would expect once the artefact is removed. **Neither is trustworthy
either.** The excluded turns are not a random sample — they are the long, high-degree,
hard instances — so conditioning on "the Critic finished" selects for easy cases in both
tables. This is *better than what is in the doc*, not clean, and if it is used it must be
labelled a sensitivity check with the selection bias stated.

## 5. Options, with costs

The key economic fact: **a higher cap is nearly free.** Generation stops at EOS, so raising
the cap costs nothing for turns that already finish. Only the 286 currently-truncated turns
generate more. Even giving every one of them another 1,536 tokens is **$0.46** across the
whole arm.

The expense is not the cap, it is that **a rerun is all-or-nothing per cell**:

| option | cost | what it buys | what it leaves |
|---|---|---|---|
| do nothing | $0 | — | one cell's phi + loop delta unusable |
| report parsed-only as a sensitivity check | $0 | usable-with-caveat numbers | selection bias, must be stated |
| rerun `node_degree/incident` at a higher cap | ~$1.25 | fixes the worst cell | per-cell cap heterogeneity to caveat |
| \+ `connected_nodes/incident` | ~$2.40 | fixes both incident cells | mild heterogeneity in the other 7 |
| rerun all 9 cells at a uniform higher cap | ~$10 | removes the cap as a variable entirely | nothing |

**On the heterogeneity objection.** A cap that varies by cell is only a confound where it
*binds*. If no cell truncates, the cap is not a variable at all. At 512 the non-incident
cells run 0.2–3.3 percent, so a targeted rerun leaves small but nonzero residual
heterogeneity — defensible with a methods note, not as clean as a uniform rerun.

**Choosing the new value.** Do not double again by reflex; that is how we got 512.
`node_degree/incident`'s distribution is censored at 512 with p90 already at the ceiling, so
the true tail is unknown. Either (a) run a cheap one-cell probe at 2048 and read the actual
distribution before committing, or (b) go straight to 2048+ on the grounds that overshoot is
nearly free and undershoot costs another rerun.

## 6. What is undecided

- [ ] Whether to rerun at all, or ship with the caveat and the parsed-only sensitivity
      check.
- [ ] If rerunning: targeted (2 cells) or uniform (9 cells).
- [ ] The new cap value, and whether to probe first.
- [ ] Whether the Critic should get its **own** cap. The Proposer truncates at 22/5,998
      (0.4 percent) while the Critic runs 264/5,677 (4.7 percent) — the problem is almost
      entirely Critic-side. `run_debate` currently passes one `max_new_tokens` to both.
      A separate `critic_max_new_tokens` would be a small change to `RunConfig` and
      `run_debate` and would target the fix precisely. Not needed if the cap simply goes
      high enough for both.

## 7. Priority note

This is **not** the largest gap in the arm. That is the absent compute control
(findings-llama70b.md section 7), which bounds every headline claim rather than two
secondary numbers in one cell of nine. If budget is limited, majority vote comes first.

The counter-argument, worth weighing when the writeup deadline is known: reruns are
all-or-nothing and the endpoint can drift (Together serves this model from an FP8 endpoint
they can update under the same name), so postponing a rerun indefinitely risks having to
redo the baseline alongside it as a drift check.
