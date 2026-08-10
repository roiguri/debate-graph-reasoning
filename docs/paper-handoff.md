# Paper handoff — state of play

Written 2026-08-10 to start the paper draft in a fresh session. Everything here is
verified against the run outputs; the full results log is
[findings-llama70b.md](findings-llama70b.md).

---

## 1. The question

From [project_proposal.md](project_proposal.md), *"Does Debate Improve Encoding-Fragile LLM
Graph Reasoning?"*:

> whether a multi-agent Proposer–Critic debate can improve accuracy across encodings at
> matched compute, and whether any improvement comes from the framework of debate, rather
> than just from combining multiple model runs.

The proposal's novelty claim (§2.3) is **not** "debate on graphs" but a controlled test of
*verification vs. aggregation*. Its hook (§1.2): debate assumes a **verification asymmetry**
— checking a claim is cheaper than producing one. Prior null results (Huang 2024, Choi 2025)
come from domains where that assumption cannot be tested, because intermediate steps are not
objectively checkable. Graphs remove the obstacle: an atomic edge claim either appears in the
serialized graph or it does not. §3.3 pre-registers that **a null result is itself a notable
finding**, and names accuracy primary, cross-encoding spread secondary.

## 2. Setup

`meta-llama/Llama-3.3-70B-Instruct-Turbo` (FP8) via Together.ai. Three tasks
(`edge_existence`, `node_degree`, `connected_nodes`) × three encodings (`adjacency`,
`incident`, `friendship`) × three seeds (7/11/13) × 200 graphs = **600 instances per cell,
5,400 per condition**. Encodings are applied to the same graphs, so all tests are paired.

| condition | what it is | decoding | cost/instance |
|---|---|---|---|
| baseline | one zero-shot answer-only response | greedy, cap 256 | 1 resp / 364 tok |
| debate | Proposer–Critic loop, prompt **v3** | greedy, cap 512 | 2.29 resp / 1,554 tok |
| majority_vote_cot | N=3 independent draws of the **same** Proposer prompt, voted | T=0.6, top_p=0.9, cap 512 | 3 resp / 1,664 tok |

N=3 was **token-matched** to debate before the run (debate buys 2.57–2.93 turn-1-equivalents
per task). Realized ratio: **1.07×**. Decoding is Llama-3.3-70B's shipped
`generation_config`, chosen before seeing any result — realized diversity is reported as a
diagnostic, never tuned.

## 3. Main results

**Accuracy.** `edge_existence` is a single-pair lookup where a claim-by-claim critique has
nothing to work on; it behaves as a control. Both scopes are always reported.

| scope | baseline | turn 1 (CoT) | debate | **MV(3)** |
|---|---|---|---|---|
| all 9 cells | 0.793 | 0.864 | 0.843 | **0.875** |
| 6 cells (no `edge_existence`) | 0.726 | 0.846 | 0.843 | **0.853** |

1. **A matched-compute vote beats debate.** +0.032 over nine cells (p=2e-16) and +0.010 over
   six (p=0.035), at 1.07× the tokens, **losing no cell**. Replicates 3/3 (+0.028/+0.033/+0.034).
2. **Debate's gain over the baseline is entirely the reasoning prompt.** Decomposing
   baseline → turn 1 → final: CoT **+0.071**, loop **−0.021** (nine cells); CoT **+0.120**,
   loop **−0.003** (six cells). Excluding `edge_existence` moves the loop from harmful to
   *inert*, not to useful.
3. **The verification asymmetry holds — and debate still loses.** The Critic genuinely
   discriminates: φ=**+0.358** pooled, detection 0.502, rising to φ=+0.694 where the check is
   a single edge lookup. The failure is **precision**: correct answers outnumber wrong ones
   ~5:1, so a 0.132 false-alarm rate makes revisions **net −113** across the arm, positive in
   one cell of nine.
4. **On `edge_existence` the loop is destructive** — −0.118 on adjacency (0.938 → 0.803),
   while the scaffold there is neutral. Replaying `turn1_only` on the traces recovers +0.118.
5. **Encoding fragility is large** and unanimous (`incident` best, `friendship` worst on all
   three tasks): baseline spreads 0.470 / 0.237 / 0.113.
6. **Reasoning reduces fragility; debate is not why.** Spread (max−min) by stage:

   | task | baseline | turn 1 | debate | MV(3) |
   |---|---|---|---|---|
   | connected_nodes | 0.470 | 0.292 | 0.253 | 0.265 |
   | node_degree | 0.237 | 0.113 | 0.107 | 0.107 |
   | edge_existence | 0.113 | 0.178 | 0.223 | 0.163 |

   The narrowing replicates 3/3 on both fragile tasks; the widening on `edge_existence`
   replicates 3/3. **MV matches debate's reduction** and is less harmful on `edge_existence`,
   so fragility repair is a property of reasoning, not of multi-agent structure. It works
   mostly by lifting the worst encoding (friendship 0.488 → 0.687) — though the best drifts
   down slightly (incident 0.958 → 0.940), so "lifts the worst without lowering the best" is
   not cleanly true.
7. **Turn-1 errors are over-inclusion** in every task: extra nodes far more often than
   missing (0.330 vs 0.033 on `connected_nodes/friendship`), positive signed degree error in
   all encodings, +0.181 yes-bias on `edge_existence/friendship`. Worst under `friendship`,
   the same encoding that is worst on accuracy.
8. **Prompt sensitivity is non-trivial.** An earlier Proposer/Critic wording (v2) scores
   **+0.041 higher** than v3 (0.884 vs 0.843) but produced 319 unparseable verdicts of 5,677
   against v3's **zero**. v2's Critic was more conservative (FA 0.050 vs 0.132), which is why
   it was more accurate. More accurate, worse behaved.

## 4. Where the data is

**All of `results/` is gitignored and exists only on the local machine.** The six Llama dirs
are 26 MB raw / **3.8 MB gzipped** — committing them is an open action item.

### Relevant (the paper)

| dir | contents |
|---|---|
| `results/llama70b-{main,seed11,seed13}` | baseline 1,800 + **v2** debate 1,800 each |
| `results/llama70b-v3-{main,seed11,seed13}` | **v3** debate 1,800 + majority_vote_cot 5,400 draws each |

Baseline rows live only in the first three; v3 debate and the vote arm only in the last
three. **Every analysis command passes all six** with `--prompt-version v3` — required, not
optional, because the dirs hold two prompt versions and the tooling refuses to pool them.

```bash
python scripts/show_results.py results/llama70b-main results/llama70b-seed11 \
    results/llama70b-seed13 results/llama70b-v3-main results/llama70b-v3-seed11 \
    results/llama70b-v3-seed13 --prompt-version v3 --fragility --by-seed
python scripts/debate_diagnostics.py <same six dirs> --prompt-version v3
python scripts/show_results.py <same six dirs> --prompt-version v3 \
    --vote-condition majority_vote_cot --compare
```

### Legacy (Qwen arm — a different paper's worth of data)

`results/{main,seed11,seed13}` and `results/v2-{main,seed11,seed13}` — Qwen2.5-3B, backing
[findings.md](findings.md). Its debate rows are **v2 only**, so its numbers are not
comparable to the Llama v3 arm. `results/main` holds the project's **only terse
majority-vote data** (18,000 rows) — the arm deliberately *not* re-run on Llama, since it
controls for the baseline rather than for debate.

### Archived (`results/_archive/`, see its README)

Three token-cap pilots, and `llama70b-v3prop-v2crit-*` — 5,400 rows **mislabelled** as v3
that are actually v3-Proposer + v2-Critic (see §6 below). Archived so `results/llama70b-v3*`
matches only the real run.

### Deleted

`debate-pilot-pv2-{cn,ee}` (Qwen, 200 rows each, superseded, uncited).

## 5. Narrative — where the discussion stopped

Two candidate framings, both inside the proposal's direction. **You preferred B.**

- **A′ — verification vs. aggregation** (proposal §2.3). Leads with the asymmetry: prior
  nulls come from domains where checking isn't objective; graphs remove that; the asymmetry
  holds here (Critic discriminates); debate still loses to voting. Strongest numbers
  (p=2e-16, no cell lost, measured compute ratio).
- **B′ — does debate fix encoding fragility** (proposal title). Leads with the 0.470 gap,
  asks what repairs it, answers "reasoning does, debate isn't why."

**B costs more, in four ways** — three cheap, one real:

1. Inverts the proposal's own metric priority (accuracy primary, spread secondary). Needs a
   sentence.
2. **Fragility has no significance test.** `report.fragility()` returns max−min with no
   interval; `stats.py` has McNemar/Cochran/Wilson but nothing for "is A's spread smaller
   than B's."
3. B's central claim is an **equivalence** (MV ≈ debate on fragility: 0.265 vs 0.253, 0.107
   vs 0.107). Claiming "the same" from a non-significant difference needs a CI tight around
   zero.
4. The ceiling caveat becomes load-bearing (`connected_nodes/incident` at 0.958). Defence is
   `node_degree`: best cell 0.873, not at ceiling, still gains +0.062 while spread falls
   0.237 → 0.107.

**Items 2 and 3 are one fix:** a paired bootstrap resampling *graphs* (instances are already
paired across encodings by `stats.graph_key`), giving CIs on both the reduction and the
debate-vs-vote difference. ~1 hour. **This is the immediate next action** — build it before
the paper leans on fragility.

Staged plan agreed: narrative → high-level flow → sections/subsections → write. We are at
the end of narrative.

## 6. Gotchas the draft must respect

- **Always pass `--prompt-version`.** Two versions exist; the tooling raises rather than
  pooling. v2 is frozen and byte-guarded by a test; v3 is current.
- **A `prompt_version` threading bug** (`conditions/debate.py` called `critic_prompt()`
  without the version, silently falling back to the default) produced a v3-Proposer/v2-Critic
  hybrid run. Fixed in `e9746444`, which made the argument mandatory on all three builders.
  The hybrid is archived and is what establishes the preamble change is a wash (0.879 vs
  0.884). Diagnosed by re-tokenizing stored prompts against the served model.
- **`edge_existence` is a control**, not a third fragile task. Report both scopes; never
  exclude silently.
- **The proposal (§3.4) said 7–8B models on the university cluster**; the headline arm is
  Llama-3.3-70B via Together. The Qwen-3B runs exist, so this can be framed as scaling up,
  but it needs a sentence.
- **Together documents no determinism guarantee**, so sampled draws record the seed they
  asked for rather than one that replays. Greedy reproduces in practice.
- **`docs/paper/` is stale** and will be rewritten from scratch — ignore it.

## 7. Open work items

1. **Bootstrap CIs for cross-encoding spread** (blocks the B framing).
2. **Commit the six Llama run dirs** — 3.8 MB gzipped; makes findings' "regenerates from
   committed run outputs" literally true. Currently one disk from gone.
3. Terse majority vote was never run on Llama (~$2 at N=3). It controls for the baseline, not
   for debate; probably not needed.
4. Untested hypothesis worth one cell: v2's Critic wording (more conservative, more accurate)
   with v3's truncation fix — might beat both.
