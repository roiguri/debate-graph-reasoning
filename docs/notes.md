# Notes & Decisions

A running log of **reference facts** (from the literature) and **design
decisions** (with rationale), kept separate from the phase plans in
[plan/](plan/) so those stay about steps. Much of this feeds the methodology
section of the final writeup. Newest decisions can go at the bottom of each
section.

---

## Data generation

### GraphQA generation parameters — fact
We use the vendored `generate_graphs`, so these come for free:
- `n` (node count) ~ uniform over **{5..19}** per graph (the code's size buckets
  are 5–9 / 10–14 / 15–19; the paper's prose says "5 to 20").
- `p` (ER edge probability) ~ uniform over **[0, 1]** per graph — deliberately
  spans empty→complete so the set isn't tuned to one density.
- Graph **count is our knob** (`n_graphs`); the paper used ~500 per generator.
- NetworkX used for both graph generation **and** ground-truth answers. Their eval
  used decoding **temperature 0** (greedy).

### Graph generation is not N-extensible — fact (P3)
`generate_graphs(number_of_graphs=N, seed)` draws all N size-choices in **one**
up-front `random.choices(..., k=N)` batch, which offsets the per-graph RNG stream
that follows. So changing N re-samples the **entire** graph sequence: the first
100 graphs of an N=200 build are NOT the first 100 of an N=100 build (verified).
This is why the dataset is now a **frozen committed artifact** (`data/main.jsonl`,
see [dataset refactor](plan/refactor-dataset.md)) that every run loads rather than
regenerates: results are anchored to the artifact, not to implicit generator
determinism. The run manifest guards `dataset_sha256` (the artifact's content hash),
so a resume against a different dataset is a hard error. To get more samples, add an
independent **seed** as a **sibling artifact** (`build_dataset.py --init --name
seedK --seed K`) rather than growing `main` in place: this keeps `main`'s hash and
every existing manifest guard intact, and `instance_id`s (namespaced by
`dataset_seed`) never collide when pooled. Realized: seeds 11 and 13 (see the
replication result below).

### ER only (structure held constant) — decision
GraphQA studies graph *structure* as a variable (ER, BA, SFN, SBM, star, path,
complete). We hold it fixed at **ER** because our question varies *encoding ×
debate*, not structure — so structure is a control. Other generators remain
available via the vendored `generate_graphs(algorithm=...)` if we later want
structure as a variable; our adapter just gates to `er`.

### Use official GraphQA code — decision (revised)
**Supersedes the earlier "reimplement, don't vendor" decision.** That call
assumed vendoring meant pulling heavy TF/JAX deps. On inspecting the actual repo
(`google-research/google-research/tree/master/graphqa`, Apache-2.0), that was
wrong: the modules we need are **pure `networkx`/`numpy`**. TensorFlow appears
only in the dataset-writing CLI layer, which we don't use.

So we **vendor** the 4 pure files (`graph_generator_utils.py`,
`name_dictionaries.py`, `graph_text_encoder.py`, `graph_task.py`) and build a
thin adapter on top. This gives **byte-exact fidelity** to the paper — encoder
wording, question phrasing, and generation sampling (n∈{5..19}, p∈U[0,1]) — which
removes every "is this what the paper used?" caveat below. The earlier
reimplementation (our `generators.py`/`encoders.py`/`naming.py`) is discarded
(kept in git history). See [plan/p1-data.md](plan/p1-data.md).

**Licensing:** Apache-2.0 permits vendoring with attribution. We keep each file's
license header, add a NOTICE (source repo + pinned commit + our import-path
change), and include the Apache-2.0 LICENSE text. Cite Fatemi et al. 2024.

**Fidelity notes now moot:** the encoder-phrasing and name-list caveats recorded
under "Tasks & encodings" are resolved by using the source directly (e.g.
adjacency *does* carry the "(i,j) means…" preamble; `_POPULAR_NAMES` interleaves
male/female from index 5; friendship says "among nodes …"; incident *skips*
isolated nodes). The HF dataset `baharef/GraphQA` is not used — it ships single-
encoding pre-baked prompts, which can't give same-graph-across-encodings control.

---

## Tasks & encodings

### The three encodings are maximally distinct — decision
adjacency / incident / friendship are the three GraphQA encodings that differ on
**both** axes (node naming × edge phrasing): integers+tuples, integers+neighbor-
lists, names+sentences. GraphQA's other six (co-authorship, social network, GOT,
SP, politician) are near-duplicates of friendship (named nodes + relational
sentences, differing only in cover story). Three covers the real variation.

### "connected nodes", not "connectivity" — decision
The proposal's "connectivity" task = GraphQA's **connected nodes** (list a node's
neighbors), i.e. the headline 19.8%→53.8% task. Renamed in code/writeup to avoid
confusion with reachability ("are u and v connected?").

### Task encoding-sensitivity spread — fact
From Fatemi 2024 Table 1 (PaLM 62B, zero-shot), our three tasks span the range:
- **edge existence** — barely sensitive to encoding (spread δ≈9)
- **node degree** — moderately sensitive (incident 25 vs friendship 11)
- **connected nodes** — highly sensitive (adjacency 19.8 vs incident 53.8)

### Encoding wording is exact (we run the source) — fact
Because encoders are the vendored `graph_text_encoder.py`, wording is byte-exact
to GraphQA — no approximation to track. Non-obvious specifics worth knowing:
- **adjacency** carries the preamble *"In an undirected graph, (i,j) means that
  node i and node j are connected with an undirected edge. "* (the paper's Figure
  2 hides it in a two-column layout; the code has it).
- **incident** handles singular/plural ("node" vs "nodes") and **omits isolated
  nodes entirely** (no line for a 0-degree node).
- **friendship** literally says *"…friendship graph among **nodes** James, …"*
  and uses `_POPULAR_NAMES`, which interleaves male/female from index 5
  (0→James … 5→Mary … 10→William …).

### Encoder assumptions (held by the vendored code) — note
- Node labels are contiguous `0..n-1` (name dicts map id → name positionally).
  Our `generate_graphs` always yields `0..n-1`, so this holds.
- No self-loops (encoders don't special-case `(i, i)`); ER never produces them.

### Edge existence is both a task and the Critic's atomic check — note
The Critic verifies edge-presence claims, and edge existence is also the *least*
encoding-sensitive task. That's a feature: it means the verification sub-task is
reliable, which is exactly the asymmetry the method assumes. State this in the
writeup.

---

## Models & compute

### VRAM ceiling: 11 GB — constraint (affects P2/P3 model choice)
The cluster GPUs are RTX 2080 **Ti**, 11 GB. A 7–8B model in fp16 needs ~16 GB
and **will not fit**. Options at P2/P3: a ≤3B model (e.g. Qwen2.5-3B, ~6 GB),
4-bit quantization of a 7B (~5 GB), or a larger GPU if the allocation has one.
The proposal's "7–8B" needs revisiting here. Smoke used Qwen2.5-1.5B (fits easily).

### Inference stack — fact
In-process HuggingFace Transformers (not vLLM), greedy by default. Generated-
token count = tokenizer output length; this feeds the matched-compute comparison.

### Cluster quota: 24 GPUs/user, no job cap — fact (P3)
QOS `gpustudentkill`: `MaxTRESPU = gres/gpu=24` (up to 24 concurrent GPUs per
student), **no** `MaxJobsPU`/`MaxSubmitPU` cap; partition `MaxTime = 24h`. So GPU
phases can fan out wide — P3's 8-shard array runs fully concurrent; the expensive
P4/P5 conditions can shard to ~24 single-GPU jobs. (Association-level limits are
empty; the QOS holds the only real cap.)

### Baseline model: Qwen2.5-3B-Instruct, config-owned — decision (P2)
Start with **Qwen2.5-3B-Instruct** (fp16, ~6 GB — fits 11 GB with KV headroom, no
quantization confound perturbing the very fragility we study). Deviates from the
proposal's "7–8B" purely for VRAM; state this in the writeup. The model id lives
in **one place — the run's TOML config** — so swapping to a 4-bit 7B (if the
P2/P3 pilot shows floor effects) is a one-line change. Proposer and Critic will
be the same model in two roles (simplest, keeps matched-compute clean).

---

### Compute metric: # responses + total tokens, per instance — decision (P5.0)
The matched-compute comparison across conditions uses **# model responses (calls)** as
the primary metric (the debate-vs-self-consistency literature standard: Huang 2024
"equivalent number of responses", Choi 2025 "number of agents", Du 2023), with **total
tokens (prompt + generated)** as the secondary, faithful measure. **Generated-only was
dropped**: for these tasks the prompt dominates hard -- a baseline instance averages
~310-430 *total* tokens but only ~2 generated (the encoding is the whole cost), so
generated-only measured under ~1% of the work. Reported **per instance** (mean cost of
one question): baseline = 1 response / ~360 tokens; MV = 10 / ~3,600 (a clean 10x on both
axes, since MV is N independent copies -- so the P4 "10x for nothing" holds in total
tokens too). Debate will place as a point on MV's accuracy-vs-#responses curve; at matched
responses its total tokens exceed MV's (the transcript grows each turn), which is why we
report both. All recoverable from stored rows (`n_prompt_tokens` + `n_gen_tokens`), so no
rerun. Surfaced by `report.summarize`/`summarize_votes`/`compare_baseline_vote` and
`show_results.py`.

## Prompting & scoring

### Baseline prompt = GraphQA zero-shot, direct answer — decision (P2)
Faithful to Fatemi 2024's headline (Table 1 **ZERO-SHOT** row): a raw
`…Q: ‹question›\nA: ` completion, terse answer, **no CoT, no few-shot**. That row
is what produces the encoding-fragility spread (edge-existence µ/δ 44.5/9.4, node
degree 14.0/16.0, connected nodes 14.7/11.0). We keep GraphQA's question verbatim
(already our `Instance.question`) and add only a **minimal terse-format
instruction** so an instruct-*chat* model (Qwen2.5-3B) emits the same short answer
shape a raw-completion model (PaLM) did after `A: `. The chat-template-vs-raw-
completion gap is the one deliberate adaptation; record it in the writeup.
Rationale for not adding CoT to the baseline: the baseline is *meant* to be the
weak reference — debate/vote are where accuracy should climb. Floor-effect risk
on a 3B model is real; the P2 pilot sanity-checks we're not at rock-bottom-zero
everywhere (which would threaten P3's "reproduce fragility" premise).

### Scoring = exact match (incl. connected_nodes full set) — fact + decision (P2)
The paper's accuracy (`scoref`) is **whole-answer exact match**; the 19.8/53.8
connected-nodes numbers are the fraction of graphs where the model's neighbor set
equals ground truth **exactly** (the 0.5% disconnected-nodes result confirms no
partial credit). We mirror it: parsed value `==` normalized `ground_truth`, with
**set-equality** for connected_nodes. **No Jaccard / partial credit** in the
reported metric. We do track a separate **`parse_ok`** flag so parse failures are
visible and measurable rather than silently scored as wrong (a high parse-failure
rate is a confound to watch, not a result).

---

## Experiment harness

### Persistence contract — decision (P2)
The results format is a keystone P4/P5 inherit; two constraints shaped it — the
`studentkillable` partition kills jobs mid-run, and matched-compute is defined as
total generated tokens per instance per condition.
- **Atomic unit = one completed *attempt*, written once at attempt end** (not one
  generation). Baseline = 1 row/instance, majority-vote = N rows (one per sample),
  debate = 1 row with tokens **summed** across all Proposer+Critic calls. Chosen
  over generation-level rows because a debate killed mid-loop would otherwise leave
  partial generations that double-count tokens on re-run; attempt-level rows are
  all-or-nothing, so a kill leaves no row and re-running is clean. Trade-off
  accepted: a killed debate redoes the whole (short, token-capped) loop.
- **One uniform, lean row schema for all three conditions**, so baseline's row *is*
  the final schema — nothing reopened. The verbose debate trace lives in a
  **sidecar** (`traces/{instance_id}.json`), keeping the main JSONL uniform;
  analysis never needs it for primary numbers.
- **Resume** = "instance done under condition C when it has ≥ `expected_attempts(C)`
  rows" (1 for baseline/debate, N for MV). One JSONL **file per shard**; done-ids
  read from the **union of all `*.jsonl` in the run**, so re-sharding never redoes
  work. Kill tolerance: flush per row (lose ≤1 in-flight attempt); reader drops a
  torn trailing line. A per-run `manifest.json` snapshots config/model/commit and
  guards against mixing two models' rows into one accuracy.
- **Majority-vote stores one row per sample** (not an aggregated blob): resume tops
  up missing sample indices, and it exposes individual-sample-vs-vote accuracy — a
  number worth reporting (does the vote actually beat the average single draw?).

### P3 result: encoding-fragility reproduces — fact + GO decision (P3.4)
Baseline over the full 3×3 at N=200 (Qwen2.5-3B, greedy, seed 7; 1800 instances,
`parse_ok` 0.99–1.00). **Premise confirmed → GO for P4.** Per-cell accuracy:

| task | adjacency | incident | friendship | spread (max−min) |
|---|---|---|---|---|
| edge_existence | 0.70 | 0.68 | 0.68 | **0.02** (insensitive) |
| node_degree | 0.37 | **0.75** | 0.46 | **0.38** |
| connected_nodes | 0.28 | **0.345** | 0.21 | **0.135** |

Reproduces Fatemi directionally: **incident is best for both node_degree and
connected_nodes**, and **edge_existence is encoding-insensitive**. Spreads are well
outside noise (N=200 → per-cell 95% CI ≈ ±0.07): node_degree's 0.38 gap is many σ;
edge_existence's 0.02 is noise, as expected. No floor/ceiling (all in 0.21–0.75).
- **Encoding-fragile targets for P4/P5** (worst cells to lift): `node_degree ×
  adjacency` (0.37 vs incident 0.75) and `connected_nodes × friendship` (0.21). The
  debate question is whether verification lifts the *worst* encoding specifically,
  not everything uniformly.
- ~4/1800 parse misses (connected_nodes) — negligible; can inspect with
  `--raw --wrong-only` if ever needed.

### P3 result: fragility replicates across seeds + is significant — fact
The seed-7 result above could be a quirk of those particular 200 graphs, so we
ran the same baseline on two **independent** seeds (11, 13), each a fresh N=200
draw from the identical generating process (built as sibling artifacts
`data/seed11.jsonl`, `data/seed13.jsonl`; frozen `main` untouched). Pooled to 600
graphs/cell:

| task | adjacency | incident | friendship | spread | omnibus (Cochran Q) | best>worst (McNemar) |
|---|---|---|---|---|---|---|
| edge_existence | 0.703 | 0.690 | 0.695 | 0.013 | Q=0.5, p=0.77 (ns) | p=0.52 (ns) |
| node_degree | 0.388 | **0.750** | 0.458 | 0.362 | Q=200, p=3e-44 | incident>adjacency 258/41, p=8e-36 |
| connected_nodes | 0.280 | **0.373** | 0.263 | 0.110 | Q=50, p=2e-11 | incident>friendship 90/24, p=1e-9 |

**Method (writeup point):** the three encodings are applied to the *same* graphs,
so per-graph correctness is **paired**, not independent. Significance is therefore
paired: Cochran's Q (omnibus, does any encoding differ) and McNemar (the headline
best-vs-worst gap); an unpaired two-proportion test would be wrong here. Per-cell
accuracy carries a 95% Wilson CI. All in `gedebate.eval.stats`, surfaced by
`show_results.py --fragility` and pooled by passing multiple run dirs.

**What replicates (per-seed `--by-seed` view):**
- **node_degree: clean replication.** `incident` best and `adjacency` worst in all
  three seeds independently. This is the strongest not-coincidental result.
- **connected_nodes: core claim replicates.** `incident` best in all three seeds
  (p=1e-9 pooled). The *worst* encoding wobbles (friendship in seeds 7/13,
  adjacency in seed 11): adjacency and friendship are statistically
  indistinguishable pooled, so report "incident helps", not "friendship is worst".
- **edge_existence: null confirmed.** Gap ~0.01, ns in every seed, and the
  best/worst labels differ across seeds (the signature of no real effect). Extra
  data did not manufacture one.

Pooling is by (seed, graph_index): different seeds' graph 0 are different graphs.
(An earlier version keyed on graph_index alone, which silently collapsed the pool
by half; fixed + regression-tested. The tripled discordant counts, e.g. node_degree
89/13 at one seed to 258/41 at three, confirm all 600 graphs are counted.)

### P4 result: majority vote does not beat greedy, does not close fragility — fact (P4.3)
Majority vote (self-consistency) over the full 3x3 at N=10, temperature=0.7, seed 7
(18,000 samples over the same frozen `data/main.jsonl` the baseline scored, parse_ok
0.995-1.00). The voted answer per instance is the mode of the 10 parsed draws
(parse-failures excluded, ties broken to the lowest-index supporter). **Voting
reproduces the greedy baseline almost exactly and leaves the encoding-fragility gap
intact, at ~10x the generated-token cost.** Per cell, greedy baseline vs voted
accuracy (Δ = vote − baseline):

| task | adjacency | incident | friendship |
|---|---|---|---|
| edge_existence | 0.700 → 0.700 (0.000) | 0.680 → 0.675 (−0.005) | 0.680 → 0.680 (0.000) |
| node_degree | 0.370 → 0.360 (−0.010) | 0.750 → 0.755 (+0.005) | 0.455 → 0.455 (0.000) |
| connected_nodes | 0.280 → 0.275 (−0.005) | 0.345 → 0.350 (+0.005) | 0.210 → 0.215 (+0.005) |

Every |Δ| ≤ 0.01, far inside the per-cell 95% CI (≈±0.07). A **paired McNemar** of vote
vs greedy per cell (both run on the same instances) makes the null rigorous, not just
overlapping CIs: the discordance is 1-9 instances out of 200 per cell and balanced
(b≈c), so **every cell is non-significant** (p ≥ 0.69; eight of nine at p=1.0). Voting
returns greedy's exact correctness on 96-99% of instances. Fragility gap (max−min per
task) is unchanged: node_degree 0.38 → 0.395, connected_nodes 0.135 → 0.135,
edge_existence 0.02 → 0.025. The worst encodings (node_degree × adjacency,
connected_nodes × friendship) are not lifted. (Per-cell b/c + p in
`analysis/main/mv_vs_baseline.csv`, from `show_results.py --compare`.)

**Why (writeup point).** The reason is structural, not incidental: our task answers are
essentially single tokens (a yes/no, a degree; connected_nodes a short set), and
**majority vote over samples of a single-token answer converges to the argmax, which is
exactly greedy**. Self-consistency's leverage comes from marginalizing over *diverse
reasoning paths* (Wang et al. 2023); with direct terse answers there are no diverse paths
to marginalize, so the vote just re-estimates greedy's mode. Consistent with this, the
single sampled draw (`1samp`) sits slightly below greedy where T=0.7 sampling adds noise
(connected_nodes × incident 0.333 vs 0.345), and voting recovers it back to ~greedy,
netting ~0. The encoding-induced errors are also systematic across draws, so aggregation
cannot repair them. (Sampling used Qwen's recommended top_p=0.8/top_k=20, now set
explicitly; a wider-diversity rerun was judged unnecessary because the single-token
argument holds regardless of sampling diversity. See docs/plan/p4-review-followups.)

**Compute.** MV spent ~83.5k generated tokens vs the baseline's ~8.5k (≈9.8x, i.e.
~10x as designed since N=10), for zero accuracy return. Per-cell totals are in the
`gen_tok` column; the connected_nodes cells dominate (~19-21k each) because listing
neighbors is longer than a yes/no or a single degree.

**Decision: GO for P5 (debate), with a sharpened hypothesis.** Spending compute on
*more samples of the same reasoning* buys nothing here and does not touch fragility, so
debate must earn its keep by changing the *reasoning* (Proposer-Critic verification),
not by resampling. The matched-compute bar is now concrete: at ~10x baseline tokens
MV = greedy, so debate has to beat greedy at a budget where the trivial compute
baseline already fails. MV is therefore the "compute alone does nothing" control the
writeup needs. (Optional P4.4: replicate the null on seeds 11/13 if we want the same
paired-significance treatment P3 got; a near-zero effect this uniform is unlikely to
change, so it is low priority.)

### P2 pilot result — fact (P2.5)
First real-GPU run (Qwen2.5-3B-Instruct, RTX 2080 Ti 11GB, greedy). Validates the
harness end-to-end; **not** a measurement (N=8–20, noisy). Key facts for the writeup:
- **The model fits and the parser is faithful to real output.** `parse_ok = 1.000`
  in all 9 task × encoding cells; manual inspection confirms correct extraction
  (terse integers, name→id mapping, source-node dropped, "none" → empty set). Every
  scored error is a genuine model error, not a parse artifact. So P3 is a config
  change (bump `n_graphs`), no parser work.
- **No floor effect**; accuracy spans 0.25–0.90.
- **Fragility is already visible even at N=8** (matches Fatemi's direction): edge
  existence is easiest and encoding-insensitive (adjacency/friendship 0.875,
  incident 0.75); connected_nodes is hardest and encoding-sensitive (adjacency
  0.625 > incident 0.50 > **friendship 0.25**); node_degree middling (adjacency
  0.375 vs incident/friendship 0.75). Real N in P3 will measure this properly.
- **Prompt behaves:** the terse-format instruction lands — outputs are bare (`"3"`,
  `"none"`, clean name lists), so baseline gen-token counts are small (16–80).
- **Follow-up for P4:** Qwen ships `top_p=0.8/top_k=20` in its generation_config;
  harmless under greedy (warned + ignored), but sampling params must be set
  explicitly once majority-vote actually samples.
