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

### Baseline model: Qwen2.5-3B-Instruct, config-owned — decision (P2)
Start with **Qwen2.5-3B-Instruct** (fp16, ~6 GB — fits 11 GB with KV headroom, no
quantization confound perturbing the very fragility we study). Deviates from the
proposal's "7–8B" purely for VRAM; state this in the writeup. The model id lives
in **one place — the run's TOML config** — so swapping to a 4-bit 7B (if the
P2/P3 pilot shows floor effects) is a one-line change. Proposer and Critic will
be the same model in two roles (simplest, keeps matched-compute clean).

---

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
