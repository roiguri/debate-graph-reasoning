# Notes & Decisions

A running log of **reference facts** (from the literature) and **design
decisions** (with rationale), kept separate from the phase plans in
[plan/](plan/) so those stay about steps. Much of this feeds the methodology
section of the final writeup. Newest decisions can go at the bottom of each
section.

---

## Data generation

### GraphQA generation parameters — reproduce in P1.4
Source: Fatemi et al. 2024, Appendix A.4.
- `n` (node count) ~ uniform over **{5..20}** per graph.
- `p` (ER edge probability) ~ uniform over **[0, 1]** per graph — deliberately
  spans empty→complete so the set isn't tuned to one density.
- **~500** ER graphs sampled this way.
- NetworkX used for both graph generation **and** ground-truth answers (same as
  us). Their eval used decoding **temperature 0** (greedy).

### ER only (structure held constant) — decision
GraphQA studies graph *structure* as a variable (ER, BA, SFN, SBM, star, path,
complete). We hold it fixed at **ER** because our question varies *encoding ×
debate*, not structure — so structure is a control. The generator registry keeps
other generators available if we later want structure as a variable.

### Reimplement, don't vendor — decision
We reimplement generators/encoders/tasks on NetworkX rather than vendoring the
`google-research/talk-like-a-graph` code. Each encoder is ~10 lines and each
ground truth is a NetworkX one-liner; vendoring would pull TF/JAX-era research
deps for a handful of string functions. GraphQA is cited as the design source.

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
