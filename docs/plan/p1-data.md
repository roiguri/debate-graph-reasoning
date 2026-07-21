# P1 — Data generation helpers

Goal: generate graphs with ground-truth answers, serialize them under the three
encodings, all unit-tested. **No GPU** — pure local Python, fully testable.
See [overview.md](overview.md) for where this fits.

**Decision (locked):** reimplement the data layer cleanly on NetworkX rather
than vendoring the GraphQA research code. Each encoder is ~10 lines and each
ground-truth is a NetworkX one-liner; vendoring would pull in TF/JAX-era deps for
a handful of string functions. GraphQA [Fatemi et al. 2024] is cited as the
design source.

**Done when:** given a seed, we can produce a list of instances — each carrying
graph, task, query, encoding name, encoded text, and ground-truth answer — and
the test suite passes (encoders match hand-verified examples, ground truth
matches known graphs, same seed → identical instances).

## Design: registry of independent components

The data layer factors into three independent, pluggable pieces so new
encoders/tasks/generators are one addition each (the extensibility ask):

```
generator ──> nx.Graph
task ──────> (query, ground-truth answer)   # given the graph
encoder ───> str                            # given the graph
                                            # instance = all of the above
```

Encodings themselves decompose into two axes — **node naming** (integers vs
names) × **edge phrasing** (tuples / neighbor-lists / sentences). We implement
the three chosen encodings as plain functions for now; the naming×phrasing
combinator is a later option only if we explore beyond the three.

## Substeps

### P1.1 — Graph generation
- `src/gedebate/data/generators.py`: Erdős–Rényi generator (matches the paper),
  `gnp_random_graph(n, p, seed)`. Parametrize `n` range and `p`.
- Registry: `@register_generator("er")`. Keep the seam for BA/star/path later.
- Return plain `nx.Graph` (undirected, matching the tasks chosen).

### P1.2 — Encoders (the three chosen)
- `src/gedebate/data/encoders.py`, each `nx.Graph -> str`:
  - `adjacency` — integer nodes; edges as `(0, 1) (0, 2) ...`
  - `incident` — integer nodes; `Node 0 is connected to nodes 1, 2.`
  - `friendship` — named nodes (fixed name list, James/Robert/...); `James and
    Robert are friends.`
- Registry: `@register_encoder("...")`. Names come from a shared naming table so
  friendship's node→name mapping is deterministic.
- Match the phrasing in the paper's Figure 2 closely (verbatim style) so results
  are comparable to Fatemi et al.

### P1.3 — Task functions + ground truth
- `src/gedebate/data/tasks.py`, each task provides `query(G, rng)` (pick what to
  ask) and `answer(G, query)` (NetworkX ground truth):
  - `edge_existence` — pick a node pair; answer `G.has_edge(u, v)`.
  - `node_degree` — pick a node; answer `G.degree(n)`.
  - `connected_nodes` — pick a node; answer `sorted(G.neighbors(n))`.
    (Renamed from the proposal's "connectivity" to match the paper and avoid
    confusion with reachability — this is the 19.8%→53.8% neighbor-listing task.)
- Registry: `@register_task("...")`.
- Query phrasing lives here too (the question text shown to the model), using
  the same node-naming as the encoding so integer/name tasks stay consistent.

### P1.4 — Instance schema
- `src/gedebate/data/instance.py`: a dataclass tying it together —
  `{graph_repr, generator, seed, task, query, encoding, encoded_text,
  question_text, ground_truth}`.
- JSON-serializable (this is what P2's eval harness consumes and what lands in
  `results/`). Store the graph as an edge list so instances are reproducible
  without re-running the generator.
- A `build_dataset(config, seed)` that iterates generator × task × encoding and
  yields instances.

### P1.5 — Tests (`tests/`)
- Encoders: hand-verify a tiny 3-node graph's string under each encoding.
- Ground truth: known small graphs → assert exact answers for all three tasks.
- Determinism: same seed → identical instance list (graphs, queries, encodings).
- Query validity: e.g. edge_existence sometimes picks present and sometimes
  absent pairs (so accuracy isn't dominated by one class — the paper notes ~54%
  of pairs are non-edges).

## Notes for later phases
- The instance schema is the contract P2's answer-parser scores against — keep
  `ground_truth` in a canonical form (bool / int / sorted list) that a parser
  can compare to a model's parsed output.
- `question_text` phrasing (graph vs application encoder) is a knob the paper
  studied; we fix one style now and can revisit if results warrant.
