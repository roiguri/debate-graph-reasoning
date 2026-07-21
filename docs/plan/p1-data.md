# P1 — Data layer (vendor GraphQA + thin adapter)

**Status: complete.** All substeps below done; see `src/gedebate/graphqa/` +
`src/gedebate/data/` and `tests/test_dataset.py`.

Goal: produce graph-reasoning instances — graph, encoded text, question, and
ground-truth answer — using the **official GraphQA code**, wrapped in a thin
adapter our framework can consume. **No GPU** — pure local Python, testable.
See [overview.md](overview.md) for where this fits.

**Decision (revised — supersedes the earlier "reimplement" decision):** vendor
the official GraphQA modules instead of reimplementing. Rationale and the reversal
are recorded in [../notes.md](../notes.md#use-official-graphqa-code-decision-revised).
In short: the relevant modules are **pure `networkx`/`numpy`** (no TF), tiny, and
Apache-2.0 — so vendoring gives byte-exact fidelity to the paper (encodings,
question phrasing, generation sampling) with none of the assumed heavy deps.

**Done when:** given a seed, `build_dataset` yields instances for the 3 tasks ×
3 encodings over ER graphs, each JSON-serializable with a normalized ground
truth; tests confirm exact encoder/task wording and determinism.

## What we vendor vs build

**Vendor** into `src/gedebate/graphqa/` (Apache-2.0 headers kept; only import
paths changed; provenance in a NOTICE — see licensing below). These 4 files form
a closed pure-python subset:
- `graph_generator_utils.py` — `generate_graphs()`; ER samples n∈{5..19}, p∈U[0,1].
- `name_dictionaries.py` — node-name tables (integer, popular, ...).
- `graph_text_encoder.py` — the encoders (`encode_graph`, `TEXT_ENCODER_FN`).
- `graph_task.py` — task classes with exact question wording + NetworkX ground truth.

**Do NOT vendor** (TensorFlow / CLI, not needed): `graph_generator.py`,
`graph_task_utils.py`, `graph_task_generator.py`, the `.sh` runners. We replace
that dataset-writing layer with our own thin builder.

**Discard** our now-superseded reimplementation: `data/generators.py`,
`data/encoders.py`, `data/naming.py`, and their tests (kept in git history).
`data/registry.py` may stay only if the thin adapter still needs it; otherwise
it goes too.

## Substeps

### P1.a — Vendor the 4 modules + licensing
- Copy the 4 files into `src/gedebate/graphqa/`, keep each Apache-2.0 header.
- Only change: rewrite intra-package imports `from graphqa import X` →
  `from gedebate.graphqa import X` (record this as the sole modification).
- Add `src/gedebate/graphqa/NOTICE.md`: source repo + commit, Apache-2.0, the
  import-path change, and the Fatemi et al. 2024 citation. Add the Apache-2.0
  `LICENSE` text alongside the vendored code.
- Pin the upstream commit hash for reproducibility.

### P1.b — Thin adapter (`src/gedebate/data/`)
- `graphs.py`: `generate(algorithm, n_graphs, seed) -> list[nx.Graph]` wrapping
  `generate_graphs` (default `algorithm="er"`; other generators available but out
  of scope). Owns our seed convention.
- Selection layer restricting scope to our config:
  - encodings: `{"adjacency", "incident", "friendship"}` (subset of GraphQA's).
  - tasks: `{"edge_existence", "node_degree", "connected_nodes"}` → the matching
    GraphQA task classes.

### P1.c — Instance schema (`src/gedebate/data/instance.py`)
- A JSON-serializable dataclass built from GraphQA's `prepare_examples_dict`
  output: `{task, encoding, algorithm, question, answer, nnodes, nedges,
  node_ids, graph_edgelist, seed}`.
- Add a **normalized ground truth** alongside GraphQA's answer string (bool for
  edge_existence, int for node_degree, sorted list for connected_nodes) so P2's
  scorer can compare robustly, not just string-match.
- `build_dataset(config, seed)`: generate graphs once, then for each task ×
  encoding call the GraphQA task's `prepare_examples_dict` and package instances.

### P1.d — Tests (`tests/`)
- Characterization: vendored encoders produce the exact expected string on a
  known graph (now byte-exact from source, incl. adjacency preamble, "among
  nodes", isolated-node handling, `_POPULAR_NAMES` order).
- Ground truth: known small graphs → correct answers for all three tasks, and our
  normalized ground truth agrees with the GraphQA answer string.
- Determinism: same seed → identical instances; JSON round-trip.
- (Light touch — we don't deep-unit-test upstream code; just pin the behavior we
  depend on.)

## Notes for later phases
- The instance's **normalized ground truth** is the contract P2's answer-parser
  scores against.
- GraphQA question phrasing is the "graph" question encoder; the "application"
  variant is a knob we can revisit later (see notes.md).
- Scope stays ER × {3 tasks} × {3 encodings}; the vendored code supports more,
  gated by our selection layer.
