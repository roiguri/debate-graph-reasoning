"""Data layer: a thin adapter over the vendored GraphQA code.

Graph generation, encoders, tasks, and ground truth come from
`gedebate.graphqa` (vendored, Apache-2.0). This package adds only:
  - `dataset.py` — scope-limited selection (ER, 3 encodings, 3 tasks) and
    `build_dataset`, which fixes the query per (graph, task) across encodings.
  - `instance.py` — the JSON-serializable Instance record + normalized ground truth.
"""
