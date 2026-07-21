# NOTICE — vendored GraphQA code

The files in this directory are vendored from the official GraphQA implementation
for the paper **"Talk like a Graph: Encoding Graphs for Large Language Models"**
(Fatemi, Halcrow, Perozzi; ICLR 2024).

- **Source:** https://github.com/google-research/google-research/tree/master/graphqa
- **Pinned commit:** `5b09c22d73a9d35eb6c5d2a99b95677a45053466`
- **Retrieved:** 2026-07-21
- **License:** Apache License 2.0 (see `LICENSE` in this directory). The original
  copyright headers are retained in each file.

## Vendored files (pure `networkx` / `numpy` — no TensorFlow)

- `graph_generator_utils.py` — random graph generation (`generate_graphs`).
- `name_dictionaries.py` — node-name tables (integer, popular names, ...).
- `graph_text_encoder.py` — graph→text encoders (`encode_graph`, `TEXT_ENCODER_FN`).
- `graph_task.py` — task classes: question wording + NetworkX ground truth.

The TensorFlow-dependent CLI / dataset-writing modules of the upstream repo
(`graph_generator.py`, `graph_task_utils.py`, `graph_task_generator.py`, and the
`.sh` runners) are intentionally **not** vendored; `gedebate` provides its own
thin builder in `src/gedebate/data/` instead.

## Modifications

The only change from upstream is rewriting intra-package imports:

- `from graphqa import name_dictionaries` → `from gedebate.graphqa import name_dictionaries`
  (in `graph_text_encoder.py`)
- `from graphqa import graph_text_encoder` → `from gedebate.graphqa import graph_text_encoder`
  (in `graph_task.py`)

No logic was altered.

## Citation

```bibtex
@inproceedings{fatemi2024talk,
  title={Talk like a Graph: Encoding Graphs for Large Language Models},
  author={Bahare Fatemi and Jonathan Halcrow and Bryan Perozzi},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
}
```
