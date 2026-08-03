# debate-graph-reasoning

**Does debate improve encoding-fragile LLM graph reasoning?**

LLMs reasoning over graphs are sensitive to how a graph is serialized into text: on
the same graph and task, accuracy can swing sharply with the encoding. This project
tests whether a Proposer-Critic debate makes graph reasoning robust to that choice,
and isolates whether any gain comes from debate itself or merely from spending more
compute. Built on [GraphQA](https://github.com/google-research/google-research/tree/master/graphqa)
(Fatemi et al. 2024).

## Approach

Three conditions, compared per task and per encoding at a matched token budget:

- **Baseline:** one zero-shot answer.
- **Majority vote:** N samples aggregated by vote (isolates the effect of extra compute).
- **Debate:** the Proposer emits atomic claims, a Critic verifies each against the
  raw encoding, and the Proposer revises.

Tasks: edge existence, node degree, connected nodes. Encodings: adjacency, incident,
friendship. The baseline condition is implemented; the vote and debate conditions
build on the same harness and dataset.

## Setup

```bash
pip install -e .             # dataset build/verify + analysis (CPU only)
pip install -e .[inference]  # + the model stack to run conditions (needs a CUDA GPU)
```

Running a condition needs a GPU that fits the model (Qwen2.5-3B is about 6 GB fp16);
CPU works but is slow.

## Reproduce

The dataset is a frozen, committed artifact (`data/main.jsonl`); every run loads it
rather than regenerating. Full walkthrough in [docs/reproduce.md](docs/reproduce.md):

```bash
python scripts/build_dataset.py --verify                          # verify the dataset reproduces
python -m gedebate.eval.runner --config configs/matrix.toml       # run the baseline over the matrix
python scripts/show_results.py results/main --fragility           # per-encoding accuracy + spread
```

## Layout

```
src/gedebate/   the library: data adapter, model wrapper, prompts, scoring, eval harness
data/           the frozen dataset artifact (+ provenance meta)
configs/        run configs (TOML)
scripts/        entry points (build_dataset, show_results, debate_viewer, smoke)
slurm/          batch-job scripts, specific to our compute setup
results/        run outputs, JSON/CSV (gitignored)
analysis/       derived tables and figures (gitignored)
docs/           proposal, reproduce guide, cluster runbook, debate viewer, design notes,
                findings (what the runs showed)
```

## Data and attribution

The dataset is generated with the vendored GraphQA code under
`src/gedebate/graphqa/` (Apache-2.0; see its `NOTICE.md`). Please cite Fatemi,
Halcrow, and Perozzi, *Talk like a Graph: Encoding Graphs for Large Language
Models*, ICLR 2024.
