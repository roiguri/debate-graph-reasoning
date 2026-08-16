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
friendship.

## Setup

```bash
pip install -e .             # dataset build/verify + analysis (CPU only)
pip install -e .[inference]  # + the local model stack, only for `provider = "hf"` runs
```

The reported runs are served by the Together API (`provider = "together"`, needs
`TOGETHER_API_KEY`), which the core install covers. Only a `provider = "hf"` config
loads weights locally and needs the `[inference]` extra plus a CUDA GPU.

## Reproduce

The dataset is a frozen, committed artifact (`data/main.jsonl`, plus `seed11`/`seed13`);
every run loads it rather than regenerating. Verification is deterministic and needs
neither a GPU nor an API key:

```bash
python scripts/build_dataset.py --verify                            # rebuild from the recorded
                                                                    # spec, assert the sha256
```

A run evaluates one condition into a run dir and is resumable and shardable
(`--shard i/n`); analysis reads the run dirs and needs no model:

```bash
python -m gedebate.eval.runner --config configs/llama70b-baseline-main.toml
python -m gedebate.eval.runner --config configs/llama70b-debate-main.toml
python -m gedebate.eval.runner --config configs/llama70b-mvcot-main.toml
python scripts/show_results.py <run_dir> --fragility                 # accuracy + encoding spread
python scripts/debate_diagnostics.py <run_dir>                       # turn split + Critic diagnostic
```

Each run writes `manifest.json`: model, dataset path and `sha256`, decoding,
`max_new_tokens`, git commit. Resuming against a different model or dataset is refused.
Seeds 11 and 13 are the replication; swap the config and pool the run dirs by passing
several to `show_results.py`.

The prompts are stored as small pieces that only meet inside the builders, so the text
sent to the model is assembled, never written down anywhere. `scripts/show_prompts.py`
assembles it from a run config and prints it whole:

```bash
python scripts/show_prompts.py configs/llama70b-debate-main.toml            # every cell
python scripts/show_prompts.py configs/llama70b-debate-main.toml --from-run # with a real transcript
```

`analysis/prompts/` holds committed snapshots of that output -- the one place the
Proposer, Critic and revision wording is readable whole, and what makes a prompt edit
show up as a diff. **Regenerate them in the same commit as any prompt change:**

```bash
python scripts/show_prompts.py configs/llama70b-debate-main.toml   --out analysis/prompts/debate.txt
python scripts/show_prompts.py configs/llama70b-baseline-main.toml --out analysis/prompts/baseline.txt
```

## Layout

```
src/gedebate/   the library: data adapter, model wrapper, prompts, scoring, eval harness
data/           the frozen dataset artifact (+ provenance meta)
configs/        run configs (TOML)
scripts/        entry points (build_dataset, show_results, show_prompts, debate_viewer, smoke)
slurm/          batch-job scripts, specific to our compute setup
results/        run outputs, JSON/CSV (gitignored)
analysis/       derived tables and figures (gitignored)
docs/           proposal, cluster runbook, debate viewer, design notes, the paper,
                findings (what the runs showed)
```

## Data and attribution

The dataset is generated with the vendored GraphQA code under
`src/gedebate/graphqa/` (Apache-2.0; see its `NOTICE.md`). Please cite Fatemi,
Halcrow, and Perozzi, *Talk like a Graph: Encoding Graphs for Large Language
Models*, ICLR 2024.
