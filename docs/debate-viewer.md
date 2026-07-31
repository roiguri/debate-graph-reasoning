# Debate Reader (`scripts/debate_viewer.py`)

A local web viewer for reading Proposer-Critic debate transcripts against the graph
each one reasoned about. Its job is to answer "is the Critic useful?" by letting you
browse many debates and read them turn by turn. Transcripts load on demand, so it
stays responsive over a full run (up to ~1800 debates) without holding them all at once.

## Launch

```bash
python scripts/debate_viewer.py <run_dir> [--port 8000]
# examples
python scripts/debate_viewer.py results/debate-pilot        # -> http://localhost:8000
python scripts/debate_viewer.py results/main --port 8080
```

Requirements:
- The repo installed (`pip install -e .`), which brings in networkx. No torch and no
  GPU: the viewer only reads results, it never runs the model.
- Internet on first load: the graph is drawn with Cytoscape, fetched from a CDN
  (`unpkg`). Everything else is served locally and self-contained.

Data is snapshotted at startup. If a run is still being written (a job in progress, or
results freshly pulled), restart the viewer to pick up the new debates.

## What you see

- **Left rail:** every debate as a row (status dot + id), filterable by outcome / task
  / encoding, with id search. `/` focuses search; `j`/`k` (or `n`/`p`) walk the list.
- **Center:** a header (id split into seed / graph / task / encoding, the question,
  correct/wrong, and Answer / Truth / Turns / Tokens tiles) over the turn-by-turn
  transcript (Proposer left, Critic right, verdict pills, per-turn tokens).
- **Right:** the graph (query node lit, its edges bolded, disconnected clusters tiled
  so each stays legible), then two panels collapsed by default: a per-turn token cost
  chart (`c` toggles it) and the raw encoding the model read (`r` toggles it). Nodes
  carry the names that encoding used, so the drawing and
  the transcript speak the same language: integers for `adjacency` / `incident`, people
  for `friendship`. Hover a node for its integer id.

The top bar has light/dark and rail-collapse toggles.

## What a run dir must contain (the contract)

The viewer is decoupled from how results are produced. It needs a run dir with:
- `<condition>/*.jsonl` result rows, including rows with `condition == "debate"`;
- `*.trace.jsonl` sidecar files (the per-turn transcripts), anywhere under the run dir;
- `manifest.json` with a `dataset` key pointing at the dataset the run used (e.g.
  `data/main.jsonl`), so the viewer can draw each graph.

`results/debate-pilot/` is a working example to point at.

## Populating results (current pipeline, likely to change)

> The viewer only depends on the contract above. However you produce a run dir that
> satisfies it, the viewer renders it. The steps below are how results are produced
> today; expect this to evolve.

A run evaluates one condition over the dataset and writes the rows plus traces:

```bash
python -m gedebate.eval.runner --config configs/debate-pilot.toml   # -> results/debate-pilot/ (GPU)
```

- The config names the model, condition (`debate`), dataset, the response budget
  (`n_samples`), and `out_dir`. See `configs/debate-pilot.toml` for the single-cell
  pilot; the full 3x3 run uses its own config.
- Running the `debate` condition needs the model stack and a GPU
  (`pip install -e .[inference]`). The debate loop writes one summed row per instance
  plus a trace sidecar; it is resumable.
- On our TAU SLURM setup the loop is: develop locally, sync to the cluster, run there,
  pull results back for local viewing. See [cluster-runbook.md](cluster-runbook.md).
  Then point the viewer at the pulled run dir.

For the broader dataset -> run -> analysis pipeline (not viewer specific), see
[reproduce.md](reproduce.md).
