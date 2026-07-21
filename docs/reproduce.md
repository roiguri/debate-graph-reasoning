# Reproducing the experiment

Reproduction is environment-agnostic: everything runs through the package's own
entry points on any machine. Two artifacts reproduce independently: the dataset
(deterministic, CPU only) and the results (per condition, needs a GPU). For our TAU
SLURM dispatch specifically (sync, sbatch, netapp paths) see
[cluster-runbook.md](cluster-runbook.md); none of it is required to reproduce.

## 0. Install
```bash
pip install -e .            # data build/verify + analysis (CPU only, no torch)
pip install -e .[inference] # also the model stack, needed to run the conditions
```
Running a condition needs a CUDA GPU that fits the model (Qwen2.5-3B is about 6 GB
fp16); CPU works but is slow. Dataset build/verify and analysis need neither.

## 1. Dataset: reproduce and verify (available now)
The dataset is a frozen, committed artifact: `data/main.jsonl` (1800 instances).
Its source of truth is `data/main.meta.json`, which records the build `spec`, the
`sha256`, and provenance (gedebate version, git commit). Reproduction is driven by
that meta file, not by any values hardcoded in a script or this doc:
```bash
python scripts/build_dataset.py --verify   # rebuild from meta['spec'], assert the hash matches, write nothing
python scripts/build_dataset.py            # regenerate data/main.jsonl from meta['spec']; idempotent, refuses to write on any drift
```
`--verify` is the read-only reproducibility check (deterministic, no GPU); it prints
the recorded spec so you never have to know the numbers in advance. `build` can only
reproduce the recorded dataset, never silently change it. Creating or intentionally
changing the dataset is a deliberate act: remove the artifact, edit `BOOTSTRAP_SPEC`
in the script, and run `--init`. To grow the set, append a new seed's instances;
existing `instance_id`s never move.

## 2. Results: reproduce
A run loads `data/main.jsonl` (never regenerates), evaluates one condition, and
writes per-instance rows. This one command is the whole reproducible unit, anywhere:
```bash
python -m gedebate.eval.runner --config configs/matrix.toml
# split across machines/GPUs with --shard i/n (e.g. --shard 0/8); resumable, so
# re-running skips already-done instances. Writes results/main/baseline/.
```
The run writes `results/main/manifest.json`, the reproduction record: model, dataset
path plus `sha256`, decoding, `max_new_tokens`, git commit, and gedebate version. It
is the source of truth for what produced a `results/main/` dir, and a resume against
a different model or dataset is refused.

Verify a run reproduces (re-run a sample, check parsed answers match the stored rows):
```bash
python -m gedebate.eval.runner --config configs/matrix.toml --verify-sample 20
```
Greedy fp16 on GPU is not byte-identical across hardware, so this checks parsed
answers, not raw text.

## 3. Analysis: reproduce (available now)
No GPU needed; point it at wherever the run wrote its results:
```bash
python scripts/show_results.py results/main --fragility --save analysis/baseline
```
Writes per-cell accuracy and the per-task fragility table to `analysis/baseline/`.
