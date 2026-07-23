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
in the script, and run `--init`.

To add data for a replication check, build an **independent seed as a sibling
artifact** (`--name`/`--seed`) rather than growing `main`: this keeps `main.jsonl`
byte-identical (its hash, and every existing run's manifest guard, unaffected).
`instance_id`s are namespaced by `dataset_seed`, so the two never collide when
pooled. Everything but the seed is held fixed, so the samples are matched:
```bash
python scripts/build_dataset.py --init --name seed11 --seed 11   # data/seed11.jsonl
python scripts/build_dataset.py --verify --name seed11           # check it reproduces
```
The committed replication seeds are `seed11` and `seed13`.

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
Writes per-cell accuracy (with 95% Wilson CIs), the per-task fragility table, and
per-task significance to `analysis/baseline/`. Because the encodings are applied to
the *same* graphs, significance is paired: Cochran's Q (omnibus per task) and a
best-vs-worst McNemar test say whether the fragility spread is real, not sampling
noise.

## 4. Replication: does the pattern hold on independent graphs?
Run the baseline on each replication seed (step 2, with its config), then pool the
run dirs. Passing more than one dir pools them; `--by-seed` also prints each seed's
own table so you can see the same best/worst ordering repeat on fresh graphs:
```bash
python -m gedebate.eval.runner --config configs/seed11.toml   # -> results/seed11/ (GPU)
python -m gedebate.eval.runner --config configs/seed13.toml   # -> results/seed13/ (GPU)
python scripts/show_results.py results/main results/seed11 results/seed13 \
    --fragility --by-seed --save analysis/pooled
```
