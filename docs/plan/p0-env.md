# P0 — Cluster env + minimal repo

Goal: prove a model loads and generates on the cluster, with the smallest
possible footprint. Mostly adapting the sibling `nlp` repo's proven scripts to
this course's netapp path. See [overview.md](overview.md) for where this fits.

**Done when:** `sbatch slurm/smoke.slurm` loads a small instruct model,
generates on one prompt, and writes `results/smoke.json` with the output text
and a generated-token count.

## Substeps

### P0.1 — Python packaging
- `pyproject.toml` defining package `gedebate` under `src/gedebate/`.
- Pinned deps: `torch`, `transformers`, `networkx`, `numpy`, `tqdm`, `pytest`.
  Start from the `nlp` repo's pins (`torch==2.5.1`, `transformers==4.49.0`) since
  they're already validated on these exact GPU nodes; adjust only if a chosen
  model needs newer.
- Editable install: `pip install -e .`
- Decision: confirm package name `gedebate` (else pick one now — it's imported
  everywhere).

### P0.2 — Slurm scaffolding (adapt from `nlp/slurm/`)
- `slurm/_activate.sh` — sources `.env` for secrets, sets `HF_HOME` on netapp,
  activates the conda env. Driven by a `NETAPP` variable.
- `slurm/setup_env.sh` — one-time env creation on netapp (conda env + deps).
- `slurm/smoke.slurm` — the de-risk job (see Done-when above).
- **Host-identifier hygiene:** the real netapp path contains a course id and
  username. It must NOT be hardcoded in git. Options: default `NETAPP` to a
  placeholder and require an override, or read it from the gitignored `.env`.
  `.env` holds only `HF_TOKEN` + `NETAPP`; commit `.env.example` with
  placeholders.
- Confirm the course's netapp base path (the `nlp` repo used a different
  course's path — this project has its own allocation).

### P0.3 — Minimal model wrapper
- `src/gedebate/model.py`: `load_model(name) -> Model` and
  `Model.generate(prompt, **kw) -> GenResult` where `GenResult` carries the
  decoded text **and** `n_gen_tokens` (output length from the tokenizer).
- Deliberately minimal — no batching, no chat templating beyond what smoke
  needs. Grows in P2. Token counting is the one thing to get right here, since
  matched-compute depends on it.

### P0.4 — Smoke job entrypoint
- `src/gedebate/smoke.py` (run as `python -m gedebate.smoke --out results/smoke.json`):
  load model, generate on a hand-written prompt, write JSON with
  `{model, prompt, output, n_gen_tokens, host, gpu}`.
- Prints host + `nvidia-smi` line like the `nlp` smoke job, for cluster sanity.

### P0.5 — Repo hygiene
- `.gitignore`: `results/`, `.env`, HF cache, `__pycache__/`, `*.egg-info/`.
- Short `README.md` stub: one-time setup + how to run smoke.

## Cluster facts (confirmed)
- **Netapp base:** `/home/yandex/MLWG2026/‹user›` (course = Machine Learning
  with Graphs 2026). The `‹user›` segment is a username — it stays OUT of git.
  `NETAPP` is read from the gitignored `.env`; `.env.example` ships a placeholder.
  Rediscover the real value on the cluster with `echo $HOME` / `ls /home/yandex/MLWG2026`.
- **Resource limits:** each student has limited memory and a limited number of
  concurrent jobs. Implications carried forward:
  - Keep per-job memory modest (small models, no oversized batches).
  - The eval harness (P2+) shards the task×encoding×condition matrix into
    several right-sized jobs rather than one monolith, and is resumable — so the
    job quota is used fully and a killed job loses little work.

## Open decisions
- **Which model for smoke?** A tiny instruct model that fits 11GB comfortably
  (e.g. `Qwen2.5-1.5B-Instruct` or similar) to keep the de-risk fast. The full
  experiment model (7-8B) is chosen later in P2/P3.
- Confirm the partition/constraint (the `nlp` repo pinned `studentkillable` +
  `geforce_rtx_2080`; verify the same applies for this course's allocation).
