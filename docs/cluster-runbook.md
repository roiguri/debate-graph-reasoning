# Cluster runbook — TAU CS SLURM (MLWG2026)

How to get code onto the cluster, build the environment, and run jobs. See
[plan/p0-env.md](plan/p0-env.md) for what the env contains and
[plan/overview.md](plan/overview.md) for the plan.

> **Prerequisite: the TAU VPN must be connected** for any `ssh`/`rsync`/`sbatch`
> to the cluster to work from off-campus. Connect it first; every step below
> assumes it's up.

> **Placeholders.** `‹user›` = your TAU CS username, `‹netapp›` =
> `/home/yandex/MLWG2026/‹user›`. Real values are never committed — they live in
> `~/.ssh/config` (login) and the gitignored `.env` (`NETAPP`, `HF_TOKEN`).
> Rediscover the netapp path on the cluster with `echo $HOME` / `ls /home/yandex/MLWG2026`.

## One-time local setup

Add a host block to your **local** `~/.ssh/config` (not in git). The login node
is load-balanced across `c-001..c-010` with rotating host keys, so host-key
verification is disabled for this host only (else every reconnect errors):

```ssh
Host slurm-client
    HostName slurm-client.cs.tau.ac.il
    User ‹user›
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

Optional passwordless login: `ssh-copy-id slurm-client`.

## One-time cluster setup

The env + model weights must live on netapp, NOT `$HOME` (home has a tiny quota).

1. **Push the repo** (from a **local** terminal; excludes secrets/caches/git):
   ```bash
   cd <local repo>
   git rev-parse HEAD > .git_commit          # provenance: see below
   rsync -av --exclude '.git' --exclude '.env' --exclude '.venv' --exclude 'results/*' \
     --exclude '__pycache__' --exclude '*.egg-info' --exclude 'docs/articles' \
     ./ slurm-client:‹netapp›/graph-encodings-with-debate/
   ```
   `.env` is deliberately excluded so a re-sync never clobbers the cluster copy.
   `.venv` is excluded too: the cluster runs jobs through the netapp conda env
   (`slurm/_activate.sh`), never the local virtualenv, so syncing it just wastes
   quota.

   **Write `.git_commit` every time you sync.** `.git` is excluded, so `git rev-parse`
   fails on the cluster and every manifest produced there records
   `"git_commit": "unknown"`. That was tolerable while a run was identified by its
   `prompt_version`, and stopped being so once v2's prompts were edited in place:
   `results/v2-*` and `results/v2b-*` both say `prompt_version: "v2"` and were produced
   by different text. The commit is what tells them apart, and a stale `.git_commit`
   is worse than none — it will claim the wrong one.

2. **Create `.env`** (on the **cluster**, once — holds the netapp path + token):
   ```bash
   ssh slurm-client
   cd ‹netapp›/graph-encodings-with-debate
   cat > .env <<EOF
   NETAPP=‹netapp›
   HF_TOKEN=
   EOF
   ```
   `HF_TOKEN` is only needed for **gated** models (Llama/Gemma — accept the
   license on the model page, then paste a **free** read token). Public models
   (e.g. Qwen2.5) need none. A HF token never costs money; we run inference
   locally on the GPU, not via HF's paid endpoints.

3. **Build the env** (on the **cluster**; installs Miniconda + the inference
   stack onto netapp — a few minutes, CPU-only, safe on the login node):
   ```bash
   bash slurm/setup_env.sh
   ```
   Success ends with: `torch 2.5.1 | transformers 4.49.0 | gedebate 0.0.1`.

## Everyday loop

1. **Edit locally** (local git is the source of truth).
2. **Re-sync** the changed files (repeat the rsync above; `.env` is untouched).
3. **Submit a job** (on the cluster):
   ```bash
   sbatch slurm/smoke.slurm     # de-risk: 1 model, 1 prompt
   squeue --me                  # watch queue/run state
   ```
4. **Read results** (on the cluster):
   ```bash
   cat results/smoke.json       # parsed answer + n_gen_tokens
   cat results/smoke.*.out      # host + GPU line
   cat results/smoke.*.err      # errors, if any
   ```

