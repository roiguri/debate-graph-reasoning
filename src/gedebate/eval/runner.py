"""Baseline runner over the frozen dataset artifact.

    python -m gedebate.eval.runner --config configs/matrix.toml           # run a condition
    python -m gedebate.eval.runner --config C --shard 0/8                     # one shard of it
    python -m gedebate.eval.runner --config C --verify-sample 20              # reproducibility spot check

A run LOADS `cfg.dataset` (never regenerates), filters to `cfg.tasks`/`encodings`,
shards, skips already-done instances, runs the condition, and appends attempt rows.
The run manifest doubles as a reproduction record (dataset + sha256, decoding, git
commit). Resumable + shardable; the union of shard files is the full run.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import gedebate
from gedebate.conditions.baseline import run_instance
from gedebate.conditions.debate import run_debate
from gedebate.conditions.majority_vote import CONDITION_COT, run_sample
from gedebate.data.dataset import build_dataset
from gedebate.data.store import dataset_identity, load_dataset
from gedebate.eval import report, results
from gedebate.eval.config import KNOWN_PROVIDERS, RunConfig, load_config

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # smoke only; real model comes from config


# --- dataset load + filter + shard --------------------------------------------

def build_instances(cfg: RunConfig) -> list:
    """Load the frozen dataset and filter to the config's tasks/encodings."""
    tasks, encodings = set(cfg.tasks), set(cfg.encodings)
    return [
        i for i in load_dataset(cfg.dataset)
        if i.task in tasks and i.encoding in encodings
    ]


def select_shard(instances: list, shard: int, n_shards: int) -> list:
    """Round-robin by index: disjoint, ~balanced, union over shards is the whole list."""
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} out of range for n_shards {n_shards}")
    return [x for i, x in enumerate(instances) if i % n_shards == shard]


def parse_shard(spec: str) -> tuple[int, int]:
    i, n = spec.split("/")
    return int(i), int(n)


# --- reproduction record ------------------------------------------------------

# Written by the sync step, read when `git` is unavailable. Module-level so tests can
# point it elsewhere.
_COMMIT_STAMP = Path(__file__).resolve().parents[3] / ".git_commit"


def _git_commit() -> str:
    """The commit these rows were produced at, or "unknown".

    Falls back to a `.git_commit` file because runs happen on the cluster, where the
    tree is an rsync copy with no `.git` -- which is why every existing manifest records
    "unknown". That was survivable while a run was identified by `prompt_version` alone,
    and stopped being so once v2's text was edited in place: two runs can now both say
    `prompt_version: "v2"` and mean different prompts, and the commit is what separates
    them. Write the file as part of the sync (see docs/cluster-runbook.md).
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    try:
        return _COMMIT_STAMP.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


def manifest_record(cfg: RunConfig, config_path: str) -> dict:
    """The reproduction record stored in the run manifest (guards on dataset_sha256)."""
    decoding = (f"temperature={cfg.temperature},top_p={cfg.top_p},top_k={cfg.top_k}"
                if cfg.condition in results.VOTE_CONDITIONS else "greedy")  # baseline + debate greedy
    rec = {
        "dataset": cfg.dataset,
        "dataset_sha256": dataset_identity(cfg.dataset),
        "provider": cfg.provider,  # which backend served `model` (see eval/config.py)
        "decoding": decoding,
        "max_new_tokens": cfg.max_new_tokens,
        "gedebate_version": gedebate.__version__,
        "git_commit": _git_commit(),
        "config": config_path,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    if cfg.condition in results.VOTE_CONDITIONS:
        rec["n_samples"] = cfg.n_samples
    if cfg.condition == CONDITION_COT:
        # The reasoned arm samples the Proposer prompt, so its rows depend on that wording
        # exactly as debate's do -- record it or two arms run months apart look identical.
        rec["prompt_version"] = cfg.prompt_version
    if cfg.condition == "debate":
        rec["max_responses"] = cfg.n_samples  # debate's response budget (= MV's N)
        rec["prompt_version"] = cfg.prompt_version  # which Proposer wording produced these rows
    return rec


# --- the run loop (torch-free: takes a model, so it is stub-testable) ----------

def run_instances(model, instances: list, cfg: RunConfig, manifest: dict, *, shard: int = 0) -> dict:
    """Run the condition over `instances`, persisting and skipping done work.

    `manifest` is the reproduction record (must include `dataset_sha256`), written
    once and guarded on resume. `written` counts rows appended, `skipped` counts
    already-complete instances. Baseline writes one row per instance; majority-vote
    writes `cfg.n_samples` (topping up only the missing sample indexes on resume).
    """
    results.ensure_manifest(cfg.out_dir, cfg.model, cfg.condition, **manifest)
    progress = results.load_progress(cfg.out_dir)
    path = results.shard_file(cfg.out_dir, cfg.condition, shard)
    trace_path = results.trace_file(cfg.out_dir, cfg.condition, shard)
    written = skipped = 0
    for inst in instances:
        if cfg.condition in results.VOTE_CONDITIONS:
            n = _run_vote_samples(model, inst, cfg, path, progress)
            written += n
            skipped += n == 0  # instance already had all N samples
        elif results.is_instance_done(progress, cfg.condition, inst.instance_id):
            skipped += 1
        elif cfg.condition == "debate":
            record, turns = run_debate(
                model, inst, max_new_tokens=cfg.max_new_tokens, max_responses=cfg.n_samples,
                prompt_version=cfg.prompt_version)
            results.append_row(path, results.make_row(
                inst, cfg.model, record, n_responses=record["n_responses"],
                prompt_version=cfg.prompt_version))
            results.append_trace(trace_path, inst.instance_id, turns)
            progress.setdefault((cfg.condition, inst.instance_id), set()).add(0)
            written += 1
        else:  # baseline
            attempt = run_instance(model, inst, max_new_tokens=cfg.max_new_tokens)
            results.append_row(path, results.make_row(inst, cfg.model, attempt))
            progress.setdefault((cfg.condition, inst.instance_id), set()).add(0)
            written += 1
    return {"written": written, "skipped": skipped}


def _run_vote_samples(model, inst, cfg: RunConfig, path, progress) -> int:
    """Persist the missing sample rows for one majority-vote instance; return the count.

    Resumes via `results.missing_samples`, so a killed shard tops up only the draws
    it did not finish. Returns 0 when the instance was already complete.
    """
    missing = results.missing_samples(progress, cfg.condition, inst.instance_id, cfg.n_samples)
    seen = progress.setdefault((cfg.condition, inst.instance_id), set())
    # None for the terse arm (baseline prompt); the config's version for the reasoned one,
    # which is what selects the Proposer prompt inside `run_sample`.
    prompt_version = cfg.prompt_version if cfg.condition == CONDITION_COT else None
    for si in missing:
        attempt = run_sample(
            model, inst, sample_index=si,
            temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
            max_new_tokens=cfg.max_new_tokens, prompt_version=prompt_version,
        )
        row = results.make_row(
            inst, cfg.model, attempt,
            sample_index=si, temperature=cfg.temperature, seed=attempt["seed"],
            prompt_version=prompt_version,
        )
        results.append_row(path, row)
        seen.add(si)
    return len(missing)


def summarize_run(cfg: RunConfig) -> dict:
    """Per-cell summary for the config's condition only (a shared out_dir holds
    baseline/ and majority_vote/ side by side; never pool them into one accuracy)."""
    rows = [
        r for f in results.result_files(cfg.out_dir) for r in results.read_rows(f)
        if r["condition"] == cfg.condition
    ]
    if cfg.condition in results.VOTE_CONDITIONS:
        return report.summarize_votes(rows)
    return report.summarize(rows)


# --- reproducibility spot check (re-run a sample, compare parsed answers) ------

def verify_sample(model, instances_by_id: dict, rows: list, k: int, *, max_new_tokens: int = 64) -> dict:
    """Re-run an even-spaced sample of K persisted rows and check the parsed answer
    still matches. Spacing spreads the sample across shards/cells rather than
    concentrating on the first rows.

    Greedy fp16 on GPU is not guaranteed byte-identical across hardware, so this
    compares parsed answers, not raw text.
    """
    step = max(1, len(rows) // k) if k else 1
    sample = rows[::step][:k]
    checked = matches = 0
    mismatches = []
    for row in sample:
        inst = instances_by_id.get(row["instance_id"])
        if inst is None:
            continue
        rec = run_instance(model, inst, max_new_tokens=max_new_tokens)
        checked += 1
        if rec["parsed_answer"] == row["parsed_answer"]:
            matches += 1
        else:
            mismatches.append((row["instance_id"], row["parsed_answer"], rec["parsed_answer"]))
    return {"checked": checked, "matches": matches, "mismatches": mismatches}


# --- single-instance smoke (regenerates one instance; dev convenience) --------

def first_instance(n_graphs: int, seed: int, task: str, encoding: str):
    for inst in build_dataset(n_graphs=n_graphs, seed=seed):
        if inst.task == task and inst.encoding == encoding:
            return inst
    raise RuntimeError(f"no {task} x {encoding} instance in {n_graphs} graphs")


# --- CLI ----------------------------------------------------------------------

def load_provider_model(provider: str, name: str):
    """Load `name` from `provider`.

    Both imports stay deferred: torch is a cluster-only extra, and the API path must
    remain importable on a machine that has never installed it.
    """
    if provider == "together":
        from gedebate.together import load_model

        return load_model(name)
    from gedebate.model import load_model

    return load_model(name)


def _run_config(config_path: str, shard_spec: str) -> None:
    cfg = load_config(config_path)
    shard, n_shards = parse_shard(shard_spec)
    instances = select_shard(build_instances(cfg), shard, n_shards)
    manifest = manifest_record(cfg, config_path)
    print(
        f"config={config_path} model={cfg.model} provider={cfg.provider} "
        f"dataset={cfg.dataset} shard={shard}/{n_shards} "
        f"instances={len(instances)} -> {cfg.out_dir}",
        flush=True,
    )

    model = load_provider_model(cfg.provider, cfg.model)
    print(f"Loaded on device={model.device}. Running ...", flush=True)
    stats = run_instances(model, instances, cfg, manifest, shard=shard)
    print(f"done: wrote {stats['written']}, skipped {stats['skipped']}\n", flush=True)
    fmt = (report.format_vote_summary if cfg.condition in results.VOTE_CONDITIONS
           else report.format_summary)
    print(fmt(summarize_run(cfg)), flush=True)


def _verify_sample(config_path: str, k: int) -> None:
    cfg = load_config(config_path)
    if cfg.condition != "baseline":
        # verify_sample re-runs greedily; that only reproduces baseline's rows, not
        # majority-vote's sampled draws.
        raise SystemExit("--verify-sample supports the baseline condition only")
    by_id = {i.instance_id: i for i in build_instances(cfg)}
    rows = [r for f in results.result_files(cfg.out_dir) for r in results.read_rows(f)]
    if not rows:
        raise SystemExit(f"no persisted rows in {cfg.out_dir} to verify")

    model = load_provider_model(cfg.provider, cfg.model)
    res = verify_sample(model, by_id, rows, k, max_new_tokens=cfg.max_new_tokens)
    print(f"verify-sample: {res['matches']}/{res['checked']} parsed answers reproduce")
    for iid, was, now in res["mismatches"]:
        print(f"  MISMATCH {iid}: stored={was!r} rerun={now!r}")
    if res["mismatches"]:
        raise SystemExit("verify-sample found mismatches")


def _run_smoke(args) -> None:
    inst = first_instance(args.n_graphs, args.seed, args.task, args.encoding)
    print(f"Loading {args.model} (provider={args.provider}) ...", flush=True)
    model = load_provider_model(args.provider, args.model)
    print(f"Loaded on device={model.device}. Running one instance ...\n", flush=True)
    record = run_instance(model, inst, max_new_tokens=args.max_new_tokens)
    print(f"question : {inst.question!r}\n")
    for key in ("raw_output", "parsed_answer", "parse_ok", "correct",
                "ground_truth", "n_prompt_tokens", "n_gen_tokens"):
        print(f"{key:16}: {record[key]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="TOML run config (the batch path)")
    ap.add_argument("--shard", default="0/1", help="'i/n' shard of the matrix")
    ap.add_argument("--verify-sample", type=int, default=None, metavar="K",
                    help="re-run K persisted instances and check parsed answers match")
    # single-instance smoke knobs (used only when --config is omitted):
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--provider", default="hf", choices=KNOWN_PROVIDERS,
                    help="serving backend for --model (smoke path only; a config "
                         "carries its own `provider`)")
    ap.add_argument("--n-graphs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--task", default="edge_existence")
    ap.add_argument("--encoding", default="adjacency")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    if args.config is not None and args.verify_sample is not None:
        _verify_sample(args.config, args.verify_sample)
    elif args.config is not None:
        _run_config(args.config, args.shard)
    else:
        _run_smoke(args)


if __name__ == "__main__":
    main()
