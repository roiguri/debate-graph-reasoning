"""P2 baseline runner.

    python -m gedebate.eval.runner --config configs/baseline.toml   # P2.3: run the matrix slice
    python -m gedebate.eval.runner --config C --shard 0/4           # one shard of it (P3 fan-out)
    python -m gedebate.eval.runner                                  # P2.1 smoke: run one instance, print

The config path builds the dataset, shards it, skips already-done instances, runs
the baseline on each, appends attempt rows, and prints per-encoding accuracy. It is
resumable (kill + rerun loses <=1 instance) and shardable (the run matrix splits
across jobs). Model + matrix come only from the config; see eval/config.py.
"""

from __future__ import annotations

import argparse

from gedebate.conditions.baseline import CONDITION, run_instance
from gedebate.data.dataset import build_dataset
from gedebate.eval import report, results
from gedebate.eval.config import RunConfig, load_config

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # smoke only; real model comes from config


# --- dataset selection + sharding ---------------------------------------------

def build_instances(cfg: RunConfig) -> list:
    """The full (config-scoped) instance list, in deterministic order."""
    return build_dataset(
        n_graphs=cfg.n_graphs,
        seed=cfg.dataset_seed,
        tasks=cfg.tasks,
        encodings=cfg.encodings,
    )


def select_shard(instances: list, shard: int, n_shards: int) -> list:
    """Instances assigned to `shard`. Round-robin by index: disjoint, ~balanced,
    and their union over all shards is the whole list (order is deterministic)."""
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} out of range for n_shards {n_shards}")
    return [x for i, x in enumerate(instances) if i % n_shards == shard]


def parse_shard(spec: str) -> tuple[int, int]:
    """'i/n' -> (i, n). Default when omitted is a single shard '0/1'."""
    i, n = spec.split("/")
    return int(i), int(n)


# --- the run loop (torch-free: takes a model, so it is stub-testable) ----------

def run_instances(model, instances: list, cfg: RunConfig, *, shard: int = 0) -> dict:
    """Run the baseline over `instances`, persisting and skipping done work.

    Progress is loaded once up front (not per instance), so this is O(n) file I/O.
    Returns {"written", "skipped"}.
    """
    results.ensure_manifest(
        cfg.out_dir,
        cfg.model,
        condition=cfg.condition,
        dataset_seed=cfg.dataset_seed,
        n_graphs=cfg.n_graphs,
    )
    progress = results.load_progress(cfg.out_dir)
    path = results.shard_file(cfg.out_dir, cfg.condition, shard)
    written = skipped = 0
    for inst in instances:
        if results.is_instance_done(progress, cfg.condition, inst.instance_id):
            skipped += 1
            continue
        attempt = run_instance(model, inst, max_new_tokens=cfg.max_new_tokens)
        results.append_row(path, results.make_row(inst, cfg.model, attempt))
        progress.setdefault((cfg.condition, inst.instance_id), set()).add(0)
        written += 1
    return {"written": written, "skipped": skipped}


def summarize_run(cfg: RunConfig) -> dict:
    """Per (task, encoding) accuracy over ALL shards of this run."""
    rows = [r for f in results.result_files(cfg.out_dir) for r in results.read_rows(f)]
    return report.summarize(rows)


# --- single-instance smoke (P2.1) ---------------------------------------------

def first_instance(n_graphs: int, seed: int, task: str, encoding: str):
    for inst in build_dataset(n_graphs=n_graphs, seed=seed):
        if inst.task == task and inst.encoding == encoding:
            return inst
    raise RuntimeError(f"no {task} x {encoding} instance in {n_graphs} graphs")


def run_one_persisted(model, instance, out_dir, model_name, *, shard=0, max_new_tokens=64):
    """Persist one baseline attempt, skipping if already done. Returns row or None."""
    progress = results.load_progress(out_dir)
    if results.is_instance_done(progress, CONDITION, instance.instance_id):
        return None
    results.ensure_manifest(out_dir, model_name)
    attempt = run_instance(model, instance, max_new_tokens=max_new_tokens)
    row = results.make_row(instance, model_name, attempt)
    results.append_row(results.shard_file(out_dir, CONDITION, shard), row)
    return row


# --- CLI ----------------------------------------------------------------------

def _run_config(config_path: str, shard_spec: str) -> None:
    cfg = load_config(config_path)
    shard, n_shards = parse_shard(shard_spec)
    instances = select_shard(build_instances(cfg), shard, n_shards)
    print(
        f"config={config_path} model={cfg.model} shard={shard}/{n_shards} "
        f"instances={len(instances)} -> {cfg.out_dir}",
        flush=True,
    )

    from gedebate.model import load_model  # deferred: torch is a cluster-only extra

    model = load_model(cfg.model)
    print(f"Loaded on device={model.device}. Running ...", flush=True)
    stats = run_instances(model, instances, cfg, shard=shard)
    print(f"done: wrote {stats['written']}, skipped {stats['skipped']}\n", flush=True)
    print(report.format_summary(summarize_run(cfg)), flush=True)


def _run_smoke(args) -> None:
    inst = first_instance(args.n_graphs, args.seed, args.task, args.encoding)
    from gedebate.model import load_model

    print(f"Loading {args.model} ...", flush=True)
    model = load_model(args.model)
    print(f"Loaded on device={model.device}. Running one instance ...\n", flush=True)
    record = run_instance(model, inst, max_new_tokens=args.max_new_tokens)
    print(f"question : {inst.question!r}\n")
    for k in ("raw_output", "parsed_answer", "parse_ok", "correct",
              "ground_truth", "n_prompt_tokens", "n_gen_tokens"):
        print(f"{k:16}: {record[k]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="TOML run config (P2.3 batch path)")
    ap.add_argument("--shard", default="0/1", help="'i/n' shard of the matrix (P3 fan-out)")
    # single-instance smoke knobs (used only when --config is omitted):
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-graphs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--task", default="edge_existence")
    ap.add_argument("--encoding", default="adjacency")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    if args.config is not None:
        _run_config(args.config, args.shard)
    else:
        _run_smoke(args)


if __name__ == "__main__":
    main()
