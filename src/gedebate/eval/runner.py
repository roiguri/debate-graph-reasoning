"""P2 baseline runner.

    python -m gedebate.eval.runner                 # P2.1: run one instance, print it
    python -m gedebate.eval.runner --out-dir DIR   # P2.2: persist it, skip if already done

Proves the pipe end-to-end -- data -> prompt -> generate -> parse -> score -- and,
with `--out-dir`, the persistence contract (attempt-level JSONL, resume-by-skip,
manifest guard). This grows into the config-driven, shardable matrix runner in P2.3.

The model default is the tiny smoke model: this slice checks the *pipe*, not the
answer, so a small model on CPU is enough. The real experiment model is chosen by
config from P2.3 on.
"""

from __future__ import annotations

import argparse

from gedebate.conditions.baseline import CONDITION, run_instance
from gedebate.data.dataset import build_dataset
from gedebate.eval import results

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_PRINT_KEYS = (
    "raw_output",
    "parsed_answer",
    "parse_ok",
    "correct",
    "ground_truth",
    "n_prompt_tokens",
    "n_gen_tokens",
)


def first_instance(n_graphs: int, seed: int, task: str, encoding: str):
    """First built instance matching (task, encoding). Pure -- no model needed."""
    for inst in build_dataset(n_graphs=n_graphs, seed=seed):
        if inst.task == task and inst.encoding == encoding:
            return inst
    raise RuntimeError(f"no {task} x {encoding} instance in {n_graphs} graphs")


def run_one_persisted(
    model, instance, out_dir: str, model_name: str, *, shard: int = 0, max_new_tokens: int = 64
) -> dict | None:
    """Persist one baseline attempt, skipping if the instance is already done.

    Returns the written row, or None if it was skipped. Torch-free (duck-typed
    model), so it is testable with a stub.
    """
    progress = results.load_progress(out_dir)
    if results.is_instance_done(progress, CONDITION, instance.instance_id):
        return None
    results.ensure_manifest(out_dir, model_name)
    attempt = run_instance(model, instance, max_new_tokens=max_new_tokens)
    row = results.make_row(instance, model_name, attempt)
    results.append_row(results.shard_file(out_dir, CONDITION, shard), row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-graphs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--task", default="edge_existence")
    ap.add_argument("--encoding", default="adjacency")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out-dir", default=None, help="persist + resume here (P2.2); omit to just print")
    args = ap.parse_args()

    inst = first_instance(args.n_graphs, args.seed, args.task, args.encoding)

    # Skip the (expensive) model load when the work is already persisted.
    if args.out_dir is not None:
        progress = results.load_progress(args.out_dir)
        if results.is_instance_done(progress, CONDITION, inst.instance_id):
            print(f"{inst.instance_id} already done in {args.out_dir} -- nothing to do.")
            return

    # Deferred so the module imports without torch (a cluster-only extra).
    from gedebate.model import load_model

    print(f"Loading {args.model} ...", flush=True)
    model = load_model(args.model)
    print(f"Loaded on device={model.device}. Running one instance ...\n", flush=True)

    if args.out_dir is not None:
        row = run_one_persisted(
            model, inst, args.out_dir, args.model, max_new_tokens=args.max_new_tokens
        )
        print(f"wrote row for {inst.instance_id} -> {results.shard_file(args.out_dir, CONDITION)}")
        record = row
    else:
        record = run_instance(model, inst, max_new_tokens=args.max_new_tokens)

    print(f"\nquestion : {inst.question!r}\n")
    for k in _PRINT_KEYS:
        print(f"{k:16}: {record[k]!r}")


if __name__ == "__main__":
    main()
