"""P2.1 walking-skeleton entry: run the baseline on one instance and print it.

    python -m gedebate.eval.runner [--model <hf-id>]

Proves the whole pipe end-to-end -- data -> prompt -> generate -> parse -> score --
on a single edge_existence x adjacency instance. No persistence yet (that is P2.2);
this grows into the config-driven, shardable, resumable matrix runner in P2.3.

The model default is the tiny smoke model: this slice checks the *pipe*, not the
answer, so a small model on CPU is enough. The real experiment model is chosen by
config from P2.3 on.
"""

from __future__ import annotations

import argparse

from gedebate.conditions.baseline import run_instance
from gedebate.data.dataset import build_dataset

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-graphs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--task", default="edge_existence")
    ap.add_argument("--encoding", default="adjacency")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    inst = first_instance(args.n_graphs, args.seed, args.task, args.encoding)

    # Deferred so the module imports without torch (a cluster-only extra).
    from gedebate.model import load_model

    print(f"Loading {args.model} ...", flush=True)
    model = load_model(args.model)
    print(f"Loaded on device={model.device}. Running one instance ...\n", flush=True)

    record = run_instance(model, inst, max_new_tokens=args.max_new_tokens)

    print(f"question : {inst.question!r}\n")
    for k in _PRINT_KEYS:
        print(f"{k:16}: {record[k]!r}")


if __name__ == "__main__":
    main()
