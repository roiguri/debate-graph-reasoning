"""P0 de-risk entrypoint: load a small model, generate once, write JSON.

Run on the cluster via slurm/smoke.slurm, or directly:
    python scripts/smoke.py --out results/smoke.json

Confirms the env, GPU, model load, generation, and token accounting all work
end-to-end before any real experiment code is built.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
from pathlib import Path

from gedebate.model import load_model

# A tiny instruct model that fits the 11GB RTX 2080 comfortably. The full
# experiment model (larger) is chosen later in P2/P3.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# A hand-written graph question -- just enough to exercise a real prompt.
SMOKE_PROMPT = (
    "G describes a graph among nodes 0, 1, 2, 3. The edges in G are: "
    "(0, 1) (1, 2) (2, 3). What is the degree of node 2? "
    "Answer with a single integer."
)


def gpu_info() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/smoke.json")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    model = load_model(args.model)
    print(f"Loaded on device={model.device}. Generating ...", flush=True)

    result = model.generate(SMOKE_PROMPT, max_new_tokens=args.max_new_tokens)

    record = {
        "model": args.model,
        "host": socket.gethostname(),
        "gpu": gpu_info(),
        "python": platform.python_version(),
        "prompt": SMOKE_PROMPT,
        "output": result.text,
        "n_prompt_tokens": result.n_prompt_tokens,
        "n_gen_tokens": result.n_gen_tokens,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))

    print(json.dumps(record, indent=2), flush=True)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
