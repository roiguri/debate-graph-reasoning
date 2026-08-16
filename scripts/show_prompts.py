"""Print the FULL prompts a run config sends, assembled exactly as the run assembles them.

The prompt text lives in `gedebate.prompts` as small pieces (preambles, format blocks,
per-task claim kinds and Critic cues) that only ever meet inside the builders. This
script calls those same builders, so what it prints is what the model is sent -- there is
no second copy of the wording here to drift, and re-running it after a prompt edit is how
you read the new prompt.

    python scripts/show_prompts.py configs/llama70b-debate-main.toml
    python scripts/show_prompts.py configs/llama70b-debate-main.toml --task node_degree
    python scripts/show_prompts.py configs/llama70b-baseline-main.toml --out prompts.txt
    python scripts/show_prompts.py C --instance-id 7/0/connected_nodes/friendship
    python scripts/show_prompts.py C --from-run          # real transcript, not placeholders
    python scripts/show_prompts.py C --rounds 2          # a longer placeholder transcript

Everything comes from the config: condition, the tasks x encodings matrix, and the
frozen dataset the questions are taken from. One instance per cell (the
first in the dataset, so the choice is deterministic); the graph differs per instance but
the scaffold around it does not.

Debate prompts grow with the transcript, so a Critic/revision prompt only exists relative
to prior turns. By default the transcript is filled with clearly marked PLACEHOLDER turns,
which keeps the output a pure function of the config. `--from-run` instead replays a real
transcript from a run's trace sidecar and reconstructs the prompt each turn was actually
generated from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gedebate.data.store import load_dataset
from gedebate.eval import results
from gedebate.eval.config import RunConfig, load_config
from gedebate.eval.results import VOTE_CONDITIONS
from gedebate.prompts import build_prompt
from gedebate.prompts.debate import critic_prompt, proposer_prompt, revision_prompt

# Stand-ins for model output in the transcript. Deliberately loud: a reader must never
# mistake scaffold for something the model wrote, and `render_transcript` splices these
# in exactly where a real turn would go (the turn-1 one completes the question's "A: ").
PLACEHOLDER_PROPOSER = (
    "<<< PLACEHOLDER: the Proposer's turn-1 output (numbered claims + an ANSWER: line) >>>"
)
PLACEHOLDER_CRITIC = (
    "<<< PLACEHOLDER: the Critic's output (a VERDICT: line, plus problems if REVISE) >>>"
)
PLACEHOLDER_REVISION = (
    "<<< PLACEHOLDER: the Proposer's revised output (numbered claims + an ANSWER: line) >>>"
)


# --- instance selection -------------------------------------------------------

def select_instances(
    cfg: RunConfig, *, task: str | None = None, encoding: str | None = None,
    instance_id: str | None = None,
) -> list:
    """One instance per (task, encoding) cell of the config's matrix.

    The first match in dataset order, so two runs of this script agree. `--task` /
    `--encoding` narrow the matrix; `--instance-id` pins one exact instance instead.
    """
    instances = load_dataset(cfg.dataset)
    if instance_id is not None:
        chosen = [i for i in instances if i.instance_id == instance_id]
        if not chosen:
            raise SystemExit(f"no instance {instance_id!r} in {cfg.dataset}")
        return chosen

    tasks = [t for t in cfg.tasks if task in (None, t)]
    encodings = [e for e in cfg.encodings if encoding in (None, e)]
    if not tasks or not encodings:
        raise SystemExit(
            f"filter selects nothing: config covers tasks={list(cfg.tasks)} "
            f"encodings={list(cfg.encodings)}"
        )
    chosen = []
    for t in tasks:
        for e in encodings:
            match = next((i for i in instances if i.task == t and i.encoding == e), None)
            if match is None:
                raise SystemExit(f"no {t}/{e} instance in {cfg.dataset}")
            chosen.append(match)
    return chosen


# --- transcripts --------------------------------------------------------------

def placeholder_turns(rounds: int) -> list[dict]:
    """A synthetic transcript: turn-1 Proposer, then `rounds` x (Critic, revision).

    Enough turns that every prompt shape appears -- one round already yields the Proposer,
    Critic and revision prompts, and a second shows how the transcript (and so the prompt)
    grows with each exchange.
    """
    turns = [{"role": "proposer", "raw": PLACEHOLDER_PROPOSER}]
    for _ in range(rounds):
        turns.append({"role": "critic", "raw": PLACEHOLDER_CRITIC})
        turns.append({"role": "proposer", "raw": PLACEHOLDER_REVISION})
    return turns


def load_trace_turns(run_dir: str | Path, instance_id: str) -> list[dict] | None:
    """The recorded transcript for `instance_id`, or None if the run has no trace for it."""
    for path in results.trace_files(run_dir):
        for rec in results.read_traces(path):
            if rec.get("instance_id") == instance_id:
                return rec["turns"]
    return None


def debate_prompts(instance, turns: list[dict]) -> list[tuple[str, str]]:
    """(label, prompt) for every turn in `turns`, rebuilt the way `run_debate` builds them.

    Each turn is generated from the transcript BEFORE it, so turn i's prompt is a function
    of `turns[:i]`: turn 1 is the Proposer prompt, a Critic turn is the Critic prompt, and
    a later Proposer turn is the revision prompt.
    """
    prompts = []
    for i, turn in enumerate(turns):
        if i == 0:
            label, prompt = "PROPOSER (turn 1)", proposer_prompt(instance)
        elif turn["role"] == "critic":
            label = f"CRITIC (turn {i + 1})"
            prompt = critic_prompt(instance, turns[:i])
        else:
            label = f"PROPOSER REVISION (turn {i + 1})"
            prompt = revision_prompt(instance, turns[:i])
        prompts.append((label, prompt))
    return prompts


# --- rendering ----------------------------------------------------------------

_RULE = "=" * 88


def _block(label: str, prompt: str) -> str:
    """One prompt under a banner, verbatim, with its size (the token cap's raw material)."""
    size = f"{len(prompt):,} chars, {len(prompt.splitlines()):,} lines"
    return f"{_RULE}\n {label}  [{size}]\n{_RULE}\n{prompt}\n"


def _header(cfg: RunConfig, config_path: str, source: str) -> str:
    lines = [
        _RULE,
        f" PROMPTS FOR {config_path}",
        _RULE,
        f"model            {cfg.model}   (provider: {cfg.provider})",
        f"condition        {cfg.condition}",
        f"dataset          {cfg.dataset}",
        f"matrix           tasks={list(cfg.tasks)}  encodings={list(cfg.encodings)}",
        f"max_new_tokens   {cfg.max_new_tokens}",
    ]
    if cfg.condition == "debate":
        lines += [
            f"max_responses    {cfg.n_samples}   (the response budget)",
            f"transcript       {source}",
        ]
    elif cfg.condition in VOTE_CONDITIONS:
        lines.append(f"n_samples        {cfg.n_samples}   (draws per instance, then voted)")
    lines += [
        "",
        "Each prompt below is sent as a SINGLE user message with no system prompt; the",
        "model's chat template wraps it (locally under provider 'hf', server-side under",
        "'together'), so the wrapper tokens are not shown here.",
        "",
    ]
    return "\n".join(lines)


def render(
    cfg: RunConfig, config_path: str, instances: list, *, rounds: int = 1,
    from_run: str | None = None, roles: tuple[str, ...] = ("proposer", "critic", "revision"),
) -> str:
    """The full report: header, then every selected cell's prompts."""
    source = f"real, replayed from {from_run}" if from_run else "placeholder turns"
    out = [_header(cfg, config_path, source)]
    for instance in instances:
        out.append(f"\n\n######## {instance.task} / {instance.encoding} "
                   f"({instance.instance_id}) ########\n")
        if cfg.condition == "debate":
            turns = None
            if from_run:
                turns = load_trace_turns(from_run, instance.instance_id)
                if turns is None:
                    out.append(f"[no trace for {instance.instance_id} under {from_run}; "
                               f"falling back to placeholder turns]\n")
            blocks = debate_prompts(instance, turns or placeholder_turns(rounds))
            blocks = [(lbl, p) for lbl, p in blocks if _role_of(lbl) in roles]
        elif cfg.condition == "majority_vote_cot":
            # The reasoned vote arm samples the debate's turn-1 Proposer prompt N times.
            blocks = [(f"PROPOSER PROMPT (x{cfg.n_samples} sampled draws, then voted)",
                       proposer_prompt(instance))]
        else:
            # baseline and the terse vote arm send the same single prompt; MV re-samples it.
            blocks = [(f"{cfg.condition.upper()} PROMPT", build_prompt(instance))]
        out.extend(_block(lbl, p) for lbl, p in blocks)
    return "\n".join(out)


def _role_of(label: str) -> str:
    if label.startswith("CRITIC"):
        return "critic"
    return "revision" if "REVISION" in label else "proposer"


# --- cli ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="run config TOML (configs/*.toml)")
    ap.add_argument("--task", help="only this task (default: every task in the config)")
    ap.add_argument("--encoding", help="only this encoding (default: all in the config)")
    ap.add_argument("--instance-id", help="pin one exact instance instead of one per cell")
    ap.add_argument("--rounds", type=int, default=1,
                    help="placeholder Critic/revision exchanges to show (default 1)")
    ap.add_argument("--from-run", nargs="?", const="", metavar="RUN_DIR",
                    help="replay a real transcript from a run's traces "
                         "(bare flag = the config's out_dir)")
    ap.add_argument("--roles", default="proposer,critic,revision",
                    help="debate prompts to print (default all three)")
    ap.add_argument("--out", help="write to this file instead of stdout")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    from_run = cfg.out_dir if args.from_run == "" else args.from_run
    roles = tuple(r.strip() for r in args.roles.split(",") if r.strip())

    instances = select_instances(cfg, task=args.task, encoding=args.encoding,
                                 instance_id=args.instance_id)
    text = render(cfg, args.config, instances, rounds=args.rounds, from_run=from_run,
                  roles=roles)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(text):,} chars)")
    else:
        print(text)


if __name__ == "__main__":
    main()
