"""Debate condition: Proposer-Critic loop with a full running transcript.

The Proposer answers with a numbered-claim trace + final answer; the Critic verifies the
latest answer against the encoding and returns AGREE / REVISE; the Proposer revises. The
loop runs to **consensus** (Critic AGREE, or an unparseable verdict defaulted to AGREE
but counted), **no progress** (the Proposer repeats a previous answer), or the **response
budget** (MV's N, so debate can never out-spend majority vote). Final answer = the last
Proposer answer. Every turn is one full model response; compute is measured in # responses
(= turns) + total tokens (prompt + generated).

Torch-free like the other conditions: `run_debate` only needs a duck-typed model with
`.generate(prompt, max_new_tokens=...) -> GenResult`, so it is stub-testable off the GPU.
Prompts + parsers live in `gedebate.prompts.debate` (format and parsing co-located).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gedebate.eval.scoring import score
from gedebate.prompts.debate import (
    critic_prompt,
    parse_critic,
    parse_proposer,
    proposer_prompt,
    revision_prompt,
)

if TYPE_CHECKING:
    from gedebate.data.instance import Instance
    from gedebate.model import Model

CONDITION = "debate"


def _proposer_turn(model, instance, prompt, max_new_tokens) -> dict:
    """Generate one Proposer turn (initial or revision) and parse it into a turn dict."""
    gen = model.generate(prompt, max_new_tokens=max_new_tokens)  # greedy
    value, ok, claims = parse_proposer(
        gen.text, instance.task, encoding=instance.encoding, node_ids=instance.node_ids
    )
    return {"role": "proposer", "raw": gen.text, "parsed": value, "parse_ok": ok,
            "claims": claims, "n_prompt_tokens": gen.n_prompt_tokens,
            "n_gen_tokens": gen.n_gen_tokens}


def run_debate(
    model: "Model", instance: "Instance", *,
    max_new_tokens: int = 256, max_responses: int = 10,
) -> tuple[dict, list[dict]]:
    """Run the debate loop -> (attempt record, transcript turns).

    `record` has baseline's shape (`condition="debate"`, final answer, correctness) plus
    summed `n_prompt_tokens`/`n_gen_tokens`, `n_responses` (= turns), and `critic_unparsed`
    (count of verdicts that defaulted to AGREE). `max_responses` is the response budget.
    """
    turns: list[dict] = [_proposer_turn(model, instance, proposer_prompt(instance), max_new_tokens)]
    answers = [turns[0]["parsed"]]  # parsed answer per Proposer turn (for no-progress)
    # True iff the debate stopped on an unparseable verdict (defaulted AGREE = "fake
    # consensus"). A flag, not a count: an unparseable verdict breaks the loop, so at
    # most one can occur, and it is always the terminal turn.
    stopped_on_unparsed_verdict = False

    while len(turns) < max_responses:
        cg = model.generate(critic_prompt(instance, turns), max_new_tokens=max_new_tokens)
        verdict, problems, parsed_ok = parse_critic(cg.text)
        turns.append({"role": "critic", "raw": cg.text, "verdict": verdict,
                      "problems": problems, "critic_verdict_parsed": parsed_ok,
                      "n_prompt_tokens": cg.n_prompt_tokens, "n_gen_tokens": cg.n_gen_tokens})
        if verdict == "AGREE":
            stopped_on_unparsed_verdict = not parsed_ok  # real agreement vs default
            break
        if len(turns) >= max_responses:
            break  # budget hit; no room to revise -> final is the last Proposer answer
        rev = _proposer_turn(model, instance, revision_prompt(instance, turns), max_new_tokens)
        turns.append(rev)
        if rev["parsed"] in answers:  # no progress: a repeated answer (incl. oscillation)
            break
        answers.append(rev["parsed"])

    final = next(t for t in reversed(turns) if t["role"] == "proposer")
    record = {
        "condition": CONDITION,
        "task": instance.task,
        "encoding": instance.encoding,
        "raw_output": final["raw"],
        "parsed_answer": final["parsed"],
        "parse_ok": final["parse_ok"],
        "correct": score(final["parsed"], instance.ground_truth),
        "ground_truth": instance.ground_truth,
        "n_prompt_tokens": sum(t["n_prompt_tokens"] for t in turns),
        "n_gen_tokens": sum(t["n_gen_tokens"] for t in turns),
        "n_responses": len(turns),
        "stopped_on_unparsed_verdict": stopped_on_unparsed_verdict,  # fake-consensus flag
    }
    return record, turns
