"""Debate (Proposer-Critic) prompts + their parsers, co-located in one file.

Prompt format and parsing are **coupled**: changing a prompt's output format MUST update
the matching parser here, so both live together (a change to one that forgets the other
is a bug we make hard to write). Answer-string -> value normalization stays in
`gedebate.eval.scoring` (shared with baseline / majority-vote); this module owns only the
debate-specific framing -- the numbered-claim trace, the `ANSWER:` line, the Critic's
`VERDICT:` -- plus the running transcript.

The three prompts are: turn-1 Proposer, Critic (verify the latest answer given the full
transcript), and Proposer revision. Prompts are **approved per task** (they are the
experiment's core); `node_degree` is implemented, other tasks raise until approved.
See docs/plan/p5-debate.md.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from gedebate.eval.scoring import parse as _parse_answer

if TYPE_CHECKING:
    from gedebate.data.instance import GroundTruth, Instance

# --- prompts (approved per task) ----------------------------------------------

# The `ANSWER:` line's task-specific tail (a human-facing format hint only; the value is
# normalized by scoring.parse). Kept for all tasks; the instruction wording is what gates.
ANSWER_FORMAT = {
    "edge_existence": "Yes or No",
    "node_degree": "a single integer, the degree",
    "connected_nodes": "a comma-separated list of node ids, or none",
}

_SUPPORTED = ("node_degree",)  # tasks whose prompts are approved + implemented (pilot)


def _require_supported(task: str) -> None:
    if task not in _SUPPORTED:
        raise NotImplementedError(
            f"debate prompts not yet approved for task {task!r}; supported: {_SUPPORTED}"
        )


# Per-task Proposer instruction (turn 1). Approved wording lives here verbatim.
_PROPOSER_INSTRUCTION = {
    "node_degree": (
        "Answer the question below using the graph. Build the answer as a numbered list of\n"
        "atomic claims -- each claim one verifiable fact about a single edge (that it exists,\n"
        "or that no further edge involves the node) -- then give the final answer.\n"
        "Use exactly this format and nothing else:\n"
        "1. <one atomic claim>\n"
        "2. <one atomic claim>\n"
        "(as many as needed)\n"
        "ANSWER: <a single integer, the degree>"
    ),
}

# Per-task revision format block (the corrected-answer format the Proposer must re-emit).
_REVISION_FORMAT = {
    "node_degree": (
        "Give your corrected answer in exactly this format and nothing else:\n"
        "1. <one atomic claim>\n"
        "2. <one atomic claim>\n"
        "(as many as needed)\n"
        "ANSWER: <a single integer, the degree>"
    ),
}

# Critic framing is task-generic (it verifies edge claims whatever the task).
_CRITIC_TOP = (
    "Another model is answering the graph question below by writing numbered atomic claims\n"
    "(each about one edge) and a final answer. You are the checker. The graph text is the\n"
    "ONLY source of truth: an edge exists only if the graph states it. Verify the LATEST\n"
    "Proposer answer against the graph; the debate so far is shown."
)
_CRITIC_CUE = (
    "Work only from the graph text. First find, from the graph, every edge that involves\n"
    "the queried node -- an edge counts only if the graph explicitly states it; never\n"
    "introduce an edge that is not written in the graph. Then compare the Proposer's claims\n"
    "and final answer to what the graph shows.\n"
    "Default to AGREE. Raise a problem ONLY when the Proposer's answer contradicts the graph\n"
    "AND you can quote the exact graph text that proves it. If you cannot quote such text, do\n"
    "not raise it. Do not repeat a problem the Proposer has already fixed. Respond in exactly\n"
    "this format and nothing else:\n"
    "VERDICT: AGREE\n"
    "(every claim matches the graph and the answer follows)\n"
    "or\n"
    "VERDICT: REVISE\n"
    "- <the problem, quoting the exact graph text that proves it>"
)
_REVISION_TOP = (
    "You are the Proposer in the debate below. A checker verified your latest claims against\n"
    "the graph and found problems. Produce a corrected answer that fixes them, using the\n"
    "whole exchange."
)


# --- running transcript -------------------------------------------------------

def render_transcript(instance: "Instance", turns: list[dict]) -> str:
    """Assemble the running debate text: the problem (`instance.question` verbatim), the
    turn-1 Proposer answer completing its trailing 'A: ', then labeled alternating turns.

    `turns` is the ordered prior turns, each `{'role': 'proposer'|'critic', 'raw': str}`.
    The graph encoding sits at the top and is re-read every turn (the dominant compute).
    """
    if not turns:
        return instance.question
    parts = [instance.question + turns[0]["raw"]]  # turn 1 completes the "A: "
    for t in turns[1:]:
        label = "Critic" if t["role"] == "critic" else "Proposer (revised)"
        parts.append(f"{label}: {t['raw']}")
    return "\n\n".join(parts)


# --- prompt builders ----------------------------------------------------------

def proposer_prompt(instance: "Instance") -> str:
    """Turn-1 Proposer prompt: numbered-claim trace + ANSWER, then the question verbatim."""
    _require_supported(instance.task)
    return f"{_PROPOSER_INSTRUCTION[instance.task]}\n\n{instance.question}"


def critic_prompt(instance: "Instance", turns: list[dict]) -> str:
    """Critic prompt: verify the latest Proposer answer given the full transcript."""
    _require_supported(instance.task)
    return f"{_CRITIC_TOP}\n\n{render_transcript(instance, turns)}\n\n{_CRITIC_CUE}"


def revision_prompt(instance: "Instance", turns: list[dict]) -> str:
    """Proposer revision prompt: corrected answer given the full transcript."""
    _require_supported(instance.task)
    return f"{_REVISION_TOP}\n\n{render_transcript(instance, turns)}\n\n{_REVISION_FORMAT[instance.task]}"


# --- parsers (co-located with the prompts they parse) -------------------------

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
_CLAIM_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$", re.MULTILINE)
_VERDICT_RE = re.compile(r"VERDICT:\s*(AGREE|REVISE)", re.IGNORECASE)
_PROBLEM_RE = re.compile(r"^\s*-\s*(.+?)\s*$", re.MULTILINE)


def parse_proposer(
    raw: str, task: str, *, encoding: str | None = None, node_ids: list | None = None
) -> tuple["GroundTruth | None", bool, list[str]]:
    """Proposer output -> (answer_value, parse_ok, claims).

    Extract the `ANSWER:` line and normalize it via `scoring.parse` (the shared answer
    parser, identical to baseline/MV); `claims` are the numbered lines (the trace, for
    the Critic + the viewer). Falls back to the whole text if there is no ANSWER line.
    """
    m = list(_ANSWER_RE.finditer(raw))
    answer_text = m[-1].group(1).strip() if m else raw
    value, ok = _parse_answer(task, answer_text, encoding=encoding, node_ids=node_ids)
    claims = [c.group(1) for c in _CLAIM_RE.finditer(raw)]
    return value, ok, claims


def parse_critic(raw: str) -> tuple[str, list[str], bool]:
    """Critic output -> (verdict, problems, parsed_ok).

    `verdict` in {'AGREE','REVISE'}. `parsed_ok` is False when no `VERDICT:` line is found
    -- the loop then treats it as AGREE **but counts it** (the fake-consensus guard).
    """
    m = _VERDICT_RE.search(raw)
    if not m:
        return "AGREE", [], False
    verdict = m.group(1).upper()
    problems = [p.group(1) for p in _PROBLEM_RE.finditer(raw)] if verdict == "REVISE" else []
    return verdict, problems, True
