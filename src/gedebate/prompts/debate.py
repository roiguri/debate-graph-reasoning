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

# --- prompts: shared scaffold + per-task variations (drift-guarded) ------------
#
# The scaffold (numbered atomic-claim trace + ANSWER line, the revision format, the Critic
# framing) is written ONCE and shared across tasks. A task varies only where it must -- the
# ANSWER line's tail (ANSWER_FORMAT) and what one claim asserts (_CLAIM_KIND). The Proposer
# (turn 1) and the revision reuse the SAME format block, so their format cannot drift apart.
# No encoding-specific wording is needed today (the encoding enters via instance.question);
# an encoding variation would slot in the same dict-override way if one is ever required.

# The `ANSWER:` line's task-specific tail (a human-facing format hint; the value is
# normalized by scoring.parse).
ANSWER_FORMAT = {
    "edge_existence": "Yes or No",
    "node_degree": "a single integer, the degree",
    "connected_nodes": "a comma-separated list of node ids, or none",
}

# What one atomic claim asserts, per task. node_degree and connected_nodes both reason over
# the queried node's incident edges, so they share ONE constant (they cannot drift);
# edge_existence gets its own wording when its prompt is approved.
_INCIDENT_CLAIM = (
    "each claim one verifiable fact about a single edge (that it exists,\n"
    "or that no further edge involves the node)"
)
_EDGE_CLAIM = (
    "each claim one verifiable fact about the queried pair (whether that exact edge\n"
    "appears in the graph's edge list)"
)
_CLAIM_KIND = {
    "node_degree": _INCIDENT_CLAIM,
    "connected_nodes": _INCIDENT_CLAIM,
    "edge_existence": _EDGE_CLAIM,
}

# A task is supported iff it has both a claim kind and an answer format -- the two per-task
# pieces the shared scaffold needs. Derived, so a task is added in exactly one place.
_SUPPORTED = tuple(t for t in _CLAIM_KIND if t in ANSWER_FORMAT)


def _require_supported(task: str) -> None:
    if task not in _SUPPORTED:
        raise NotImplementedError(
            f"debate prompts not yet approved for task {task!r}; supported: {_SUPPORTED}"
        )


# The shared numbered-claim + ANSWER block. `_format_block` is used by BOTH the Proposer
# (turn 1) and the revision, so their format is identical by construction, not by copy.
_FORMAT_BLOCK = (
    "1. <one atomic claim>\n"
    "2. <one atomic claim>\n"
    "(as many as needed)\n"
    "ANSWER: <{answer}>"
)
_PROPOSER_PREAMBLE = (
    "Answer the question below using the graph. Build the answer as a numbered list of\n"
    "atomic claims -- {claim} -- then give the final answer.\n"
    "Use exactly this format and nothing else:\n"
)
_REVISION_PREAMBLE = "Give your corrected answer in exactly this format and nothing else:\n"


def _format_block(task: str) -> str:
    """The claim-list + ANSWER format for `task`, shared by the proposer and the revision."""
    return _FORMAT_BLOCK.format(answer=ANSWER_FORMAT[task])


# Critic framing. `_CRITIC_TOP` is shared. The cue varies by task where the verification
# differs: node_degree + connected_nodes both check the queried node's incident edges, so
# they share ONE cue (they cannot drift); edge_existence checks a single pair, so it gets
# its own. `_CRITIC_CUE[task]` selects.
_CRITIC_TOP = (
    "Another model is answering the graph question below by writing numbered atomic claims\n"
    "(each about one edge) and a final answer. You are the checker. The graph text is the\n"
    "ONLY source of truth: an edge exists only if it appears in the graph's edge list. Work\n"
    "the answer out yourself from the graph, then verify the LATEST Proposer answer; the\n"
    "debate so far is shown."
)
_CRITIC_CUE_INCIDENT = (
    "Work only from the graph text. First, independently go through the graph's edge list and\n"
    "pick out every edge that involves the queried node, copying each exactly as written (an\n"
    "edge counts only if it is in the list; never introduce one that is not there). Those\n"
    "edges determine the correct answer. Then compare them to the Proposer's claims and final\n"
    "answer, in both directions:\n"
    "- did the Proposer MISS an edge that is in the list?\n"
    "- did the Proposer INCLUDE an edge that is not in the list?\n"
    "AGREE only if the Proposer's supporting edges are exactly the ones in the graph and the\n"
    "answer follows. Otherwise REVISE and name each wrong edge, quoting it from the graph's\n"
    "edge list. Every problem you raise must cite an edge that actually appears in the list;\n"
    "do not repeat a problem the Proposer has already fixed. Respond in exactly this format\n"
    "and nothing else:\n"
    "VERDICT: AGREE\n"
    "or\n"
    "VERDICT: REVISE\n"
    "- <an edge the Proposer missed or wrongly included, quoted from the graph's edge list>"
)
_CRITIC_CUE_EDGE = (
    "Work only from the graph text. Look in the graph's edge list for the exact pair the\n"
    "question asks about, in either order. If that pair appears, the answer is Yes; if it\n"
    "does not, the answer is No -- decide this yourself from the list. Then check the\n"
    "Proposer's answer.\n"
    "AGREE if the Proposer's Yes/No matches the edge list. Otherwise REVISE and ground it:\n"
    "quote the queried edge from the list if it is there, or state that no such edge appears.\n"
    "Every problem you raise must be grounded in the edge list; never invent an edge. Respond\n"
    "in exactly this format and nothing else:\n"
    "VERDICT: AGREE\n"
    "or\n"
    "VERDICT: REVISE\n"
    "- <the queried edge quoted from the list, or a note that it does not appear>"
)
_CRITIC_CUE = {
    "node_degree": _CRITIC_CUE_INCIDENT,
    "connected_nodes": _CRITIC_CUE_INCIDENT,
    "edge_existence": _CRITIC_CUE_EDGE,
}
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
    instruction = (_PROPOSER_PREAMBLE.format(claim=_CLAIM_KIND[instance.task])
                   + _format_block(instance.task))
    return f"{instruction}\n\n{instance.question}"


def critic_prompt(instance: "Instance", turns: list[dict]) -> str:
    """Critic prompt: verify the latest Proposer answer given the full transcript."""
    _require_supported(instance.task)
    return f"{_CRITIC_TOP}\n\n{render_transcript(instance, turns)}\n\n{_CRITIC_CUE[instance.task]}"


def revision_prompt(instance: "Instance", turns: list[dict]) -> str:
    """Proposer revision prompt: corrected answer given the full transcript."""
    _require_supported(instance.task)
    fmt = _REVISION_PREAMBLE + _format_block(instance.task)
    return f"{_REVISION_TOP}\n\n{render_transcript(instance, turns)}\n\n{fmt}"


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
