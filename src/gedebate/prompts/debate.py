"""Debate (Proposer-Critic) prompts + their parsers, co-located in one file.

Prompt format and parsing are **coupled**: changing a prompt's output format MUST update
the matching parser here, so both live together (a change to one that forgets the other
is a bug we make hard to write). Answer-string -> value normalization stays in
`gedebate.eval.scoring` (shared with baseline / majority-vote); this module owns only the
debate-specific framing -- the numbered-claim trace, the `ANSWER:` line, the Critic's
`VERDICT:` -- plus the running transcript.

The three prompts are: turn-1 Proposer, Critic (verify the latest answer given the full
transcript), and Proposer revision. All three tasks are approved; an unapproved task
raises. The wording is FROZEN -- see the comment below.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from gedebate.eval.scoring import parse as _parse_answer

if TYPE_CHECKING:
    from gedebate.data.instance import GroundTruth, Instance

# --- the prompts ---------------------------------------------------------------
#
# **One wording, FROZEN.** This is the text every reported result was produced under;
# never edit it in place. Earlier iterations were deleted once the runs they backed were
# superseded, and their reasoning is worth keeping because it is what this wording is
# built to avoid:
#
#   * an early version's connected_nodes answer hint said "node ids" for every encoding,
#     so under friendship the model answered in integers and 64 of 600 answers were
#     unparseable;
#   * a later revision was significantly WORSE on five of nine cells over 5,400 paired
#     instances, 0.450 mean against 0.496, and drove Proposer capitulation from 0.63 to
#     0.96.
#
# Changing any of this text is a new experiment, not a fix: the committed results depend
# on it byte for byte, and the run's git commit is what identifies which text produced a
# given row.

_ANSWER_FORMAT = {
    "edge_existence": "Yes or No",
    # Deliberately label-space free: "as the graph writes them" resolves to integers
    # under adjacency/incident and to names under friendship, with no per-encoding
    # override to keep in sync.
    "node_degree": "a single integer, the degree",
    "connected_nodes": ("the connected nodes, comma-separated, written exactly as the "
                        "graph writes them, or none"),
}
# ONE preamble for all three tasks: what a claim asserts is stated generically ("about the
# graph's nodes or edges"), so there is no per-task slot left. An earlier wording filled a
# `{claim}` hole from a per-task dict; the only per-task detail remaining is the ANSWER
# hint above.
_PREAMBLE = (
    "Answer the question below using the graph. Build the answer as a numbered list of\n"
    "concise, atomic claims -- each stating exactly one simple, verifiable fact about\n"
    "the graph's nodes or edges -- then give the final answer.\n"
)
# Resolved per task here rather than at build time, so `proposer_prompt` reads one
# task -> preamble mapping and needs no branch.
_PROPOSER_PREAMBLE = {task: _PREAMBLE for task in _ANSWER_FORMAT}
# The shared claim-list + ANSWER block, used by BOTH the Proposer (turn 1) and the
# revision, so their format is identical by construction rather than by copy. It states
# the numbering explicitly and carries no fill-in template: an early "1. <one atomic
# claim>" placeholder was echoed verbatim to the token cap, and a draft that dropped the
# scaffold entirely made the model stop numbering and cost 0.085 turn-1 accuracy.
_FORMAT_BLOCK = (
    "Number your claims 1., 2., 3., and so on, one claim per line. Then write a\n"
    "final line that begins with ANSWER: followed by {answer}. Write nothing after\n"
    "that line."
)
_REVISION_PREAMBLE = "Give your corrected answer.\n"
# SETUP ONLY -- who is speaking, what they produced, and that a transcript follows. The
# rules themselves live in the cues below, which is the single place they are stated: an
# earlier top carried four instructions (work only from the graph text, an edge counts
# only if the list has it, derive the answer yourself, verify the latest answer) that both
# cues already restated, so it said every one of them twice.
_CRITIC_TOP = (
    "Another model is answering the graph question below by writing numbered atomic claims\n"
    "(each about one edge or node) and a final answer. You are the checker; the debate so\n"
    "far is shown."
)
# Reviews the Proposer's claims rather than deriving the answer first.
_CRITIC_CUE_INCIDENT = (
    "Work only from the graph text. Review the Proposer's claims and final\n"
    "answer, in both directions:\n"
    "- did the Proposer MISS an edge that is in the graph?\n"
    "- did the Proposer INCLUDE an edge that is not in the graph?\n"
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
# node_degree and connected_nodes both reason over the queried node's incident edges, so
# they share one cue; edge_existence checks a single pair, so it gets its own.
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


def supported_tasks() -> tuple[str, ...]:
    """Tasks the debate prompts can be built for.

    A task is supported iff EVERY per-task piece exists for it: an answer format, a
    Critic cue, and a Proposer preamble. Derived, so approving a task adds it in exactly
    one place, and a task missing any one piece cannot pass here and then KeyError inside
    a builder.
    """
    return tuple(t for t in _ANSWER_FORMAT
                 if t in _CRITIC_CUE and t in _PROPOSER_PREAMBLE)


def _require_supported(task: str) -> None:
    supported = supported_tasks()
    if task not in supported:
        raise NotImplementedError(
            f"debate prompts not yet approved for task {task!r}; supported: {supported}"
        )


def _format_block(task: str) -> str:
    """The claim-list + ANSWER format for `task`, shared by the proposer and the revision."""
    return _FORMAT_BLOCK.format(answer=_ANSWER_FORMAT[task])


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
    instruction = _PROPOSER_PREAMBLE[instance.task] + _format_block(instance.task)
    return f"{instruction}\n\n{instance.question}"


def critic_prompt(instance: "Instance", turns: list[dict]) -> str:
    """Critic prompt: verify the latest Proposer answer given the full transcript."""
    _require_supported(instance.task)
    cue = _CRITIC_CUE[instance.task]
    return f"{_CRITIC_TOP}\n\n{render_transcript(instance, turns)}\n\n{cue}"


def revision_prompt(instance: "Instance", turns: list[dict]) -> str:
    """Proposer revision prompt: corrected answer given the full transcript."""
    _require_supported(instance.task)
    fmt = _REVISION_PREAMBLE + _format_block(instance.task)
    return f"{_REVISION_TOP}\n\n{render_transcript(instance, turns)}\n\n{fmt}"


# --- parsers (co-located with the prompts they parse) -------------------------

# Scoped to the REST OF THE ANSWER LINE (`.` never crosses a newline), so a bare
# `ANSWER:` matches with an empty capture instead of not matching at all. The old
# `\s*(.+)` skipped over the newline to harvest a later line, which meant a model that
# wrote `ANSWER:` and stopped -- the format's own way of saying "none" -- fell through to
# the last-line fallback and landed on the string "ANSWER:", scoring a parse failure.
_ANSWER_RE = re.compile(r"ANSWER:[ \t]*(.*)", re.IGNORECASE)
_CLAIM_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$", re.MULTILINE)
# A leading claim number on a fallback line ("5. The nodes connected to 3 are 0.").
# Under adjacency/incident the labels ARE integers, so the claim number parses as a node.
_CLAIM_NUM_RE = re.compile(r"^\s*\d+[.:]\s+")
_VERDICT_RE = re.compile(r"VERDICT:\s*(AGREE|REVISE)", re.IGNORECASE)
_PROBLEM_RE = re.compile(r"^\s*-\s*(.+?)\s*$", re.MULTILINE)


def _last_line(raw: str) -> str:
    """The last non-empty line: the answer fallback when the `ANSWER:` line is missing."""
    lines = [line for line in raw.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def has_answer_line(raw: str) -> bool:
    """Whether a Proposer turn emitted the `ANSWER:` line the format block asks for.

    Lives here (not in the diagnostics that consume it) for the same reason the parsers
    do: it is a fact about the prompt's output format. Without the line `parse_proposer`
    falls back to the last line, a *silent* degradation -- it happens to work for
    node_degree (last integer wins) and misreads a claim line for connected_nodes -- so
    the rate is worth reporting rather than inferring from `parse_ok`.

    A bare `ANSWER:` counts as emitted: the model produced the line the format asked for,
    and an empty one is how it says "none". Format compliance and answer content are
    different facts, and this reports the first.
    """
    return _ANSWER_RE.search(raw) is not None


def parse_proposer(
    raw: str, task: str, *, encoding: str | None = None, node_ids: list | None = None
) -> tuple["GroundTruth | None", bool, list[str]]:
    """Proposer output -> (answer_value, parse_ok, claims).

    Extract the `ANSWER:` line and normalize it via `scoring.parse` (the shared answer
    parser, identical to baseline/MV); `claims` are the numbered lines (the trace, for
    the Critic + the viewer).

    **The reasoning is never parsed.** With no `ANSWER:` line the fallback is the LAST
    NON-EMPTY LINE, not the whole output. Scanning the whole output was not answer
    extraction at all for connected_nodes: `scoring._parse_node_list` collects every
    recognized label it sees, so a claim like "Robert is *not* connected to Susan" put
    Susan in the answer. Last-line matches what `_parse_int`/`_parse_bool` already do
    ("the answer is the last thing stated") and what the label-free baseline relies on,
    and it costs nothing: on seed 7 it scores 0.210 / 0.460 against the old whole-text
    harvest's 0.210 / 0.455.

    Two shapes the raw text takes that the answer parser must not see:

    **An empty `ANSWER:` line is the empty set, not a parse failure.** The format hint
    ends "...or none", and the model's way of complying is to write `ANSWER:` and stop.
    That is a real answer for `connected_nodes` and a non-answer for the other two tasks,
    whose answers (a degree, a Yes/No) are never empty. It is not truncation: every such
    output stopped well inside the token cap.

    **A fallback line keeps its claim number out of the answer.** The fallback lands on a
    numbered line often enough ("5. The nodes connected to 3 are 0.") that the leading "5"
    was being harvested as node 5 under the integer encodings.
    """
    claims = [c.group(1) for c in _CLAIM_RE.finditer(raw)]
    m = list(_ANSWER_RE.finditer(raw))
    if not m:
        answer_text = _CLAIM_NUM_RE.sub("", _last_line(raw))
    else:
        answer_text = m[-1].group(1).strip()
        if not answer_text:
            # `ANSWER:` ends the line. Content on a LATER line is still the answer (the
            # old regex's one useful behaviour); nothing after it at all means "none".
            following = [ln for ln in raw[m[-1].end():].splitlines() if ln.strip()]
            if following:
                answer_text = following[0].strip()
            elif task == "connected_nodes":
                return [], True, claims
            else:
                return None, False, claims
    value, ok = _parse_answer(task, answer_text, encoding=encoding, node_ids=node_ids)
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
