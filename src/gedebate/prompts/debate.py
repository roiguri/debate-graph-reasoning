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

# --- prompt versions ----------------------------------------------------------
#
# **Both v1 and v2 are now FROZEN.** A version is frozen once results depend on it: edit
# it and those results stop being reproducible from the config that made them. v1 backs
# `results/main`, `seed11` and `seed13`; v2 was adopted after its pilot (docs/findings.md
# 3f) and backs the full re-run. Iterating either one now means adding **v3**, not editing
# in place. (v2 itself was iterated in place while it was still a draft feeding only
# throwaway pilot dirs, the same way the Critic wording went c1 -> c2 -> c3 during P5.)
#
# A config selects via `prompt_version`, defaulting to v1, so configs predating v2 are
# unaffected. Both are pinned by golden literals in tests/prompts/test_debate.py.
#
# v2 fixes two defects the P5 traces exposed (docs/findings.md 3c, 3d):
#   - v1's `connected_nodes` answer hint said "node ids" for every encoding, but that
#     answer is in the ENCODING's label space -- names under friendship. The model
#     obeyed the hint, answered in integers, and 64 of 600 friendship answers became
#     unparseable. v2 names no label space.
#   - v1's fill-in template ("1. <one atomic claim>") was copied verbatim until the model
#     hit the token cap, and it never reliably produced the ANSWER line. v2 states the
#     format in prose instead, and states the numbering explicitly.
#
# The explicit numbering is not cosmetic. The first v2 draft dropped the template for
# pure prose ("one claim per numbered line") and the model stopped numbering entirely
# (2/200 numbered, 174/200 switched to a "CLAIM:" prefix). On connected_nodes that cost
# 0.085 turn-1 accuracy: without the list scaffold the Proposer enumerated the whole
# graph's edges instead of the queried node's. See docs/findings.md 3f.
#
# v2 stays zero-shot: a worked example would fix the copying too, but it would make the
# Proposer one-shot and break comparability with the zero-shot baseline.

# The `ANSWER:` line's task-specific tail (a human-facing format hint; the value is
# normalized by scoring.parse).
_ANSWER_FORMAT_V1 = {
    "edge_existence": "Yes or No",
    "node_degree": "a single integer, the degree",
    "connected_nodes": "a comma-separated list of node ids, or none",
}
_ANSWER_FORMAT_V2 = {
    "edge_existence": "Yes or No",
    "node_degree": "a single integer, the degree",
    # Deliberately label-space free: "as the graph writes them" resolves to integers
    # under adjacency/incident and to names under friendship, with no per-encoding
    # override to keep in sync.
    "connected_nodes": ("the connected nodes, comma-separated, written exactly as the "
                        "graph writes them, or none"),
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

# The shared numbered-claim + ANSWER block. `_format_block` is used by BOTH the Proposer
# (turn 1) and the revision, so their format is identical by construction, not by copy.
_FORMAT_BLOCK_V1 = (
    "1. <one atomic claim>\n"
    "2. <one atomic claim>\n"
    "(as many as needed)\n"
    "ANSWER: <{answer}>"
)
_PROPOSER_PREAMBLE_V1 = (
    "Answer the question below using the graph. Build the answer as a numbered list of\n"
    "atomic claims -- {claim} -- then give the final answer.\n"
    "Use exactly this format and nothing else:\n"
)
_REVISION_PREAMBLE_V1 = "Give your corrected answer in exactly this format and nothing else:\n"

_FORMAT_BLOCK_V2 = (
    "Number your claims 1., 2., 3., and so on, one claim per line. Then write a\n"
    "final line that begins with ANSWER: followed by {answer}. Write nothing after\n"
    "that line."
)
_PROPOSER_PREAMBLE_V2 = (
    "Answer the question below using the graph. Build the answer as a numbered list of\n"
    "atomic claims -- {claim} -- then give the final answer.\n"
)
_REVISION_PREAMBLE_V2 = "Give your corrected answer.\n"

PROMPT_VERSIONS = {
    "v1": {"answer_format": _ANSWER_FORMAT_V1, "format_block": _FORMAT_BLOCK_V1,
           "proposer_preamble": _PROPOSER_PREAMBLE_V1,
           "revision_preamble": _REVISION_PREAMBLE_V1},
    "v2": {"answer_format": _ANSWER_FORMAT_V2, "format_block": _FORMAT_BLOCK_V2,
           "proposer_preamble": _PROPOSER_PREAMBLE_V2,
           "revision_preamble": _REVISION_PREAMBLE_V2},
}
# What a config gets when it does not ask. v1, so configs predating v2 are unaffected.
DEFAULT_PROMPT_VERSION = "v1"


def _spec(version: str) -> dict:
    try:
        return PROMPT_VERSIONS[version]
    except KeyError:
        raise ValueError(
            f"unknown prompt version {version!r}; known: {tuple(PROMPT_VERSIONS)}"
        ) from None


def supported_tasks(version: str = DEFAULT_PROMPT_VERSION) -> tuple[str, ...]:
    """Tasks this prompt version can build for.

    A task is supported iff it has both a claim kind and an answer format, the two
    per-task pieces the shared scaffold needs. Derived, so approving a task adds it in
    exactly one place.
    """
    return tuple(t for t in _CLAIM_KIND if t in _spec(version)["answer_format"])


def _require_supported(task: str, version: str = DEFAULT_PROMPT_VERSION) -> None:
    supported = supported_tasks(version)
    if task not in supported:
        raise NotImplementedError(
            f"debate prompts not yet approved for task {task!r}; supported: {supported}"
        )


def _format_block(task: str, version: str = DEFAULT_PROMPT_VERSION) -> str:
    """The claim-list + ANSWER format for `task`, shared by the proposer and the revision."""
    spec = _spec(version)
    return spec["format_block"].format(answer=spec["answer_format"][task])


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

def proposer_prompt(instance: "Instance", version: str = DEFAULT_PROMPT_VERSION) -> str:
    """Turn-1 Proposer prompt: numbered-claim trace + ANSWER, then the question verbatim."""
    spec = _spec(version)
    _require_supported(instance.task, version)
    instruction = (spec["proposer_preamble"].format(claim=_CLAIM_KIND[instance.task])
                   + _format_block(instance.task, version))
    return f"{instruction}\n\n{instance.question}"


def critic_prompt(
    instance: "Instance", turns: list[dict], version: str = DEFAULT_PROMPT_VERSION
) -> str:
    """Critic prompt: verify the latest Proposer answer given the full transcript.

    Takes `version` for a uniform call signature, but the Critic wording is currently
    identical across versions (v2 changed only the Proposer side), so it is unused here.
    """
    _require_supported(instance.task, version)
    return f"{_CRITIC_TOP}\n\n{render_transcript(instance, turns)}\n\n{_CRITIC_CUE[instance.task]}"


def revision_prompt(
    instance: "Instance", turns: list[dict], version: str = DEFAULT_PROMPT_VERSION
) -> str:
    """Proposer revision prompt: corrected answer given the full transcript."""
    spec = _spec(version)
    _require_supported(instance.task, version)
    fmt = spec["revision_preamble"] + _format_block(instance.task, version)
    return f"{_REVISION_TOP}\n\n{render_transcript(instance, turns)}\n\n{fmt}"


# --- parsers (co-located with the prompts they parse) -------------------------

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
_CLAIM_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$", re.MULTILINE)
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
    falls back to the whole text, which is a *silent* degradation -- it happens to work
    for node_degree (last integer wins) and fails for connected_nodes -- so the rate is
    worth reporting rather than inferring from `parse_ok`.
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
    harvest's 0.210 / 0.455 (docs/findings.md 3g).
    """
    m = list(_ANSWER_RE.finditer(raw))
    answer_text = m[-1].group(1).strip() if m else _last_line(raw)
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
