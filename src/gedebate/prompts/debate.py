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
# **v1 is FROZEN.** A version is frozen once results depend on it: edit it and those
# results stop being reproducible from the config that made them. v1 backs `results/main`,
# `seed11` and `seed13`, and must stay byte-identical.
#
# **v2 was edited in place after `results/v2-*` were produced** (the symmetry fix below),
# by explicit decision rather than by adding a v3. The cost is recorded here because
# nothing else records it: those runs' manifests say `prompt_version: "v2"`, but the v2
# they ran is the text as of commit 73ba8e0, not the text in this file. Re-running that
# config now produces different prompts, so `results/v2-*` are no longer reproducible from
# it. Anything comparing new v2 output against those rows is comparing two prompts.
#
# Because the claim kinds were shared across versions, editing v2's would have silently
# rewritten v1's prompts too; `claim_kind` is now part of the per-version spec so v1 is
# unaffected.
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
    # override to keep in sync. "the other end" is the edge->node step: the claims are
    # about edges (pairs) and the answer is nodes, and nothing used to bridge them, so
    # 578 answers listed the queried node itself -- 405 of them after claims that had
    # already named the right edges, i.e. the reasoning was right and the write-up lost it.
    "connected_nodes": ("the node at the other end of each of those edges, "
                        "comma-separated, written exactly as the graph writes them, "
                        "or none"),
}

# What one atomic claim asserts, per task. node_degree and connected_nodes both reason over
# the queried node's incident edges, so they share ONE constant (they cannot drift);
# edge_existence gets its own wording when its prompt is approved.
_INCIDENT_CLAIM_V1 = (
    "each claim one verifiable fact about a single edge (that it exists,\n"
    "or that no further edge involves the node)"
)
_EDGE_CLAIM_V1 = (
    "each claim one verifiable fact about the queried pair (whether that exact edge\n"
    "appears in the graph's edge list)"
)
# v2 states that the relation is symmetric in the TEXT, which neither v1 claim kind did.
# The Proposer read an edge only in the direction it was written: recall of a true
# neighbour was 0.909 when the queried node was written first and 0.333 when it was
# written second (adjacency; friendship 0.850/0.653, incident 0.870/0.778), and on
# edge_existence a true edge asked in the reverse of its written order scored 0.678
# against 0.963. Not early stopping -- recall is flat across the list's first, middle and
# last third. The Critic was told this from the start ("in either order", "every edge that
# involves the queried node") and the Proposer never was.
_INCIDENT_CLAIM_V2 = (
    "each claim one verifiable fact about a single edge (that it exists,\n"
    "or that no further edge involves the node); an edge involves the node whether\n"
    "the node is written first or second"
)
_EDGE_CLAIM_V2 = (
    "each claim one verifiable fact about the queried pair (whether that pair, in\n"
    "either order, appears in the graph's edge list)"
)
_CLAIM_KIND_V1 = {
    "node_degree": _INCIDENT_CLAIM_V1,
    "connected_nodes": _INCIDENT_CLAIM_V1,
    "edge_existence": _EDGE_CLAIM_V1,
}
_CLAIM_KIND_V2 = {
    "node_degree": _INCIDENT_CLAIM_V2,
    "connected_nodes": _INCIDENT_CLAIM_V2,
    "edge_existence": _EDGE_CLAIM_V2,
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
# "corrected", like v1's "found problems", presupposed that a change is due. The answer
# may be right; the instruction should not decide that before the graph is re-read.
_REVISION_PREAMBLE_V2 = "Give your answer.\n"

# The registry itself is assembled further down, once the Critic cues it collects have
# been defined (see PROMPT_VERSIONS below the Critic section).

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
    spec = _spec(version)
    return tuple(t for t in spec["claim_kind"] if t in spec["answer_format"])


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
_CRITIC_CUE_INCIDENT_V1 = (
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
# v2 fixes three defects the P5 traces exposed, all of them about what the format LETS the
# Critic say rather than about its judgement:
#   - AGREE had no evidence slot, so a Critic holding the evidence for a correct answer had
#     only the REVISE channel to put it in. On connected_nodes it REVISEd a correct "none"
#     while citing an edge from elsewhere in the graph.
#   - "cite an edge that actually appears in the list" constrained existence but not
#     RELEVANCE: only 39 percent of problem lines cited an edge involving the queried node,
#     12 percent cited a real but unrelated edge, 11 percent an edge in no list at all.
#   - AGREE was gated on the CLAIMS ("supporting edges are exactly the ones in the graph")
#     while the experiment scores the ANSWER, so a right answer with an untidy trace had to
#     be REVISEd. 19 of 76 false alarms on connected_nodes/adjacency then turned a right
#     answer wrong -- in one, a perfect 7-claim answer was revised down to a single node.
_CRITIC_CUE_INCIDENT_V2 = (
    "Work only from the graph text. First, independently go through the graph's edge list and\n"
    "pick out every edge that involves the queried node, copying each exactly as written (an\n"
    "edge counts only if it is in the list; never introduce one that is not there). Those\n"
    "edges determine the correct answer. Then compare them to the Proposer's final answer, in\n"
    "both directions:\n"
    "- did the Proposer MISS an edge that is in the list?\n"
    "- did the Proposer INCLUDE an edge that is not in the list?\n"
    "Judge the answer, not the tidiness of the claims: AGREE whenever the answer matches the\n"
    "edges you found, including when the correct answer is that no edge involves the node.\n"
    "Otherwise REVISE and name each wrong edge. Every edge you cite, under either verdict,\n"
    "must appear in the list and involve the queried node; do not repeat a problem the\n"
    "Proposer has already fixed. Respond in exactly this format and nothing else:\n"
    "VERDICT: AGREE\n"
    "- <an edge from the list that supports the answer, or: no edge involves the node>\n"
    "or\n"
    "VERDICT: REVISE\n"
    "- <an edge the Proposer missed or wrongly included, quoted from the graph's edge list>"
)
_CRITIC_CUE_EDGE_V1 = (
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
# v1 put the evidence for a correct "No" -- "or state that no such edge appears" -- inside
# the REVISE branch, and gave AGREE no bullet to put it in. So a Critic that had correctly
# verified the pair was absent filed that finding as a REVISE: on gold=False instances it
# REVISEd 96.3 percent of the time, and 1024 of 2308 Critic turns (44.4 percent) were
# REVISEs whose only grounding asserted absence against a Proposer who had answered No.
# Reading those as the agreement they are moves false alarm 0.774 -> 0.305 and the
# detection gap +0.180 -> +0.258, i.e. findings.md 4e was partly measuring this format.
_CRITIC_CUE_EDGE_V2 = (
    "Work only from the graph text. Look in the graph's edge list for the exact pair the\n"
    "question asks about, in either order. If that pair appears, the answer is Yes; if it\n"
    "does not, the answer is No -- decide this yourself from the list. Then check the\n"
    "Proposer's answer.\n"
    "AGREE if the Proposer's Yes/No matches the edge list. That includes the case where the\n"
    "pair is absent and the Proposer answered No: absence is the evidence FOR that answer,\n"
    "not a problem with it. Otherwise REVISE. Ground both verdicts the same way -- quote the\n"
    "queried pair from the list if it is there, or say that it does not appear -- and never\n"
    "invent an edge. Respond in exactly this format and nothing else:\n"
    "VERDICT: AGREE\n"
    "- <the queried pair quoted from the list, or a note that it does not appear>\n"
    "or\n"
    "VERDICT: REVISE\n"
    "- <the queried pair quoted from the list, or a note that it does not appear>"
)
_CRITIC_CUE_V1 = {
    "node_degree": _CRITIC_CUE_INCIDENT_V1,
    "connected_nodes": _CRITIC_CUE_INCIDENT_V1,
    "edge_existence": _CRITIC_CUE_EDGE_V1,
}
_CRITIC_CUE_V2 = {
    "node_degree": _CRITIC_CUE_INCIDENT_V2,
    "connected_nodes": _CRITIC_CUE_INCIDENT_V2,
    "edge_existence": _CRITIC_CUE_EDGE_V2,
}

# (the version registry is assembled at the end of this section, once every per-version
# piece it collects -- including the revision framing below -- has been defined)
_REVISION_TOP_V1 = (
    "You are the Proposer in the debate below. A checker verified your latest claims against\n"
    "the graph and found problems. Produce a corrected answer that fixes them, using the\n"
    "whole exchange."
)
# v1 pointed the reviser at the TRANSCRIPT ("using the whole exchange") and nowhere at the
# graph: `_CRITIC_TOP` opens with "the graph text is the ONLY source of truth" and the
# revision had no equivalent line. It also asserted the objections were real ("found
# problems", "verified") before the Proposer had looked, and never said an answer may
# stand. So the Proposer treated an objection as evidence instead of a claim to check:
# where the Critic asserted the queried pair was absent against a "Yes", it flipped to "No"
# 330 times out of 482 (68.5 percent), and 151 of those flips (45.8 percent) abandoned a
# CORRECT answer. It did not re-derive anything -- it rewrote the contested claim into its
# own negation in the Critic's words and left the uncontested claims untouched.
#
# This is not the Critic's false-alarm problem wearing a different hat. Those 482 are
# genuine disagreements, so the v2 Critic still says REVISE on them; the deference is the
# revision prompt's own.
_REVISION_TOP_V2 = (
    "You are the Proposer in the debate below. A checker has objected to your latest\n"
    "answer. The graph text is the ONLY source of truth: check each objection against\n"
    "the graph's edge list yourself. Keep your answer if the objection is wrong; change\n"
    "it only if the objection is right."
)


# Every per-version piece -- Proposer scaffold, claim kind, answer format, Critic cue,
# revision framing -- collected in one place. Lives at the end of the section rather than
# beside DEFAULT_PROMPT_VERSION because a dict literal evaluates its values immediately,
# so it must follow every constant it names.
PROMPT_VERSIONS = {
    "v1": {"answer_format": _ANSWER_FORMAT_V1, "format_block": _FORMAT_BLOCK_V1,
           "proposer_preamble": _PROPOSER_PREAMBLE_V1,
           "revision_preamble": _REVISION_PREAMBLE_V1,
           "claim_kind": _CLAIM_KIND_V1, "critic_cue": _CRITIC_CUE_V1,
           "revision_top": _REVISION_TOP_V1},
    "v2": {"answer_format": _ANSWER_FORMAT_V2, "format_block": _FORMAT_BLOCK_V2,
           "proposer_preamble": _PROPOSER_PREAMBLE_V2,
           "revision_preamble": _REVISION_PREAMBLE_V2,
           "claim_kind": _CLAIM_KIND_V2, "critic_cue": _CRITIC_CUE_V2,
           "revision_top": _REVISION_TOP_V2},
}


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
    instruction = (spec["proposer_preamble"].format(claim=spec["claim_kind"][instance.task])
                   + _format_block(instance.task, version))
    return f"{instruction}\n\n{instance.question}"


def critic_prompt(
    instance: "Instance", turns: list[dict], version: str = DEFAULT_PROMPT_VERSION
) -> str:
    """Critic prompt: verify the latest Proposer answer given the full transcript.

    The cue is per-version: v2 gives AGREE its own evidence bullet, ties grounding to the
    queried node, and judges the answer rather than the claim list. v1's wording is frozen
    because `results/main`, `seed11` and `seed13` were produced under it.
    """
    _require_supported(instance.task, version)
    cue = _spec(version)["critic_cue"][instance.task]
    return f"{_CRITIC_TOP}\n\n{render_transcript(instance, turns)}\n\n{cue}"


def revision_prompt(
    instance: "Instance", turns: list[dict], version: str = DEFAULT_PROMPT_VERSION
) -> str:
    """Proposer revision prompt: re-answer given the full transcript.

    The framing is per-version: v2 sends the Proposer back to the graph and says the
    answer may stand, where v1 asserted the objections were real and pointed only at the
    exchange. v1's wording is frozen -- `results/main`, `seed11` and `seed13` used it.
    """
    spec = _spec(version)
    _require_supported(instance.task, version)
    fmt = spec["revision_preamble"] + _format_block(instance.task, version)
    return f"{spec['revision_top']}\n\n{render_transcript(instance, turns)}\n\n{fmt}"


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
    harvest's 0.210 / 0.455 (docs/findings.md 3g).

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
