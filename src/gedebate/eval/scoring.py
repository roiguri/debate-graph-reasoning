"""Parse a model's free-text answer to a normalized value, then score it.

The parser is a keystone contract: it maps the model's output to the *same*
normalized type as `Instance.ground_truth` (bool | int | sorted list), so scoring
is exact-match rather than brittle string compare. `parse` always returns
`(value, parse_ok)` -- a failed parse yields `(None, False)` so it is *measurable*
and never silently counted as a wrong answer.

The three answer shapes -- bool (edge_existence), int (node_degree), and sorted
list (connected_nodes) -- are dispatched by task.
"""

from __future__ import annotations

import re

from gedebate.data.instance import GroundTruth
from gedebate.graphqa import graph_text_encoder

_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_INT = re.compile(r"-?\d+")
_WORD = re.compile(r"[A-Za-z0-9]+")
# Explicit "empty set" markers for connected_nodes (matches the "none" instruction
# and GraphQA's own "No nodes." gold answer).
_EMPTY_MARKER = re.compile(r"\b(none|no nodes|no other|nothing|empty)\b", re.IGNORECASE)


def parse(
    task: str,
    text: str,
    *,
    encoding: str | None = None,
    node_ids: list | None = None,
) -> tuple[GroundTruth | None, bool]:
    """Normalized answer + whether parsing succeeded, dispatched by task.

    `encoding`/`node_ids` are only needed for connected_nodes, whose answer is in
    the encoding's label space (integers for adjacency/incident, names for
    friendship) and whose source node must be excluded.
    """
    if task == "edge_existence":
        return _parse_bool(text)
    if task == "node_degree":
        return _parse_int(text)
    if task == "connected_nodes":
        return _parse_node_list(text, encoding, node_ids)
    raise NotImplementedError(f"no parser for task '{task}'")


def _parse_bool(text: str) -> tuple[bool | None, bool]:
    """Last standalone yes/no wins -- a lead-in shouldn't override the final word."""
    matches = _YES_NO.findall(text)
    if not matches:
        return None, False
    return matches[-1].lower() == "yes", True


def _parse_int(text: str) -> tuple[int | None, bool]:
    """Last integer wins -- the degree usually trails any echoed node id."""
    matches = _INT.findall(text)
    if not matches:
        return None, False
    return int(matches[-1]), True


def _parse_node_list(
    text: str, encoding: str | None, node_ids: list | None
) -> tuple[list | None, bool]:
    """Map answer labels back to sorted node ids, dropping the (never-neighbor) source.

    Labels are resolved through the encoding's id->label table, so this handles both
    integer encodings ("1, 2") and named ones ("James, John"). An explicit empty
    marker ("none" / "No nodes.") yields []; unrecognized output yields a failure.
    """
    if encoding is None:
        raise ValueError("connected_nodes parsing requires the encoding")
    label_to_id = {
        label: nid for nid, label in graph_text_encoder.TEXT_ENCODER_DICT[encoding].items()
    }
    source = node_ids[0] if node_ids else None

    found = {
        label_to_id[tok]
        for tok in _WORD.findall(text)
        if tok in label_to_id and label_to_id[tok] != source
    }
    if found:
        return sorted(found), True
    if _EMPTY_MARKER.search(text):
        return [], True
    return None, False


def score(parsed: GroundTruth | None, ground_truth: GroundTruth) -> bool:
    """Exact match against normalized ground truth. A parse failure is incorrect.

    `==` handles all three shapes: bool, int, and (sorted) list for
    connected_nodes -- both sides are sorted, so list equality is set-equality.
    """
    if parsed is None:
        return False
    return parsed == ground_truth
