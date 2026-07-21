"""Parse a model's free-text answer to a normalized value, then score it.

The parser is a keystone contract: it maps the model's output to the *same*
normalized type as `Instance.ground_truth` (bool | int | sorted list), so scoring
is exact-match rather than brittle string compare. `parse` always returns
`(value, parse_ok)` -- a failed parse yields `(None, False)` so it is *measurable*
and never silently counted as a wrong answer.

P2.1 implements the bool (edge_existence) path. The int (node_degree) and set
(connected_nodes) shapes are added in P2.4; the `task`-keyed dispatch is already
in place so that is purely additive.
"""

from __future__ import annotations

import re

from gedebate.data.instance import GroundTruth

_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse(task: str, text: str) -> tuple[GroundTruth | None, bool]:
    """Normalized answer + whether parsing succeeded, dispatched by task."""
    if task == "edge_existence":
        return _parse_bool(text)
    raise NotImplementedError(f"no parser for task '{task}' yet (added in P2.4)")


def _parse_bool(text: str) -> tuple[bool | None, bool]:
    """Last standalone yes/no wins -- a lead-in shouldn't override the final word."""
    matches = _YES_NO.findall(text)
    if not matches:
        return None, False
    return matches[-1].lower() == "yes", True


def score(parsed: GroundTruth | None, ground_truth: GroundTruth) -> bool:
    """Exact match against normalized ground truth. A parse failure is incorrect.

    `==` handles all three shapes: bool, int, and (sorted) list for
    connected_nodes -- both sides are sorted, so list equality is set-equality.
    """
    if parsed is None:
        return False
    return parsed == ground_truth
