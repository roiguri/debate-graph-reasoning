"""Unit tests for `gedebate.eval.scoring` -- the keystone answer contract.

Independent of the data layer and the model: parse/score are exercised on literal
inputs only. Covers the bool (edge_existence) parser path and the exact-match
scorer; the int (node_degree) and set (connected_nodes) parser paths and their
tests arrive with P2.4.
"""

from __future__ import annotations

import pytest

from gedebate.eval.scoring import parse, score


# --- parse: edge_existence (bool) ---------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes.", True),
        ("No.", False),
        ("yes", True),
        ("NO", False),
        ("The answer is Yes", True),
        ("No, actually yes.", True),  # last standalone yes/no wins
    ],
)
def test_parse_edge_existence_ok(text, expected):
    value, ok = parse("edge_existence", text)
    assert ok is True
    assert value is expected


@pytest.mark.parametrize("text", ["maybe?", "", "connected", "42"])
def test_parse_edge_existence_failure_is_measurable(text):
    # A failed parse is (None, False) -- never silently a wrong bool.
    value, ok = parse("edge_existence", text)
    assert value is None
    assert ok is False


def test_parse_unimplemented_task_raises():
    # Documents the P2.4 boundary rather than mis-parsing an unsupported task.
    with pytest.raises(NotImplementedError):
        parse("node_degree", "3")


# --- score: exact match -------------------------------------------------------

def test_score_exact_match_across_shapes():
    assert score(True, True) is True
    assert score(False, True) is False
    assert score(3, 3) is True
    assert score(2, 3) is False
    assert score([1, 2, 5], [1, 2, 5]) is True   # connected_nodes: list == set-equality
    assert score([1, 2], [1, 2, 5]) is False


def test_score_parse_failure_is_incorrect():
    assert score(None, True) is False
    assert score(None, False) is False
