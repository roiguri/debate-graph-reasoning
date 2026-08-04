"""Unit tests for `gedebate.eval.scoring` -- the keystone answer contract.

Independent of the data layer and the model: parse/score are exercised on literal
inputs only. Covers the bool (edge_existence) parser path and the exact-match
scorer; the int (node_degree) and set (connected_nodes) parser paths and their
tests live alongside.
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


def test_parse_unknown_task_raises():
    with pytest.raises(NotImplementedError):
        parse("cycle_check", "yes")


# --- parse: node_degree (int) -------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("3.", 3),
        ("3", 3),
        ("0", 0),
        ("The degree is 3.", 3),
        ("Node 5 has degree 3.", 3),   # last integer wins over the echoed node id
    ],
)
def test_parse_node_degree_ok(text, expected):
    value, ok = parse("node_degree", text)
    assert ok is True
    assert value == expected


@pytest.mark.parametrize("text", ["", "many", "no idea"])
def test_parse_node_degree_failure(text):
    value, ok = parse("node_degree", text)
    assert value is None and ok is False


# --- parse: connected_nodes (set, encoding-aware) -----------------------------

def test_parse_connected_nodes_integer_encoding():
    # adjacency/incident: labels are integers.
    value, ok = parse("connected_nodes", "1, 2.", encoding="adjacency", node_ids=[0])
    assert ok is True and value == [1, 2]


def test_parse_connected_nodes_drops_echoed_source():
    # A model that echoes "connected to 0: 1, 2" must not count the source node.
    value, ok = parse(
        "connected_nodes", "Node 0 is connected to 1 and 2", encoding="adjacency", node_ids=[0]
    )
    assert ok is True and value == [1, 2]


def test_parse_connected_nodes_named_encoding():
    # friendship: labels are names (James=0, John=2). Source is Mary=5.
    value, ok = parse(
        "connected_nodes", "James, John.", encoding="friendship", node_ids=[5]
    )
    assert ok is True and value == [0, 2]


@pytest.mark.parametrize("text", ["No nodes.", "none", "There are no other nodes."])
def test_parse_connected_nodes_empty(text):
    value, ok = parse("connected_nodes", text, encoding="adjacency", node_ids=[0])
    assert ok is True and value == []


def test_parse_connected_nodes_failure():
    value, ok = parse("connected_nodes", "I am unsure", encoding="adjacency", node_ids=[0])
    assert value is None and ok is False


def test_parse_connected_nodes_requires_encoding():
    with pytest.raises(ValueError):
        parse("connected_nodes", "1, 2", node_ids=[0])


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


def test_parse_connected_nodes_keeps_the_source_in_a_bare_list():
    # A bare list is the answer verbatim: naming the queried node asserts a self-
    # connection, which the gold never contains, so it must survive to score wrong.
    # Dropping it silently forgave 10 percent of debate's rows against the baseline's 3.
    value, ok = parse("connected_nodes", "0, 1, 2", encoding="adjacency", node_ids=[0])
    assert ok is True and value == [0, 1, 2]


def test_parse_connected_nodes_still_drops_a_source_echoed_in_prose():
    # Prose names the source because it restates the question, not to claim membership.
    value, ok = parse(
        "connected_nodes", "The nodes connected to 2 are: 6, 10.",
        encoding="adjacency", node_ids=[2],
    )
    assert ok is True and value == [6, 10]
