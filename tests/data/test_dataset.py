"""Tests for the data adapter: `build_dataset` + normalized ground truth.

Covers the same-query-across-encodings invariant, the 3x3 scope gate, agreement
between our normalized ground truth and GraphQA's answer string, and
determinism / JSON round-trip. Encoder wording is pinned separately in
tests/graphqa/test_graph_text_encoder.py.
"""

import json

from gedebate.data.dataset import ENCODINGS, TASKS, build_dataset
from gedebate.data.instance import normalized_ground_truth

import networkx as nx


# --- the same-query-across-encodings invariant --------------------------------

def test_query_fixed_across_encodings():
    insts = build_dataset(n_graphs=4, seed=1234)
    by_key = {}
    for inst in insts:
        gkey = (inst.task, tuple(tuple(e) for e in inst.graph_edgelist))
        by_key.setdefault(gkey, set()).add(tuple(inst.node_ids))
    # For each (graph, task) the queried node(s) must be identical across encodings.
    for key, node_id_sets in by_key.items():
        assert len(node_id_sets) == 1, f"query varied across encodings for {key}"


def test_scope_is_three_by_three():
    insts = build_dataset(n_graphs=2, seed=7)
    assert {i.task for i in insts} == set(TASKS)
    assert {i.encoding for i in insts} == set(ENCODINGS)
    assert len(insts) == 2 * len(TASKS) * len(ENCODINGS)


# --- normalized ground truth --------------------------------------------------

def test_normalized_ground_truth_values():
    g = nx.path_graph(4)  # degrees: 0->1,1->2,2->2,3->1; neighbors(1)=[0,2]
    assert normalized_ground_truth("edge_existence", g, [0, 1]) is True
    assert normalized_ground_truth("edge_existence", g, [0, 3]) is False
    assert normalized_ground_truth("node_degree", g, [2]) == 2
    assert normalized_ground_truth("connected_nodes", g, [1]) == [0, 2]
    assert normalized_ground_truth("connected_nodes", g, [0]) == [1]


def test_ground_truth_agrees_with_graphqa_answer():
    # Our normalized GT must be consistent with GraphQA's formatted answer string.
    for inst in build_dataset(n_graphs=6, seed=99):
        if inst.task == "edge_existence":
            assert inst.ground_truth == (inst.answer.strip() == "Yes.")
        elif inst.task == "node_degree":
            assert inst.ground_truth == int(inst.answer.strip().rstrip("."))
        elif inst.task == "connected_nodes":
            n_gt = len(inst.ground_truth)
            n_ans = 0 if "No nodes" in inst.answer else inst.answer.count(",") + 1
            assert n_gt == n_ans


# --- determinism + serialization ----------------------------------------------

def test_determinism():
    a = [i.to_dict() for i in build_dataset(n_graphs=5, seed=2024)]
    b = [i.to_dict() for i in build_dataset(n_graphs=5, seed=2024)]
    assert a == b


def test_json_roundtrip():
    inst = build_dataset(n_graphs=1, seed=1)[0]
    restored = json.loads(json.dumps(inst.to_dict()))
    assert restored["task"] in TASKS
    assert isinstance(restored["question"], str)
    assert restored["instance_id"] == inst.instance_id  # convenience field present


# --- instance_id (the resume key) ---------------------------------------------

def test_instance_id_format_deterministic_and_unique():
    insts = build_dataset(n_graphs=3, seed=7)

    i0 = insts[0]
    assert i0.instance_id == f"{i0.dataset_seed}/{i0.graph_index}/{i0.task}/{i0.encoding}"
    assert i0.dataset_seed == 7

    ids = [i.instance_id for i in insts]
    assert len(ids) == len(set(ids))  # unique across the whole dataset

    # deterministic: the same seed rebuilds the same ids.
    assert set(ids) == {i.instance_id for i in build_dataset(n_graphs=3, seed=7)}


def test_instance_id_shares_graph_index_across_encodings():
    # One (graph, task) differs only by the encoding suffix -> same graph_index.
    insts = [i for i in build_dataset(n_graphs=2, seed=7)
             if i.task == "edge_existence" and i.graph_index == 0]
    assert {i.encoding for i in insts} == set(ENCODINGS)
    assert {i.instance_id.rsplit("/", 1)[0] for i in insts} == {"7/0/edge_existence"}
