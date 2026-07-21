import json

import networkx as nx
import pytest

from gedebate.graphqa import graph_text_encoder as enc
from gedebate.data.dataset import ENCODINGS, TASKS, build_dataset
from gedebate.data.instance import normalized_ground_truth


# --- vendored encoder characterization (byte-exact, incl. adjacency preamble) --

def test_adjacency_encoder_exact():
    g = nx.path_graph(4)  # 0-1-2-3
    assert enc.encode_graph(g, "adjacency") == (
        "In an undirected graph, (i,j) means that node i and node j are"
        " connected with an undirected edge. "
        "G describes a graph among nodes 0, 1, 2, and 3.\n"
        "The edges in G are: (0, 1) (1, 2) (2, 3).\n"
    )


def test_friendship_says_among_nodes():
    # Fidelity check: the released code writes "among nodes <names>".
    g = nx.path_graph(3)
    out = enc.encode_graph(g, "friendship")
    assert out.startswith("G describes a friendship graph among nodes James, Robert, and John.")


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
