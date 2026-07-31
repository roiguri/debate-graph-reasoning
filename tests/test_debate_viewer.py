"""Tests for scripts/debate_viewer.py: the data layer (rows+traces join, graph payload)
and that the page loads transcripts dynamically via the API (nothing embedded up front)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from gedebate.data.dataset import build_dataset
from gedebate.eval import results

REPO = Path(__file__).resolve().parents[1]


def _viewer():
    spec = importlib.util.spec_from_file_location("debate_viewer", REPO / "scripts" / "debate_viewer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record():
    return {"condition": "debate", "raw_output": "1. edge (0,3)\nANSWER: 1",
            "parsed_answer": 1, "parse_ok": True, "correct": True, "ground_truth": 1,
            "n_prompt_tokens": 200, "n_gen_tokens": 30}


def test_load_joins_rows_and_traces(tmp_path):
    run = tmp_path / "run"
    inst = next(i for i in build_dataset(n_graphs=4, seed=7)
                if i.task == "node_degree" and i.encoding == "adjacency")
    results.append_row(results.shard_file(run, "debate"),
                       results.make_row(inst, "m", _record(), n_responses=3))
    turns = [{"role": "proposer", "raw": "1. edge\nANSWER: 2"},
             {"role": "critic", "raw": "VERDICT: REVISE", "verdict": "REVISE"},
             {"role": "proposer", "raw": "1. edge\nANSWER: 1"}]
    results.append_trace(results.trace_file(run, "debate"), inst.instance_id, turns)

    mod = _viewer()
    rows, turns_by_id, instances, index = mod.load(str(run))
    assert index == [{"instance_id": inst.instance_id, "task": "node_degree",
                      "encoding": "adjacency", "correct": True, "n_responses": 3}]
    assert turns_by_id[inst.instance_id] == turns          # fetched on demand, not in index
    assert rows[inst.instance_id]["parsed_answer"] == 1
    assert instances == {}                                 # no manifest in tmp run -> no graph data


def test_graph_payload_has_nodes_edges_positions_query():
    mod = _viewer()
    inst = next(i for i in build_dataset(n_graphs=4, seed=7)
                if i.task == "node_degree" and i.encoding == "adjacency")
    g = mod._graph_payload(inst)
    assert g["nodes"] == list(range(inst.nnodes))
    assert g["edges"] == [list(e) for e in inst.graph_edgelist]
    assert g["query"] == inst.node_ids[0]
    assert set(g["positions"]) == {str(n) for n in range(inst.nnodes)}   # seeded layout, one per node
    assert g["encoding_text"] == inst.question
    assert mod._graph_payload(None) is None


def test_graph_payload_labels_nodes_the_way_the_encoding_does():
    """The drawing must use the same node names the model read, per encoding."""
    mod = _viewer()
    ds = [i for i in build_dataset(n_graphs=4, seed=7) if i.task == "node_degree"]
    adj = next(i for i in ds if i.encoding == "adjacency")
    fri = next(i for i in ds if i.encoding == "friendship")
    assert mod._graph_payload(adj)["labels"] == {str(n): str(n) for n in range(adj.nnodes)}
    labels = mod._graph_payload(fri)["labels"]
    assert set(labels) == {str(n) for n in range(fri.nnodes)}      # one per node
    assert all(name in fri.question for name in labels.values())   # names the run used
    assert labels["0"] == "James"


def test_page_loads_via_api_and_draws_graph():
    mod = _viewer()
    assert "/api/index" in mod._PAGE and "/api/trace?id=" in mod._PAGE  # dynamic fetch
    assert "__DATA__" not in mod._PAGE                                  # nothing embedded up front
    assert "cytoscape" in mod._PAGE                                     # graph rendering
    assert 'data-theme=dark' in mod._PAGE and 'data-theme=light' in mod._PAGE  # both themes
