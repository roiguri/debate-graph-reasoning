"""Characterization tests for the vendored GraphQA encoders (byte-exact).

Pins the wording we depend on -- the adjacency preamble and friendship's
"among nodes <names>" phrasing -- so a future re-vendor can't silently drift.
"""

import networkx as nx

from gedebate.graphqa import graph_text_encoder as enc


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
