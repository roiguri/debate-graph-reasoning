"""The instance record our framework consumes, plus normalized ground truth.

An Instance packages one GraphQA example (graph + encoding + question + answer)
into a JSON-serializable record, and adds a *normalized* ground truth (bool / int
/ sorted list) so the scorer can compare structurally rather than string-match
GraphQA's formatted answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Union

import networkx as nx

GroundTruth = Union[bool, int, list]


@dataclass(frozen=True)
class Instance:
    task: str
    encoding: str
    algorithm: str
    question: str  # encoded graph + task question, exactly as the model sees it
    answer: str  # GraphQA's formatted answer string (e.g. "Yes.", "3.", "1, 2.")
    ground_truth: GroundTruth  # normalized: bool | int | sorted list[int]
    node_ids: list  # the queried node(s); shared across encodings of one graph+task
    nnodes: int
    nedges: int
    graph_edgelist: list  # sorted [ [u, v], ... ] so the graph is reproducible
    query_seed: int  # per-(graph, task) seed that fixed the query across encodings
    dataset_seed: int  # the master seed the whole dataset was built from
    graph_index: int  # position of this graph within that dataset (0-based)

    @property
    def instance_id(self) -> str:
        """Deterministic resume key: unique per (dataset, graph, task, encoding).

        Stable across runs because the dataset is deterministic in `dataset_seed`.
        This is the key the results layer counts attempts against.
        """
        return f"{self.dataset_seed}/{self.graph_index}/{self.task}/{self.encoding}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["instance_id"] = self.instance_id  # convenience; not a stored field
        return d


def normalized_ground_truth(task: str, graph: nx.Graph, node_ids: list) -> GroundTruth:
    """Structured ground truth from the graph, independent of GraphQA's answer text."""
    if task == "edge_existence":
        u, v = node_ids
        return bool(graph.has_edge(u, v))
    if task == "node_degree":
        (n,) = node_ids
        return int(graph.degree[n])
    if task == "connected_nodes":
        (n,) = node_ids
        return sorted(int(x) for x in graph.neighbors(n))
    raise ValueError(f"unknown task '{task}'")
