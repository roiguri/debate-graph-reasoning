"""Thin adapter over vendored GraphQA: generate graphs and build instances.

Scope is deliberately narrower than GraphQA's: ER graphs, three encodings, three
tasks (the study's controls). The vendored code supports more; this layer gates
to what we use.

Key invariant: for a given (graph, task), the queried node(s) are FIXED across
all encodings, so a per-encoding accuracy difference reflects the encoding alone
— not a different question. We achieve this by resetting the global RNG that
GraphQA's task classes sample from to a deterministic per-(graph, task) seed
before each encoding call.
"""

from __future__ import annotations

import random

import networkx as nx

from gedebate.graphqa import graph_generator_utils
from gedebate.graphqa.graph_task import ConnectedNodes, EdgeExistence, NodeDegree
from gedebate.data.instance import Instance, normalized_ground_truth

ENCODINGS = ("adjacency", "incident", "friendship")

TASKS = {
    "edge_existence": EdgeExistence,
    "node_degree": NodeDegree,
    "connected_nodes": ConnectedNodes,
}
_TASK_INDEX = {name: i for i, name in enumerate(TASKS)}


def generate(algorithm: str, n_graphs: int, seed: int) -> list[nx.Graph]:
    """Generate `n_graphs` graphs (default use `algorithm="er"`)."""
    return graph_generator_utils.generate_graphs(
        number_of_graphs=n_graphs,
        algorithm=algorithm,
        directed=False,
        random_seed=seed,
    )


def _query_seed(seed: int, graph_index: int, task: str) -> int:
    """Stable seed per (master seed, graph, task) — same for all encodings."""
    return (seed * 100003 + graph_index * 97 + _TASK_INDEX[task]) % (2**31)


def _edgelist(graph: nx.Graph) -> list:
    return [list(e) for e in sorted(tuple(sorted(e)) for e in graph.edges())]


def build_dataset(
    n_graphs: int,
    seed: int,
    *,
    algorithm: str = "er",
    tasks: tuple = tuple(TASKS),
    encodings: tuple = ENCODINGS,
) -> list[Instance]:
    """Build instances for `tasks` x `encodings` over `n_graphs` ER graphs.

    Deterministic in `seed`. Within a (graph, task), the query is identical across
    encodings; across (graph, task) pairs it varies.
    """
    graphs = generate(algorithm, n_graphs, seed)
    instances: list[Instance] = []
    for gi, graph in enumerate(graphs):
        for task in tasks:
            task_cls = TASKS[task]
            qseed = _query_seed(seed, gi, task)
            for encoding in encodings:
                random.seed(qseed)  # fix the query across encodings
                ex = task_cls().prepare_examples_dict([graph], [algorithm], encoding)[0]
                node_ids = list(ex["node_ids"])
                instances.append(
                    Instance(
                        task=task,
                        encoding=encoding,
                        algorithm=algorithm,
                        question=ex["question"],
                        answer=ex["answer"],
                        ground_truth=normalized_ground_truth(task, graph, node_ids),
                        node_ids=node_ids,
                        nnodes=int(ex["nnodes"]),
                        nedges=int(ex["nedges"]),
                        graph_edgelist=_edgelist(graph),
                        query_seed=qseed,
                        dataset_seed=seed,
                        graph_index=gi,
                    )
                )
    return instances
