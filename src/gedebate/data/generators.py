"""Graph generators. Each returns a plain undirected `nx.Graph` given a size and
a seed (plus generator-specific params). Registered by name.

GraphQA [Fatemi et al. 2024] uses Erdős–Rényi graphs (following Wang et al.
2023); that's our default. The registry leaves room for BA / star / path later.
"""

from __future__ import annotations

import networkx as nx

from .registry import Registry

GENERATORS: Registry = Registry("generator")


@GENERATORS.register("er")
def erdos_renyi(n: int, seed: int, *, p: float = 0.3) -> nx.Graph:
    """Erdős–Rényi G(n, p): each of the n*(n-1)/2 possible edges included w.p. p.

    Deterministic given (n, seed, p). Relabels nothing -- nodes are ints 0..n-1,
    which the integer encodings (adjacency/incident) rely on.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    return nx.gnp_random_graph(n, p, seed=seed)
