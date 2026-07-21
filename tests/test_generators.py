import networkx as nx

from gedebate.data.generators import GENERATORS, erdos_renyi


def test_registry_lookup():
    assert "er" in GENERATORS.names()
    assert GENERATORS.get("er") is erdos_renyi


def test_registry_unknown_raises():
    try:
        GENERATORS.get("nope")
    except KeyError as e:
        assert "unknown generator" in str(e)
    else:
        raise AssertionError("expected KeyError for unknown generator")


def test_er_node_count():
    g = erdos_renyi(9, seed=0, p=0.3)
    assert g.number_of_nodes() == 9
    assert set(g.nodes) == set(range(9))  # integer-labeled 0..n-1


def test_er_determinism():
    a = erdos_renyi(12, seed=42, p=0.4)
    b = erdos_renyi(12, seed=42, p=0.4)
    assert sorted(a.edges) == sorted(b.edges)


def test_er_seed_changes_graph():
    a = erdos_renyi(20, seed=1, p=0.3)
    b = erdos_renyi(20, seed=2, p=0.3)
    # Overwhelmingly likely to differ for n=20; guards against ignored seed.
    assert sorted(a.edges) != sorted(b.edges)


def test_er_p_bounds():
    assert erdos_renyi(10, seed=0, p=0.0).number_of_edges() == 0
    complete = erdos_renyi(10, seed=0, p=1.0)
    assert complete.number_of_edges() == 10 * 9 // 2


def test_er_undirected():
    assert not erdos_renyi(5, seed=0, p=0.5).is_directed()


def test_er_invalid_args():
    for bad in (dict(n=0, seed=0), dict(n=5, seed=0, p=1.5)):
        try:
            erdos_renyi(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
