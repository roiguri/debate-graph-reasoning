"""Tests for `gedebate.eval.config` -- TOML run config + validation."""

from __future__ import annotations

import pytest

from gedebate.eval.config import RunConfig, load_config


def test_from_dict_minimal_uses_defaults():
    cfg = RunConfig.from_dict({"model": "m", "out_dir": "o"})
    assert cfg.model == "m" and cfg.out_dir == "o"
    assert cfg.condition == "baseline"
    assert cfg.n_graphs == 20 and cfg.dataset_seed == 7 and cfg.max_new_tokens == 64
    assert set(cfg.tasks) == {"edge_existence", "node_degree", "connected_nodes"}
    assert set(cfg.encodings) == {"adjacency", "incident", "friendship"}


def test_from_dict_full():
    cfg = RunConfig.from_dict({
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "out_dir": "results/x",
        "tasks": ["edge_existence"],
        "encodings": ["adjacency"],
        "n_graphs": 5,
        "dataset_seed": 99,
        "max_new_tokens": 32,
    })
    assert cfg.tasks == ("edge_existence",)
    assert cfg.encodings == ("adjacency",)
    assert (cfg.n_graphs, cfg.dataset_seed, cfg.max_new_tokens) == (5, 99, 32)


@pytest.mark.parametrize("data", [
    {"out_dir": "o"},                                   # missing model
    {"model": "m"},                                     # missing out_dir
    {"model": "m", "out_dir": "o", "bogus": 1},         # unknown key
    {"model": "m", "out_dir": "o", "tasks": ["nope"]},  # unknown task
    {"model": "m", "out_dir": "o", "encodings": ["z"]}, # unknown encoding
    {"model": "m", "out_dir": "o", "condition": "debate"},  # not yet known
])
def test_from_dict_rejects_bad_config(data):
    with pytest.raises(ValueError):
        RunConfig.from_dict(data)


def test_load_config_from_file(tmp_path):
    p = tmp_path / "run.toml"
    p.write_text(
        'model = "m"\nout_dir = "o"\ntasks = ["edge_existence"]\n'
        'encodings = ["adjacency"]\nn_graphs = 3\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.model == "m" and cfg.tasks == ("edge_existence",) and cfg.n_graphs == 3


def test_repo_baseline_config_is_valid():
    # The committed config must load and pin the 3B model as the single source.
    cfg = load_config("configs/baseline.toml")
    assert cfg.condition == "baseline"
    assert cfg.tasks == ("edge_existence",) and cfg.encodings == ("adjacency",)
    assert "Qwen" in cfg.model


def test_repo_pilot_matrix_config_covers_full_matrix():
    cfg = load_config("configs/pilot-matrix.toml")
    assert set(cfg.tasks) == {"edge_existence", "node_degree", "connected_nodes"}
    assert set(cfg.encodings) == {"adjacency", "incident", "friendship"}


def test_repo_p3_matrix_config():
    # The P3 run config: full matrix, N/seed fixed (generator not N-extensible).
    cfg = load_config("configs/p3-matrix.toml")
    assert set(cfg.tasks) == {"edge_existence", "node_degree", "connected_nodes"}
    assert set(cfg.encodings) == {"adjacency", "incident", "friendship"}
    assert cfg.n_graphs == 200 and cfg.dataset_seed == 7
    assert cfg.out_dir == "results/main"  # shared experiment dir; condition in a subfolder
