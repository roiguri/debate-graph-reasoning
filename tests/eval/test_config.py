"""Tests for `gedebate.eval.config` -- TOML run config + validation."""

from __future__ import annotations

import pytest

from gedebate.eval.config import RunConfig, load_config


def test_from_dict_minimal_uses_defaults():
    cfg = RunConfig.from_dict({"model": "m", "out_dir": "o", "dataset": "data/main.jsonl"})
    assert cfg.model == "m" and cfg.out_dir == "o" and cfg.dataset == "data/main.jsonl"
    assert cfg.condition == "baseline" and cfg.max_new_tokens == 64
    assert set(cfg.tasks) == {"edge_existence", "node_degree", "connected_nodes"}
    assert set(cfg.encodings) == {"adjacency", "incident", "friendship"}


def test_from_dict_full():
    cfg = RunConfig.from_dict({
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "out_dir": "results/main",
        "dataset": "data/main.jsonl",
        "tasks": ["edge_existence"],
        "encodings": ["adjacency"],
        "max_new_tokens": 32,
    })
    assert cfg.tasks == ("edge_existence",) and cfg.encodings == ("adjacency",)
    assert cfg.max_new_tokens == 32


@pytest.mark.parametrize("data", [
    {"out_dir": "o", "dataset": "d"},                       # missing model
    {"model": "m", "dataset": "d"},                         # missing out_dir
    {"model": "m", "out_dir": "o"},                         # missing dataset
    {"model": "m", "out_dir": "o", "dataset": "d", "bogus": 1},          # unknown key
    {"model": "m", "out_dir": "o", "dataset": "d", "tasks": ["nope"]},   # unknown task
    {"model": "m", "out_dir": "o", "dataset": "d", "encodings": ["z"]},  # unknown encoding
    {"model": "m", "out_dir": "o", "dataset": "d", "condition": "debate"},  # not yet known
])
def test_from_dict_rejects_bad_config(data):
    with pytest.raises(ValueError):
        RunConfig.from_dict(data)


def test_load_config_from_file(tmp_path):
    p = tmp_path / "run.toml"
    p.write_text(
        'model = "m"\nout_dir = "o"\ndataset = "data/main.jsonl"\n'
        'tasks = ["edge_existence"]\nencodings = ["adjacency"]\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.dataset == "data/main.jsonl" and cfg.tasks == ("edge_existence",)


def test_repo_p3_matrix_config():
    # The run config: full matrix over the frozen dataset artifact.
    cfg = load_config("configs/p3-matrix.toml")
    assert set(cfg.tasks) == {"edge_existence", "node_degree", "connected_nodes"}
    assert set(cfg.encodings) == {"adjacency", "incident", "friendship"}
    assert cfg.dataset == "data/main.jsonl"
    assert cfg.out_dir == "results/main"
    assert "Qwen" in cfg.model
