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
    {"model": "m", "out_dir": "o", "dataset": "d", "condition": "quux"},  # unknown condition
    {"model": "m", "out_dir": "o", "dataset": "d", "n_samples": 0},      # must be >= 1
    {"model": "m", "out_dir": "o", "dataset": "d", "temperature": 0},    # must be > 0
    {"model": "m", "out_dir": "o", "dataset": "d", "prompt_version": "v99"},  # unknown version
    {"model": "m", "out_dir": "o", "dataset": "d", "top_p": 0},          # must be in (0,1]
    {"model": "m", "out_dir": "o", "dataset": "d", "top_p": 1.5},        # must be <= 1
    {"model": "m", "out_dir": "o", "dataset": "d", "top_k": -1},         # must be >= 0
])
def test_from_dict_rejects_bad_config(data):
    with pytest.raises(ValueError):
        RunConfig.from_dict(data)


def test_majority_vote_config_fields():
    cfg = RunConfig.from_dict({
        "model": "m", "out_dir": "o", "dataset": "data/main.jsonl",
        "condition": "majority_vote", "n_samples": 10, "temperature": 0.7,
        "top_p": 1.0, "top_k": 0,
    })
    assert cfg.condition == "majority_vote"
    assert cfg.n_samples == 10 and cfg.temperature == 0.7
    assert cfg.top_p == 1.0 and cfg.top_k == 0


def test_sampling_defaults_are_explicit():
    # Omitting top_p/top_k yields explicit defaults (not None), so the model never
    # falls back to its shipped generation_config sampling params.
    cfg = RunConfig.from_dict({"model": "m", "out_dir": "o", "dataset": "d"})
    assert cfg.top_p == 0.8 and cfg.top_k == 20 and cfg.n_samples == 10


def test_load_config_from_file(tmp_path):
    p = tmp_path / "run.toml"
    p.write_text(
        'model = "m"\nout_dir = "o"\ndataset = "data/main.jsonl"\n'
        'tasks = ["edge_existence"]\nencodings = ["adjacency"]\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.dataset == "data/main.jsonl" and cfg.tasks == ("edge_existence",)


def test_repo_mv_config():
    # The majority-vote run config: N=10 sampled draws over the frozen dataset.
    cfg = load_config("configs/mv.toml")
    assert cfg.condition == "majority_vote"
    assert cfg.n_samples == 10 and cfg.temperature == 0.7
    assert cfg.dataset == "data/main.jsonl" and cfg.out_dir == "results/main"
    assert cfg.max_new_tokens == 128


def test_repo_p3_matrix_config():
    # The run config: full matrix over the frozen dataset artifact.
    cfg = load_config("configs/matrix.toml")
    assert set(cfg.tasks) == {"edge_existence", "node_degree", "connected_nodes"}
    assert set(cfg.encodings) == {"adjacency", "incident", "friendship"}
    assert cfg.dataset == "data/main.jsonl"
    assert cfg.out_dir == "results/main"
    assert "Qwen" in cfg.model


def test_prompt_version_defaults_to_the_single_frozen_version():
    # v1 was deleted when the project consolidated on one prompt; a config that names no
    # version gets v2, the only one that exists.
    cfg = RunConfig.from_dict({"model": "m", "out_dir": "o", "dataset": "d"})
    assert cfg.prompt_version == "v2"


def test_prompt_version_is_opt_in():
    cfg = RunConfig.from_dict({"model": "m", "out_dir": "o", "dataset": "d",
                               "condition": "debate", "prompt_version": "v2"})
    assert cfg.prompt_version == "v2"
