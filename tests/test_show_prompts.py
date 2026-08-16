"""Tests for scripts/show_prompts.py: that what it prints is what the run would send.

The point of the script is that there is no second copy of the prompt text in it, so the
tests assert exactly that -- every printed prompt is compared against the builder in
`gedebate.prompts` that the conditions call, never against a pasted string.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gedebate.data.dataset import build_dataset
from gedebate.data.store import dump_dataset
from gedebate.eval import results
from gedebate.eval.config import load_config
from gedebate.prompts import build_prompt
from gedebate.prompts.debate import critic_prompt, proposer_prompt, revision_prompt

REPO = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "show_prompts", REPO / "scripts" / "show_prompts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config(tmp_path, condition="debate", **extra) -> Path:
    """A run config over a small frozen dataset written under tmp_path."""
    dataset = tmp_path / "ds.jsonl"
    dump_dataset(build_dataset(n_graphs=2, seed=7), dataset)
    body = {"model": "m", "condition": condition, "dataset": str(dataset),
            "out_dir": str(tmp_path / "run"), "tasks": ["node_degree"],
            "encodings": ["adjacency", "friendship"], **extra}
    lines = []
    for k, v in body.items():
        lines.append(f"{k} = {v!r}" if isinstance(v, str) else f"{k} = {v}")
    path = tmp_path / "cfg.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_prints_one_cell_per_config_matrix_entry(tmp_path):
    mod = _mod()
    cfg = load_config(_config(tmp_path))
    instances = mod.select_instances(cfg)
    assert [(i.task, i.encoding) for i in instances] == [
        ("node_degree", "adjacency"), ("node_degree", "friendship")]


def test_debate_prompts_are_the_builders_output(tmp_path):
    """The three debate prompts, verbatim from prompts.debate -- the anti-drift check."""
    mod = _mod()
    cfg = load_config(_config(tmp_path))
    inst = mod.select_instances(cfg, encoding="adjacency")[0]
    turns = mod.placeholder_turns(1)

    blocks = mod.debate_prompts(inst, turns)
    labels = [lbl for lbl, _ in blocks]
    assert labels == ["PROPOSER (turn 1)", "CRITIC (turn 2)", "PROPOSER REVISION (turn 3)"]
    assert [p for _, p in blocks] == [
        proposer_prompt(inst),
        critic_prompt(inst, turns[:1]),      # transcript before the turn
        revision_prompt(inst, turns[:2]),
    ]

    text = mod.render(cfg, "cfg.toml", [inst])
    for _, prompt in blocks:
        assert prompt in text  # printed whole, not summarized


def test_from_run_replays_the_recorded_transcript(tmp_path):
    """With --from-run the transcript is the real one, so no placeholder text survives."""
    mod = _mod()
    cfg = load_config(_config(tmp_path))
    inst = mod.select_instances(cfg, encoding="adjacency")[0]
    turns = [{"role": "proposer", "raw": "1. real claim\nANSWER: 2"},
             {"role": "critic", "raw": "VERDICT: REVISE\n- edge (0, 1)"},
             {"role": "proposer", "raw": "1. fixed claim\nANSWER: 1"}]
    run = Path(cfg.out_dir)
    results.append_trace(results.trace_file(run, "debate"), inst.instance_id, turns)

    assert mod.load_trace_turns(run, inst.instance_id) == turns
    text = mod.render(cfg, "cfg.toml", [inst], from_run=str(run))
    assert critic_prompt(inst, turns[:1]) in text
    assert revision_prompt(inst, turns[:2]) in text
    assert "PLACEHOLDER" not in text


def test_from_run_falls_back_when_the_instance_has_no_trace(tmp_path):
    mod = _mod()
    cfg = load_config(_config(tmp_path))
    inst = mod.select_instances(cfg, encoding="adjacency")[0]
    text = mod.render(cfg, "cfg.toml", [inst], from_run=str(tmp_path / "empty"))
    assert "no trace" in text and "PLACEHOLDER" in text


def test_extra_rounds_grow_the_transcript(tmp_path):
    mod = _mod()
    cfg = load_config(_config(tmp_path))
    inst = mod.select_instances(cfg, encoding="adjacency")[0]
    one = mod.render(cfg, "cfg.toml", [inst], rounds=1)
    two = mod.render(cfg, "cfg.toml", [inst], rounds=2)
    assert "CRITIC (turn 4)" in two and "CRITIC (turn 4)" not in one
    assert len(two) > len(one)


def test_baseline_config_prints_the_baseline_prompt(tmp_path):
    mod = _mod()
    cfg = load_config(_config(tmp_path, condition="baseline"))
    inst = mod.select_instances(cfg, encoding="friendship")[0]
    text = mod.render(cfg, "cfg.toml", [inst])
    assert build_prompt(inst) in text
    assert "PROPOSER" not in text


def test_cli_writes_a_file(tmp_path):
    mod = _mod()
    out = tmp_path / "prompts.txt"
    mod.main([str(_config(tmp_path)), "--task", "node_degree", "--out", str(out)])
    assert "PROPOSER (turn 1)" in out.read_text(encoding="utf-8")


def test_unknown_instance_id_is_an_error(tmp_path):
    mod = _mod()
    cfg = load_config(_config(tmp_path))
    with pytest.raises(SystemExit):
        mod.select_instances(cfg, instance_id="nope")
