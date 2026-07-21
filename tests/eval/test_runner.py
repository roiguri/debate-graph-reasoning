"""Tests for the runner: single-instance persist/skip (P2.2) and the config-driven,
shardable batch loop with resume (P2.3). A stub model keeps it torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from gedebate.eval import results
from gedebate.eval.config import RunConfig
from gedebate.eval.runner import (
    build_instances,
    first_instance,
    parse_shard,
    run_instances,
    run_one_persisted,
    select_shard,
    summarize_run,
)


@dataclass
class _StubGen:
    text: str
    n_gen_tokens: int
    n_prompt_tokens: int


class _StubModel:
    def __init__(self, reply="Yes."):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, *, max_new_tokens=64, **_):
        self.calls += 1
        return _StubGen(self.reply, n_gen_tokens=7, n_prompt_tokens=11)


def _inst():
    return first_instance(n_graphs=4, seed=7, task="edge_existence", encoding="adjacency")


def test_persist_then_skip_on_rerun(tmp_path):
    inst = _inst()
    model = _StubModel()

    row = run_one_persisted(model, inst, str(tmp_path), "stub-model")
    assert row is not None
    assert row["instance_id"] == inst.instance_id
    assert model.calls == 1

    # Manifest written; exactly one persisted row.
    assert results.read_manifest(tmp_path)["model"] == "stub-model"
    path = results.shard_file(tmp_path, "baseline")
    assert len(results.read_rows(path)) == 1

    # Re-run: instance already done -> skipped, model not called again, no new row.
    skipped = run_one_persisted(model, inst, str(tmp_path), "stub-model")
    assert skipped is None
    assert model.calls == 1
    assert len(results.read_rows(path)) == 1


# --- sharding -----------------------------------------------------------------

def test_parse_shard():
    assert parse_shard("0/1") == (0, 1)
    assert parse_shard("2/4") == (2, 4)


def test_select_shard_partitions_disjointly_and_completely():
    items = list(range(23))
    n = 4
    shards = [select_shard(items, s, n) for s in range(n)]
    # union == all, and pairwise disjoint
    assert sorted(x for sh in shards for x in sh) == items
    seen = set()
    for sh in shards:
        assert not (set(sh) & seen)
        seen |= set(sh)


def test_select_shard_out_of_range():
    with pytest.raises(ValueError):
        select_shard([1, 2, 3], 4, 4)


# --- config-driven batch loop with resume -------------------------------------

def _cfg(tmp_path, **over):
    base = dict(
        model="stub-model", out_dir=str(tmp_path), condition="baseline",
        tasks=("edge_existence",), encodings=("adjacency",),
        n_graphs=5, dataset_seed=7, max_new_tokens=16,
    )
    base.update(over)
    return RunConfig(**base)


def test_run_instances_persists_and_resumes(tmp_path):
    cfg = _cfg(tmp_path)
    instances = build_instances(cfg)
    assert len(instances) == 5  # 5 graphs x 1 task x 1 encoding

    # First pass over the first 3 instances (simulating a kill after 3).
    model = _StubModel()
    stats = run_instances(model, instances[:3], cfg)
    assert stats == {"written": 3, "skipped": 0}
    assert model.calls == 3

    # Resume over ALL 5: the first 3 are skipped, only 2 new generations happen.
    model2 = _StubModel()
    stats2 = run_instances(model2, instances, cfg)
    assert stats2 == {"written": 2, "skipped": 3}
    assert model2.calls == 2

    # One row per instance, and the summary covers all 5.
    path = results.shard_file(cfg.out_dir, "baseline")
    assert len(results.read_rows(path)) == 5
    summary = summarize_run(cfg)
    assert summary[("edge_existence", "adjacency")]["n"] == 5


def test_run_instances_manifest_guards_model(tmp_path):
    cfg = _cfg(tmp_path)
    run_instances(_StubModel(), build_instances(cfg), cfg)
    # Same out_dir, different model -> refused.
    with pytest.raises(ValueError):
        run_instances(_StubModel(), build_instances(cfg), _cfg(tmp_path, model="other"))
