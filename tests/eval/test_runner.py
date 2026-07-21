"""Tests for the runner's persist-and-resume path (`run_one_persisted`).

The P2.2 done-when: a persisted run writes one row, and re-running skips it (writes
nothing new). Uses a stub model so no torch is required.
"""

from __future__ import annotations

from dataclasses import dataclass

from gedebate.eval import results
from gedebate.eval.runner import first_instance, run_one_persisted


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
