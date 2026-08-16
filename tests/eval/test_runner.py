"""Tests for the runner: dataset load+filter, sharding, the config-driven batch
loop with resume, the manifest reproduction-record guard, and verify_sample.
Stub model + a tmp dataset artifact, so no torch is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from gedebate.data.dataset import build_dataset
from gedebate.data.store import dump_dataset
from gedebate.eval import results
from gedebate.eval.config import RunConfig
from gedebate.eval.runner import (
    _verify_sample,
    build_instances,
    manifest_record,
    parse_shard,
    run_instances,
    select_shard,
    verify_sample,
)
from gedebate.prompts import build_prompt
from gedebate.prompts.debate import proposer_prompt

MANIFEST = {"dataset_sha256": "testhash"}  # minimal reproduction record for the guard


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
        return _StubGen(self.reply, 7, 11)


def _dataset(tmp_path, n=5):
    p = tmp_path / "ds.jsonl"
    dump_dataset(build_dataset(n_graphs=n, seed=7), p)
    return str(p)


def _cfg(tmp_path, **over):
    base = dict(
        model="stub-model", out_dir=str(tmp_path / "out"), dataset=_dataset(tmp_path),
        condition="baseline", tasks=("edge_existence",), encodings=("adjacency",),
        max_new_tokens=16,
    )
    base.update(over)
    return RunConfig(**base)


# --- load + filter ------------------------------------------------------------

def test_build_instances_loads_and_filters(tmp_path):
    cfg = _cfg(tmp_path)  # edge_existence x adjacency
    insts = build_instances(cfg)
    assert len(insts) == 5  # 5 graphs x 1 task x 1 encoding
    assert {(i.task, i.encoding) for i in insts} == {("edge_existence", "adjacency")}


# --- sharding -----------------------------------------------------------------

def test_parse_shard():
    assert parse_shard("0/1") == (0, 1)
    assert parse_shard("2/4") == (2, 4)


def test_select_shard_partitions_disjointly_and_completely():
    items = list(range(23))
    n = 4
    shards = [select_shard(items, s, n) for s in range(n)]
    assert sorted(x for sh in shards for x in sh) == items
    seen = set()
    for sh in shards:
        assert not (set(sh) & seen)
        seen |= set(sh)


def test_select_shard_out_of_range():
    with pytest.raises(ValueError):
        select_shard([1, 2, 3], 4, 4)


# --- batch loop with resume ---------------------------------------------------

def test_run_instances_persists_and_resumes(tmp_path):
    cfg = _cfg(tmp_path)
    instances = build_instances(cfg)
    assert len(instances) == 5

    model = _StubModel()
    stats = run_instances(model, instances[:3], cfg, MANIFEST)
    assert stats == {"written": 3, "skipped": 0} and model.calls == 3

    model2 = _StubModel()
    stats2 = run_instances(model2, instances, cfg, MANIFEST)
    assert stats2 == {"written": 2, "skipped": 3} and model2.calls == 2

    path = results.shard_file(cfg.out_dir, "baseline")
    assert len(results.read_rows(path)) == 5


def test_run_instances_manifest_guards_model(tmp_path):
    cfg = _cfg(tmp_path)
    run_instances(_StubModel(), build_instances(cfg), cfg, MANIFEST)
    with pytest.raises(ValueError):
        run_instances(_StubModel(), build_instances(cfg), _cfg(tmp_path, model="other"), MANIFEST)


def test_run_instances_manifest_guards_dataset(tmp_path):
    cfg = _cfg(tmp_path)
    run_instances(_StubModel(), build_instances(cfg), cfg, {"dataset_sha256": "h1"})
    with pytest.raises(ValueError):
        run_instances(_StubModel(), build_instances(cfg), cfg, {"dataset_sha256": "h2"})


# --- majority-vote path (N rows per instance, resume tops up missing draws) ---

def test_majority_vote_persists_n_rows_per_instance(tmp_path):
    cfg = _cfg(tmp_path, condition="majority_vote", n_samples=4, temperature=0.7)
    instances = build_instances(cfg)

    model = _StubModel()
    stats = run_instances(model, instances, cfg, MANIFEST)
    # 5 instances x 4 draws each
    assert stats == {"written": 20, "skipped": 0} and model.calls == 20

    rows = results.read_rows(results.shard_file(cfg.out_dir, "majority_vote"))
    assert len(rows) == 20
    first_id = instances[0].instance_id
    got = {r["sample_index"] for r in rows if r["instance_id"] == first_id}
    assert got == {0, 1, 2, 3}
    # sampling metadata persisted per row
    assert all(r["temperature"] == 0.7 and r["seed"] is not None for r in rows)


def test_majority_vote_resume_tops_up_missing_draws(tmp_path):
    # First pass runs only 2 of 4 draws (simulate a kill), then resume completes.
    cfg2 = _cfg(tmp_path, condition="majority_vote", n_samples=2, temperature=0.7)
    insts = build_instances(cfg2)
    run_instances(_StubModel(), insts, cfg2, MANIFEST)  # writes samples 0,1

    cfg4 = _cfg(tmp_path, condition="majority_vote", n_samples=4, temperature=0.7,
                dataset=cfg2.dataset, out_dir=cfg2.out_dir)
    model = _StubModel()
    stats = run_instances(model, insts, cfg4, MANIFEST)  # tops up samples 2,3
    assert stats == {"written": 10, "skipped": 0} and model.calls == 10  # 5 insts x 2

    rows = results.read_rows(results.shard_file(cfg4.out_dir, "majority_vote"))
    assert len(rows) == 20  # now 4 draws x 5 instances
    first_id = insts[0].instance_id
    assert {r["sample_index"] for r in rows if r["instance_id"] == first_id} == {0, 1, 2, 3}

    # fully-done instances are skipped, not rerun
    stats2 = run_instances(_StubModel(), insts, cfg4, MANIFEST)
    assert stats2 == {"written": 0, "skipped": 5}


# --- debate path (1 row + a trace sidecar per instance) -----------------------

class _DebateStub:
    """Prompt-aware stub: a Critic prompt (shows the VERDICT format) -> AGREE; otherwise
    a Proposer answer. So every debate converges in 2 turns (proposer + critic)."""

    def generate(self, prompt, *, max_new_tokens=64, **_):
        if "VERDICT:" in prompt:
            return _StubGen("VERDICT: AGREE", 2, 10)
        return _StubGen("1. some claim\nANSWER: 1", 3, 20)


def test_debate_dispatch_persists_row_and_trace(tmp_path):
    cfg = _cfg(tmp_path, condition="debate", tasks=("node_degree",),
               encodings=("adjacency",), n_samples=10)
    insts = build_instances(cfg)
    assert len(insts) == 5

    stats = run_instances(_DebateStub(), insts, cfg, MANIFEST)
    assert stats == {"written": 5, "skipped": 0}  # one row per instance

    rows = results.read_rows(results.shard_file(cfg.out_dir, "debate"))
    assert len(rows) == 5
    assert rows[0]["condition"] == "debate" and rows[0]["n_responses"] == 2  # proposer + AGREE

    # the verbose transcript lands in the trace sidecar, not the row
    traces = results.read_traces(results.trace_file(cfg.out_dir, "debate"))
    assert len(traces) == 5
    assert [t["role"] for t in traces[0]["turns"]] == ["proposer", "critic"]

    # resume: every instance already has its 1 row -> all skipped
    assert run_instances(_DebateStub(), insts, cfg, MANIFEST) == {"written": 0, "skipped": 5}


# --- verify_sample (reproducibility spot check) -------------------------------

def test_verify_sample_matches_and_detects_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    insts = build_instances(cfg)
    run_instances(_StubModel("Yes."), insts, cfg, MANIFEST)  # persist: all parse to True
    rows = [r for f in results.result_files(cfg.out_dir) for r in results.read_rows(f)]
    by_id = {i.instance_id: i for i in insts}

    same = verify_sample(_StubModel("Yes."), by_id, rows, k=3)
    assert same["checked"] == 3 and same["matches"] == 3 and not same["mismatches"]

    diff = verify_sample(_StubModel("No."), by_id, rows, k=3)  # now parse to False
    assert diff["checked"] == 3 and diff["matches"] == 0 and len(diff["mismatches"]) == 3


# --- provenance: the commit must survive the trip to the cluster ---------------

def test_git_commit_falls_back_to_the_sync_stamp(tmp_path, monkeypatch):
    # Runs happen on an rsync copy with no .git, so `git rev-parse` fails there and every
    # manifest recorded "unknown". The prompts are frozen but not immutable, so the commit
    # is the only thing that says which wording produced a row.
    from gedebate.eval import runner

    stamp = tmp_path / ".git_commit"
    stamp.write_text("deadbeefcafe\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_COMMIT_STAMP", stamp)
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no git here")))
    assert runner._git_commit() == "deadbeefcafe"


def test_git_commit_is_unknown_when_there_is_no_git_and_no_stamp(tmp_path, monkeypatch):
    from gedebate.eval import runner

    monkeypatch.setattr(runner, "_COMMIT_STAMP", tmp_path / "absent")
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no git here")))
    assert runner._git_commit() == "unknown"


# --- the reasoned vote arm (same N-row persistence, Proposer prompt) -----------

def test_majority_vote_cot_persists_n_rows(tmp_path):
    cfg = _cfg(tmp_path, condition="majority_vote_cot", n_samples=3, temperature=0.6,
               max_new_tokens=512)
    instances = build_instances(cfg)

    model = _StubModel("1. The pair is in the edge list.\nANSWER: Yes")
    stats = run_instances(model, instances, cfg, MANIFEST)
    assert stats == {"written": 15, "skipped": 0} and model.calls == 15  # 5 insts x 3

    rows = results.read_rows(results.shard_file(cfg.out_dir, "majority_vote_cot"))
    assert len(rows) == 15
    assert {r["condition"] for r in rows} == {"majority_vote_cot"}
    assert all(r["temperature"] == 0.6 and r["seed"] is not None for r in rows)

    # fully-done instances are skipped on resume, exactly like the terse arm
    assert run_instances(_StubModel(), instances, cfg, MANIFEST) == {"written": 0, "skipped": 5}


def test_majority_vote_cot_sends_the_proposer_prompt(tmp_path):
    cfg = _cfg(tmp_path, condition="majority_vote_cot", n_samples=2, temperature=0.6)
    inst = build_instances(cfg)[0]

    seen = []

    class _Recorder(_StubModel):
        def generate(self, prompt, *, max_new_tokens=64, **kw):
            seen.append(prompt)
            return super().generate(prompt, max_new_tokens=max_new_tokens, **kw)

    run_instances(_Recorder("ANSWER: Yes"), [inst], cfg, MANIFEST)
    assert seen and all(p == proposer_prompt(inst) for p in seen)
    assert build_prompt(inst) not in seen


def test_manifest_records_the_vote_arms_budget(tmp_path):
    terse = manifest_record(_cfg(tmp_path, condition="majority_vote", n_samples=4), "c.toml")
    assert terse["n_samples"] == 4
    assert terse["decoding"].startswith("temperature=")

    cot = manifest_record(
        _cfg(tmp_path, condition="majority_vote_cot", n_samples=3), "c.toml")
    assert cot["n_samples"] == 3
    assert cot["decoding"].startswith("temperature=")


def test_verify_sample_only_reads_its_own_conditions_rows(tmp_path):
    """A run dir holds every condition side by side, and verify_sample re-runs each row
    as a greedy BASELINE answer -- so feeding it debate or vote rows would report a
    different condition's answer as a reproducibility failure."""
    cfg = _cfg(tmp_path, condition="debate", n_samples=4)
    run_instances(_StubModel("1. a\nANSWER: 1"), build_instances(cfg), cfg, MANIFEST)

    # same out_dir, baseline config: the debate rows sitting there must not be verified
    base = _cfg(tmp_path, condition="baseline", dataset=cfg.dataset, out_dir=cfg.out_dir)
    cfg_path = tmp_path / "base.toml"
    cfg_path.write_text(
        "\n".join(f'{k} = {v!r}' if isinstance(v, str) else f"{k} = {list(v) if isinstance(v, tuple) else v}"
                  for k, v in [("model", base.model), ("out_dir", base.out_dir),
                               ("dataset", base.dataset), ("condition", "baseline"),
                               ("tasks", base.tasks), ("encodings", base.encodings)]),
        encoding="utf-8")

    with pytest.raises(SystemExit, match="no persisted baseline rows"):
        _verify_sample(str(cfg_path), 5)
