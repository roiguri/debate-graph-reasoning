"""Tests for the frozen dataset artifact (`gedebate.data.store`) + the refactor's
safety gates: the committed data/main.jsonl reproduces today's generation, and any
existing baseline results are consistent with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gedebate.data.dataset import build_dataset
from gedebate.data.store import dataset_sha256, dump_dataset, load_dataset
from gedebate.eval import results

DATA = Path("data/main.jsonl")
META = Path("data/main.meta.json")


def test_dump_load_roundtrip(tmp_path):
    insts = build_dataset(n_graphs=5, seed=7)
    p = tmp_path / "d.jsonl"
    dump_dataset(insts, p)
    assert load_dataset(p) == insts  # field-by-field (frozen dataclass __eq__)


def test_sha256_is_stable_across_rebuilds():
    assert dataset_sha256(build_dataset(n_graphs=5, seed=7)) == \
        dataset_sha256(build_dataset(n_graphs=5, seed=7))


# --- safety gate 1: committed artifact == today's generation ------------------

@pytest.mark.skipif(not DATA.exists(), reason="dataset artifact not built yet")
def test_committed_dataset_matches_generator_and_meta():
    loaded = load_dataset(DATA)
    meta = json.loads(META.read_text(encoding="utf-8"))
    assert meta["n_instances"] == len(loaded) == 1800
    assert meta["sha256"] == dataset_sha256(loaded)


@pytest.mark.skipif(not META.exists(), reason="dataset artifact not built yet")
def test_meta_spec_reproduces_recorded_hash():
    # Reproducibility is anchored to meta['spec'] (what build_dataset.py --verify does):
    # rebuilding from the recorded spec must reproduce the recorded hash.
    meta = json.loads(META.read_text(encoding="utf-8"))
    spec = meta["spec"]
    insts = build_dataset(
        n_graphs=spec["n_graphs"], seed=spec["dataset_seed"], algorithm=spec["algorithm"],
        tasks=tuple(spec["tasks"]), encodings=tuple(spec["encodings"]),
    )
    assert dataset_sha256(insts) == meta["sha256"]
    assert len(insts) == meta["n_instances"]


# --- safety gate 2: existing baseline results are consistent with the dataset --

@pytest.mark.skipif(
    not (DATA.exists() and Path("results/main").exists()),
    reason="no local baseline results to cross-check",
)
def test_existing_baseline_consistent_with_dataset():
    ds = {i.instance_id: i for i in load_dataset(DATA)}
    rows = [r for f in results.result_files("results/main") for r in results.read_rows(f)]
    assert rows, "expected existing baseline rows"
    for r in rows:
        inst = ds.get(r["instance_id"])
        assert inst is not None, f"{r['instance_id']} not in dataset"
        assert r["ground_truth"] == inst.ground_truth
        assert (r["task"], r["encoding"]) == (inst.task, inst.encoding)
