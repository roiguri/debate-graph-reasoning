"""Tests for the persistence contract (`gedebate.eval.results`).

Exercises the schema row, atomic append + tolerant read (incl. a torn trailing
line and mid-file corruption), resume-by-counting across shard files, and the
manifest model guard. No torch, no model -- pure I/O on tmp dirs.
"""

from __future__ import annotations

import json

import pytest

from gedebate.data.dataset import build_dataset
from gedebate.eval import results


def _edge_instances(n=2, seed=7):
    return [i for i in build_dataset(n_graphs=n, seed=seed)
            if i.task == "edge_existence" and i.encoding == "adjacency"]


def _attempt(condition="baseline", correct=True, gt=True):
    return {
        "condition": condition,
        "raw_output": "Yes.",
        "parsed_answer": True,
        "parse_ok": True,
        "correct": correct,
        "ground_truth": gt,
        "n_prompt_tokens": 11,
        "n_gen_tokens": 7,
        "prompt": "should NOT be persisted",
    }


# --- make_row -----------------------------------------------------------------

def test_make_row_has_full_schema_and_drops_prompt():
    inst = _edge_instances()[0]
    row = results.make_row(inst, "some-model", _attempt())
    assert set(row) == set(results.ROW_FIELDS)
    assert "prompt" not in row
    assert row["instance_id"] == inst.instance_id
    assert row["schema_version"] == results.SCHEMA_VERSION
    assert row["model"] == "some-model"
    assert row["sample_index"] == 0 and row["temperature"] is None and row["seed"] is None


def test_make_row_carries_sample_metadata():
    inst = _edge_instances()[0]
    row = results.make_row(inst, "m", _attempt("majority_vote"), sample_index=3, temperature=0.7, seed=42)
    assert (row["condition"], row["sample_index"], row["temperature"], row["seed"]) == (
        "majority_vote", 3, 0.7, 42)


# --- append + read round trip -------------------------------------------------

def test_append_and_read_roundtrip(tmp_path):
    inst = _edge_instances()[0]
    path = results.shard_file(tmp_path, "baseline")
    row = results.make_row(inst, "m", _attempt())
    results.append_row(path, row)
    results.append_row(path, results.make_row(_edge_instances()[1], "m", _attempt()))
    got = results.read_rows(path)
    assert len(got) == 2
    assert got[0] == row


def test_read_missing_file_is_empty(tmp_path):
    assert results.read_rows(tmp_path / "nope.jsonl") == []


# --- torn / corrupt lines -----------------------------------------------------

def test_read_tolerates_torn_trailing_line(tmp_path):
    path = results.shard_file(tmp_path, "baseline")
    results.append_row(path, results.make_row(_edge_instances()[0], "m", _attempt()))
    # Simulate a job killed mid-write: a partial final line with no newline.
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"instance_id": "7/1/edge_existence/adjac')  # truncated
    got = results.read_rows(path)
    assert len(got) == 1  # the good row survives; the torn tail is dropped


def test_read_raises_on_midfile_corruption(tmp_path):
    path = results.shard_file(tmp_path, "baseline")
    good = json.dumps(results.make_row(_edge_instances()[0], "m", _attempt()))
    path.write_text(good + "\n" + "{not json}\n" + good + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        results.read_rows(path)


# --- resume progress ----------------------------------------------------------

def test_load_progress_unions_across_shards(tmp_path):
    a, b = _edge_instances()
    results.append_row(results.shard_file(tmp_path, "baseline", 0),
                       results.make_row(a, "m", _attempt()))
    results.append_row(results.shard_file(tmp_path, "baseline", 1),
                       results.make_row(b, "m", _attempt()))
    progress = results.load_progress(tmp_path)
    assert progress[("baseline", a.instance_id)] == {0}
    assert results.is_instance_done(progress, "baseline", a.instance_id)
    assert results.is_instance_done(progress, "baseline", b.instance_id)


def test_baseline_not_done_when_absent(tmp_path):
    a = _edge_instances()[0]
    progress = results.load_progress(tmp_path)  # empty dir
    assert not results.is_instance_done(progress, "baseline", a.instance_id)


def test_majority_vote_resume_counts_samples(tmp_path):
    inst = _edge_instances()[0]
    path = results.shard_file(tmp_path, "majority_vote")
    for k in (0, 2):  # samples 0 and 2 done; 1 and 3 still missing
        results.append_row(path, results.make_row(inst, "m", _attempt("majority_vote"), sample_index=k))
    progress = results.load_progress(tmp_path)
    assert not results.is_instance_done(progress, "majority_vote", inst.instance_id, n_samples=4)
    assert results.missing_samples(progress, "majority_vote", inst.instance_id, 4) == [1, 3]
    # once all 4 exist, it is done.
    for k in (1, 3):
        results.append_row(path, results.make_row(inst, "m", _attempt("majority_vote"), sample_index=k))
    progress = results.load_progress(tmp_path)
    assert results.is_instance_done(progress, "majority_vote", inst.instance_id, n_samples=4)


def test_expected_attempts():
    assert results.expected_attempts("baseline") == 1
    assert results.expected_attempts("debate") == 1
    assert results.expected_attempts("majority_vote", n_samples=5) == 5


# --- manifest guard -----------------------------------------------------------

def test_ensure_manifest_creates_then_matches(tmp_path):
    m = results.ensure_manifest(tmp_path, "model-A", note="first")
    assert m["model"] == "model-A" and m["schema_version"] == results.SCHEMA_VERSION
    # resuming with the same model returns the existing manifest (no overwrite).
    again = results.ensure_manifest(tmp_path, "model-A")
    assert again["note"] == "first"


def test_ensure_manifest_rejects_model_mismatch(tmp_path):
    results.ensure_manifest(tmp_path, "model-A")
    with pytest.raises(ValueError):
        results.ensure_manifest(tmp_path, "model-B")
