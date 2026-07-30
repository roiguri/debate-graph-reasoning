"""Integration test for scripts/show_results.py --save file naming.

Runs the real script against a synthetic run dir (baseline + majority_vote rows for
one cell) and asserts --save writes condition-tagged, unambiguous filenames. Torch-
free: show_results imports only report/results/stats.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gedebate.data.dataset import build_dataset
from gedebate.eval import results

REPO = Path(__file__).resolve().parents[1]


def _make_run_dir(run_dir: Path) -> None:
    inst = next(i for i in build_dataset(n_graphs=2, seed=7)
                if i.task == "edge_existence" and i.encoding == "adjacency")
    attempt = {
        "condition": "baseline", "raw_output": "Yes.", "parsed_answer": True,
        "parse_ok": True, "correct": True, "ground_truth": inst.ground_truth,
        "n_prompt_tokens": 5, "n_gen_tokens": 2,
    }
    results.append_row(results.shard_file(run_dir, "baseline"),
                       results.make_row(inst, "m", attempt))
    for si in range(3):  # 3 MV draws for the same instance -> a votable cell
        mv = dict(attempt, condition="majority_vote")
        results.append_row(results.shard_file(run_dir, "majority_vote"),
                           results.make_row(inst, "m", mv, sample_index=si,
                                            temperature=0.7, seed=si))


def test_save_writes_condition_tagged_files(tmp_path):
    run_dir = tmp_path / "run"
    _make_run_dir(run_dir)
    out = tmp_path / "analysis"
    subprocess.run(
        [sys.executable, "scripts/show_results.py", str(run_dir), "--compare", "--save", str(out)],
        cwd=REPO, check=True,
    )
    names = {p.name for p in out.iterdir()}
    assert names == {
        "baseline_summary.csv", "baseline_fragility.csv", "baseline_significance.csv",
        "mv_vote_summary.csv", "mv_vs_baseline.csv",
    }
    # the old generic names must be gone (they were the source of the baseline leak)
    assert "summary.csv" not in names and "vote_summary.csv" not in names


def test_condition_filter_excludes_mv(tmp_path):
    # --condition baseline drops MV rows, so a baseline-only (e.g. pooled) save
    # writes no mv_* files.
    run_dir = tmp_path / "run"
    _make_run_dir(run_dir)
    out = tmp_path / "analysis"
    subprocess.run(
        [sys.executable, "scripts/show_results.py", str(run_dir),
         "--condition", "baseline", "--save", str(out)],
        cwd=REPO, check=True,
    )
    names = {p.name for p in out.iterdir()}
    assert names == {"baseline_summary.csv", "baseline_fragility.csv", "baseline_significance.csv"}
    assert not any(n.startswith("mv_") for n in names)
