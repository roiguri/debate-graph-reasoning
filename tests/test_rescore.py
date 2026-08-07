"""Tests for scripts/rescore.py -- specifically that it reads each row with the parser
that row's PROMPT FORMAT requires, not the one its condition name suggests.

`majority_vote_cot` samples the debate Proposer prompt, so its stored output is a
numbered-claim trace and must go through `parse_proposer`. Sending it to the shared
`scoring.parse` scans the whole text and harvests labels out of the reasoning, which
silently rewrites answers during a re-score.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from gedebate.eval.results import PROPOSER_FORMAT_CONDITIONS

REPO = Path(__file__).resolve().parents[1]

# A Proposer-format trace whose reasoning names a node the answer excludes. The two
# parsers disagree on it, which is the whole point.
_TRACE = "1. Node 1 is connected to node 0.\n2. Node 1 is not connected to node 3.\nANSWER: 0"


def _rescore():
    spec = importlib.util.spec_from_file_location("rescore", REPO / "scripts" / "rescore.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(condition):
    return {"condition": condition, "task": "connected_nodes", "encoding": "adjacency",
            "ground_truth": [0]}


def test_proposer_format_conditions_read_the_answer_line():
    mod = _rescore()
    for condition in PROPOSER_FORMAT_CONDITIONS:
        value, ok, correct = mod._reparse(_TRACE, _row(condition), [1])
        assert (value, ok, correct) == ([0], True, True), condition


def test_bare_answer_conditions_use_the_shared_parser():
    # baseline / terse majority-vote never emit a claim trace; they keep scoring.parse,
    # which is what they called when they ran.
    mod = _rescore()
    for condition in ("baseline", "majority_vote"):
        value, _ok, _correct = mod._reparse(_TRACE, _row(condition), [1])
        assert value != [0], condition  # whole-text scan -> harvests the negated claim


def test_the_cot_arm_is_declared_proposer_format():
    # The bug this file guards: the arm was added to the runner and the results layer but
    # not to rescore's dispatch, so a re-score would have silently rewritten its answers.
    assert "majority_vote_cot" in PROPOSER_FORMAT_CONDITIONS
    assert "debate" in PROPOSER_FORMAT_CONDITIONS
    assert "baseline" not in PROPOSER_FORMAT_CONDITIONS
    assert "majority_vote" not in PROPOSER_FORMAT_CONDITIONS
