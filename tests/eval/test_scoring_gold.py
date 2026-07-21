"""Fidelity check: our parser must score GraphQA's own gold answers as correct
for every task x encoding. This is what makes the P3 3x3 matrix a config-only
change -- if the parser round-trips the gold across all nine combos, no new parsing
code is needed to widen the run.
"""

from __future__ import annotations

from gedebate.data.dataset import ENCODINGS, TASKS, build_dataset
from gedebate.eval.scoring import parse, score


def test_parser_scores_gold_answers_correct_across_matrix():
    seen = set()
    for inst in build_dataset(n_graphs=40, seed=7):
        parsed, ok = parse(
            inst.task, inst.answer, encoding=inst.encoding, node_ids=inst.node_ids
        )
        assert ok, f"gold answer failed to parse: {inst.task}/{inst.encoding}: {inst.answer!r}"
        assert score(parsed, inst.ground_truth), (
            f"gold mismatch {inst.task}/{inst.encoding}: "
            f"parsed={parsed} gold={inst.answer!r} gt={inst.ground_truth}"
        )
        seen.add((inst.task, inst.encoding))
    # all nine task x encoding combinations were actually exercised
    assert seen == {(t, e) for t in TASKS for e in ENCODINGS}
