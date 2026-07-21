"""Conditions: baseline (P2), majority_vote (P4), debate (P5).

Each condition exposes a `run_instance(model, instance, ...) -> dict` that produces
one attempt-level record under the shared persistence schema. Added one at a time,
baseline-first, so every condition is proven before the next (see overview.md).
"""
