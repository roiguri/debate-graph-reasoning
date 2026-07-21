"""Conditions: baseline, majority_vote, debate.

Each condition exposes a `run_instance(model, instance, ...) -> dict` that produces
one attempt-level record under the shared persistence schema. Added one at a time,
baseline-first, so every condition is proven before the next (see overview.md).
"""
