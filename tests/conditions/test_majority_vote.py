"""Tests for the majority-vote condition -- the vote rule, per-draw seeding, and
`run_sample` end-to-end with a stub model.

The runner-level N-sample persistence + resume is tested in tests/eval/test_runner.py;
here we cover the pure logic (torch-free).
"""

from __future__ import annotations

from dataclasses import dataclass

from gedebate.conditions.majority_vote import run_sample, sample_seed, vote
from gedebate.eval.runner import first_instance


# --- vote rule ----------------------------------------------------------------

def test_vote_bool_majority():
    assert vote([True, True, False]) == (True, True, 2)


def test_vote_int_majority():
    assert vote([3, 2, 3, 3, 1]) == (3, True, 3)


def test_vote_list_answers_keyed_by_value():
    # connected_nodes: unhashable list answers still tally by value.
    assert vote([[1, 2], [1, 2], [3]]) == ([1, 2], True, 2)


def test_vote_excludes_parse_failures():
    # None (parse failure) is dropped from the tally, not counted as an answer.
    assert vote([None, False, None, False]) == (False, True, 2)


def test_vote_all_fail_is_parse_failure():
    assert vote([None, None]) == (None, False, 0)


def test_vote_tiebreak_lowest_first_index():
    # 2-2 tie between True and False: True's first support is index 0, so it wins.
    assert vote([True, False, True, False]) == (True, True, 2)
    # ...and the mirror image resolves the other way.
    assert vote([False, True, False, True]) == (False, True, 2)


def test_vote_tiebreak_at_real_even_n10():
    # N=10 is even, so ties are possible (~5% of real instances, mostly multi-class).
    # Binary 5-5 split: first supporter of False is index 0, so False wins.
    draws = [False, True, False, True, False, True, False, True, False, True]
    assert vote(draws) == (False, True, 5)
    # Multi-class 3-way tie an odd N would NOT prevent (node_degree-style): 4-3-3,
    # here forced to a clean tie 3-3-4 -> the count-4 answer wins outright...
    assert vote([1, 1, 1, 2, 2, 2, 3, 3, 3, 3]) == (3, True, 4)
    # ...but a genuine top-tie (3-3, rest scattered) resolves to the lowest index.
    assert vote([2, 2, 2, 3, 3, 3, 4, 5, 6, 7]) == (2, True, 3)


# --- per-draw seed derivation -------------------------------------------------

def test_sample_seed_deterministic_and_varies():
    a = sample_seed("7/0/edge_existence/adjacency", 0)
    assert a == sample_seed("7/0/edge_existence/adjacency", 0)  # stable across calls
    assert a != sample_seed("7/0/edge_existence/adjacency", 1)  # varies by index
    assert a != sample_seed("7/1/edge_existence/adjacency", 0)  # varies by instance
    assert 0 <= a < 2**32  # safe range for torch.manual_seed


# --- run_sample end-to-end with a stub model ----------------------------------

@dataclass
class _StubGen:
    text: str
    n_gen_tokens: int
    n_prompt_tokens: int


class _StubModel:
    """Duck-types Model.generate -> GenResult, recording the sampling kwargs it saw."""

    def __init__(self, reply: str):
        self.reply = reply
        self.seen: list[dict] = []

    def generate(self, prompt: str, *, max_new_tokens: int = 64, temperature=None,
                 top_p=None, top_k=None, seed=None):
        self.seen.append({"temperature": temperature, "top_p": top_p,
                          "top_k": top_k, "seed": seed})
        return _StubGen(self.reply, n_gen_tokens=9, n_prompt_tokens=13)


def _edge_instance():
    return first_instance(n_graphs=4, seed=7, task="edge_existence", encoding="adjacency")


def test_run_sample_record_shape_and_passes_sampling_kwargs():
    inst = _edge_instance()
    gold = "Yes." if inst.ground_truth else "No."
    model = _StubModel(gold)

    rec = run_sample(model, inst, sample_index=2, temperature=0.7, top_p=1.0, top_k=0)

    assert rec["condition"] == "majority_vote"
    assert rec["sample_index"] == 2
    assert rec["correct"] is True
    assert rec["parse_ok"] is True
    assert rec["n_gen_tokens"] == 9 and rec["n_prompt_tokens"] == 13
    # the draw was sampled (temperature + explicit truncation) with the derived seed
    assert model.seen[0]["temperature"] == 0.7
    assert model.seen[0]["top_p"] == 1.0 and model.seen[0]["top_k"] == 0
    assert rec["seed"] == sample_seed(inst.instance_id, 2)
    assert model.seen[0]["seed"] == rec["seed"]
