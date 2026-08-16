"""Debate-trace diagnostics: *why* the debate condition did or did not move accuracy.

`report.py` answers "did debate beat baseline". It did not, and a null delta on its
own is not a finding. The
tables here open the loop up and ask where the answer actually goes, using only the
committed trace sidecars, so every number regenerates with no GPU time:

- **turn split** -- debate turn 1 is a single-turn chain-of-thought answer, so `baseline
  -> turn 1` isolates the CoT prompt and `turn 1 -> final` isolates the debate loop. The
  headline `debate vs baseline` delta is the *sum* of the two and hides both.
- **critic confusion** -- is a REVISE verdict associated with the Proposer being wrong?
  This is the condition's load-bearing assumption, and it is testable directly.
- **critic grounding** -- the Critic is told to quote an edge from the graph. Checking
  each cited pair against `graph_edgelist` separates "verified badly" from "hallucinated
  the evidence", which point at different fixes.
- **revision effect** -- given a REVISE, does the answer change, and in which direction.
- **compliance** -- token-cap hits and missing `ANSWER:` lines, i.e. instances scored
  wrong for a formatting reason rather than a reasoning one.

Everything is computed from per-instance **views** (`debate_views`), so a table is a pure
aggregation over a normalized structure rather than another pass over raw JSON.

Correctness always goes through `scoring.score`, the same exact-match contract the rest
of the harness uses, so a diagnostic can never disagree with the summary tables.
"""

from __future__ import annotations

import re
from collections import defaultdict

from gedebate.eval.scoring import score
from gedebate.eval.stats import chi2_2x2, mcnemar_from_bc
from gedebate.graphqa import graph_text_encoder
from gedebate.prompts.debate import has_answer_line

# Node labels inside a Critic's cited problem line, in the encoding's label space
# ("0", "12", "James"). Same token shape scoring uses to read an answer.
_LABEL = re.compile(r"[A-Za-z0-9]+")

CELL = tuple[str, str]  # (task, encoding)


# --- per-instance views -------------------------------------------------------

def debate_views(
    traces: list[dict], rows: list[dict], *,
    max_new_tokens: int | None = None, edgelists: dict[str, list] | None = None,
) -> list[dict]:
    """Normalize (trace, result row) pairs into one flat view per debate instance.

    A view carries the turn-1 Proposer answer, the final answer, and one entry per
    Critic verdict recording *which* Proposer answer that verdict judged and what the
    revision after it did. Traces without a matching row (or vice versa) are dropped:
    a row is written only when its instance completes, so an unmatched trace is a
    killed attempt, not data.

    `max_new_tokens` is the generation cap; a turn that reached it is flagged
    `truncated`. Pass None to skip truncation flags (they become False).

    `edgelists` (instance_id -> the graph's edge list) turns on the evidence audit: each
    verdict gets an `evidence` breakdown of the pairs it cited. Without it `evidence` is
    None, meaning *unknown* rather than *ungrounded* -- the grounding table and the
    evidence-gated replay both need it, and both treat unknown as "cannot judge".
    """
    by_id = {r["instance_id"]: r for r in rows}
    views: list[dict] = []
    for t in traces:
        row = by_id.get(t["instance_id"])
        if row is None:
            continue
        edges = (edgelists or {}).get(t["instance_id"])
        views.append(_view(t["turns"], row, max_new_tokens, edges))
    return views


def _classify_evidence(problems: list[str], encoding: str, edgelist: list | None) -> dict | None:
    """Count a verdict's cited pairs as real / hallucinated / not-a-pair.

    None means the graph was not supplied, which is *unknown*, not *ungrounded*.
    """
    if edgelist is None:
        return None
    label_to_id = {label: nid for nid, label
                   in graph_text_encoder.TEXT_ENCODER_DICT[encoding].items()}
    edges = {frozenset(e) for e in edgelist}
    out = {"real": 0, "hallucinated": 0, "no_pair": 0}
    for p in problems:
        pair = _cited_pair(p, label_to_id)
        if pair is None:
            out["no_pair"] += 1
        elif pair in edges:
            out["real"] += 1
        else:
            out["hallucinated"] += 1
    return out


def _truncated(turn: dict, cap: int | None) -> bool:
    """A turn that ran into the generation cap (>= because the cap is inclusive)."""
    return cap is not None and int(turn["n_gen_tokens"]) >= cap


def _view(turns: list[dict], row: dict, cap: int | None, edgelist: list | None) -> dict:
    gt = row["ground_truth"]
    proposers = [t for t in turns if t["role"] == "proposer"]
    t1, final = proposers[0], proposers[-1]

    verdicts = []
    for i, turn in enumerate(turns):
        if turn["role"] != "critic":
            continue
        # the Proposer answer this verdict was judging, and the revision it triggered
        judged = [t for t in turns[:i] if t["role"] == "proposer"][-1]
        nxt = next((t for t in turns[i + 1:] if t["role"] == "proposer"), None)
        problems = list(turn.get("problems") or [])
        verdicts.append({
            "revise": turn["verdict"] == "REVISE",
            "parsed_ok": bool(turn["critic_verdict_parsed"]),
            "problems": problems,
            "evidence": _classify_evidence(problems, row["encoding"], edgelist),
            "truncated": _truncated(turn, cap),
            "judged_correct": score(judged["parsed"], gt),
            "next_correct": None if nxt is None else score(nxt["parsed"], gt),
            "changed": None if nxt is None else nxt["parsed"] != judged["parsed"],
        })

    return {
        "instance_id": row["instance_id"],
        "cell": (row["task"], row["encoding"]),
        "ground_truth": gt,
        "n_turns": len(turns),
        "n_proposer_turns": len(proposers),
        "turn1_answer": t1["parsed"],
        "turn1_correct": score(t1["parsed"], gt),
        "turn1_parse_ok": bool(t1["parse_ok"]),
        "turn1_has_answer_line": has_answer_line(t1["raw"]),
        "turn1_truncated": _truncated(t1, cap),
        "final_correct": bool(row["correct"]),
        "proposer_truncated": sum(_truncated(t, cap) for t in proposers),
        "verdicts": verdicts,
    }


def _by_cell(views: list[dict]) -> dict[CELL, list[dict]]:
    out: dict[CELL, list[dict]] = defaultdict(list)
    for v in views:
        out[v["cell"]].append(v)
    return out


# --- 1. the turn split: CoT effect vs loop effect ------------------------------

def turn_split(views: list[dict], base_rows: list[dict] | None = None) -> dict[CELL, dict]:
    """Per-cell `baseline -> turn 1 -> final`, each step with its own paired McNemar.

    `cot_*` compares the baseline row to the debate's turn-1 answer on the same
    instance (both single answers, so the pairing is exact); it is only populated when
    `base_rows` is given. `loop_*` compares turn 1 to the final answer within the same
    trace. `loop_fixed` = turn 1 wrong and final right, `loop_broke` = the reverse.
    """
    base_ok = {r["instance_id"]: bool(r["correct"]) for r in (base_rows or [])}
    out: dict[CELL, dict] = {}
    for cell, vs in sorted(_by_cell(views).items()):
        n = len(vs)
        fixed = sum(v["final_correct"] and not v["turn1_correct"] for v in vs)
        broke = sum(v["turn1_correct"] and not v["final_correct"] for v in vs)
        loop = mcnemar_from_bc(broke, fixed)  # b = turn-1-only right, c = final-only right

        paired = [v for v in vs if v["instance_id"] in base_ok]
        t1_only = sum(v["turn1_correct"] and not base_ok[v["instance_id"]] for v in paired)
        base_only = sum(base_ok[v["instance_id"]] and not v["turn1_correct"] for v in paired)
        cot = mcnemar_from_bc(base_only, t1_only)  # b = baseline-only right

        out[cell] = {
            "n": n,
            "baseline_accuracy": (sum(base_ok[v["instance_id"]] for v in paired) / len(paired)
                                  if paired else float("nan")),
            "turn1_accuracy": sum(v["turn1_correct"] for v in vs) / n,
            "final_accuracy": sum(v["final_correct"] for v in vs) / n,
            "n_paired_baseline": len(paired),
            "cot_delta": ((sum(v["turn1_correct"] for v in paired)
                           - sum(base_ok[v["instance_id"]] for v in paired)) / len(paired)
                          if paired else float("nan")),
            "cot_b": cot["b"], "cot_c": cot["c"], "cot_p": cot["p"],
            "loop_delta": (sum(v["final_correct"] for v in vs)
                           - sum(v["turn1_correct"] for v in vs)) / n,
            "loop_broke": broke, "loop_fixed": fixed, "loop_p": loop["p"],
            "turns_per_instance": sum(v["n_turns"] for v in vs) / n,
        }
    return out


# --- 2. is the Critic's verdict worth anything? --------------------------------

def critic_confusion(views: list[dict]) -> dict[CELL, dict]:
    """Per-cell 2x2 of {Critic verdict} x {was the judged Proposer answer correct}.

    Every verdict in every turn is one observation (not one per instance), because the
    question is about the verdict as a signal. `false_alarm` = P(REVISE | correct),
    `detection` = P(REVISE | wrong); a Critic carrying real signal has detection well
    above false-alarm. `revise_precision` vs `base_rate_wrong` says the same thing the
    way it actually matters to the loop: how much a REVISE should update our belief.
    """
    return {cell: confusion_stats(_count_verdicts(vs))
            for cell, vs in sorted(_by_cell(views).items())}


def _count_verdicts(views: list[dict]) -> dict[str, int]:
    c = defaultdict(int)
    for v in views:
        for d in v["verdicts"]:
            key = ("ok" if d["judged_correct"] else "bad") + ("_revise" if d["revise"] else "_agree")
            c[key] += 1
            c["unparsed"] += not d["parsed_ok"]
    return c


def confusion_stats(c: dict[str, int]) -> dict:
    """Rates + association test from raw {ok,bad} x {agree,revise} counts."""
    ok_r, ok_a = c.get("ok_revise", 0), c.get("ok_agree", 0)
    bad_r, bad_a = c.get("bad_revise", 0), c.get("bad_agree", 0)
    n = ok_r + ok_a + bad_r + bad_a
    # Laid out [[bad_revise, ok_revise], [bad_agree, ok_agree]] so the *useful*
    # direction is positive: phi > 0 and odds_ratio > 1 mean a REVISE tracks the answer
    # being wrong. A Critic doing its job scores high here; ours sits near zero.
    assoc = chi2_2x2(bad_r, ok_r, bad_a, ok_a)
    return {
        "ok_agree": ok_a, "ok_revise": ok_r, "bad_agree": bad_a, "bad_revise": bad_r,
        "n_verdicts": n,
        "false_alarm": ok_r / (ok_r + ok_a) if (ok_r + ok_a) else float("nan"),
        "detection": bad_r / (bad_r + bad_a) if (bad_r + bad_a) else float("nan"),
        "base_rate_wrong": (bad_r + bad_a) / n if n else float("nan"),
        "revise_precision": bad_r / (ok_r + bad_r) if (ok_r + bad_r) else float("nan"),
        "chi2": assoc["chi2"], "p": assoc["p"], "phi": assoc["phi"],
        "odds_ratio": assoc["odds_ratio"],
        "unparsed": c.get("unparsed", 0),
    }


def pooled_confusion(views: list[dict]) -> dict:
    """The confusion table over every cell at once (the headline number)."""
    return confusion_stats(_count_verdicts(views))


# --- 3. is the Critic's cited evidence real? -----------------------------------

def critic_grounding(views: list[dict]) -> dict[CELL, dict]:
    """Per-cell audit of the edges a REVISE cites against the graph's true edge list.

    The Critic prompt requires each problem to quote an edge from the graph text. Each
    cited pair (annotated onto the view by `debate_views(..., edgelists=...)`) lands in
    one of three buckets:

    - `real_edge` -- the cited pair is in the graph (a legitimate citation, whether or
      not the surrounding claim is right)
    - `hallucinated` -- two labels naming a pair that is not in the graph
    - `no_pair` -- fewer than two distinct labels, i.e. prose rather than a citation

    Verdicts whose evidence was never classified (no edge list supplied) are skipped, so
    a partial dataset degrades to a smaller denominator rather than a wrong rate.
    """
    out: dict[CELL, dict] = {}
    for cell, vs in sorted(_by_cell(views).items()):
        c = defaultdict(int)
        for v in vs:
            for d in v["verdicts"]:
                if not d["revise"] or d["evidence"] is None:
                    continue
                c["revise_turns"] += 1
                if not d["problems"]:
                    c["no_problem_line"] += 1
                c["real_edge"] += d["evidence"]["real"]
                c["hallucinated"] += d["evidence"]["hallucinated"]
                c["no_pair"] += d["evidence"]["no_pair"]
                c["problems"] += sum(d["evidence"].values())
        n = c["problems"]
        out[cell] = {
            "revise_turns": c["revise_turns"], "no_problem_line": c["no_problem_line"],
            "problems": n, "real_edge": c["real_edge"],
            "hallucinated": c["hallucinated"], "no_pair": c["no_pair"],
            "real_rate": c["real_edge"] / n if n else float("nan"),
            "hallucinated_rate": c["hallucinated"] / n if n else float("nan"),
            "no_pair_rate": c["no_pair"] / n if n else float("nan"),
        }
    return out


def _cited_pair(problem: str, label_to_id: dict[str, int]) -> frozenset | None:
    """The first two distinct node ids named in a problem line, or None if it names <2.

    First two, not any two: a citation is meant to be one edge, and taking the first
    pair mentioned matches how the line reads ("no edge between David and Robert").
    """
    ids: list[int] = []
    for tok in _LABEL.findall(problem):
        nid = label_to_id.get(tok)
        if nid is not None and nid not in ids:
            ids.append(nid)
        if len(ids) == 2:
            return frozenset(ids)
    return None


# --- 4. what a REVISE actually does to the answer ------------------------------

def revision_effect(views: list[dict]) -> dict[CELL, dict]:
    """Per-cell transition counts for the Proposer turn that follows a REVISE.

    `changed` is the share of REVISEs the Proposer acted on at all; the four transitions
    say which way it went. `ok_to_bad` >= `bad_to_ok` means the revision step is a net
    loss, i.e. extra turns are buying variance rather than accuracy.
    """
    out: dict[CELL, dict] = {}
    for cell, vs in sorted(_by_cell(views).items()):
        c = defaultdict(int)
        for v in vs:
            for d in v["verdicts"]:
                if not d["revise"] or d["next_correct"] is None:
                    continue  # budget hit: a REVISE with no room left to revise
                c["revisions"] += 1
                c["changed"] += bool(d["changed"])
                before = "ok" if d["judged_correct"] else "bad"
                after = "ok" if d["next_correct"] else "bad"
                c[f"{before}_to_{after}"] += 1
        n = c["revisions"]
        out[cell] = {
            "revisions": n, "changed": c["changed"],
            "changed_rate": c["changed"] / n if n else float("nan"),
            "ok_to_ok": c["ok_to_ok"], "ok_to_bad": c["ok_to_bad"],
            "bad_to_ok": c["bad_to_ok"], "bad_to_bad": c["bad_to_bad"],
            "net": c["bad_to_ok"] - c["ok_to_bad"],
        }
    return out


# --- 5. format compliance (wrong for a non-reasoning reason) -------------------

def compliance(views: list[dict]) -> dict[CELL, dict]:
    """Per-cell format failures at turn 1, plus truncation across all turns.

    A missing `ANSWER:` line is reported separately from `parse_ok` on purpose: the
    parser falls back to the whole turn text, which silently rescues node_degree (last
    integer wins) and silently fails connected_nodes. Counting both shows how much of a
    cell's error is the prompt rather than the graph reasoning.
    """
    out: dict[CELL, dict] = {}
    for cell, vs in sorted(_by_cell(views).items()):
        n = len(vs)
        crit = [d for v in vs for d in v["verdicts"]]
        out[cell] = {
            "n": n,
            "turn1_truncated": sum(v["turn1_truncated"] for v in vs),
            "turn1_no_answer_line": sum(not v["turn1_has_answer_line"] for v in vs),
            "turn1_unparsed": sum(not v["turn1_parse_ok"] for v in vs),
            "turn1_unparsed_rate": sum(not v["turn1_parse_ok"] for v in vs) / n,
            "proposer_turns": sum(v["n_proposer_turns"] for v in vs),
            "proposer_truncated": sum(v["proposer_truncated"] for v in vs),
            "critic_turns": len(crit),
            "critic_truncated": sum(d["truncated"] for d in crit),
            "critic_no_verdict": sum(not d["parsed_ok"] for d in crit),
        }
    return out


# --- 6. counterfactual stopping rules (replayed, not simulated) ----------------
#
# A trace records every turn, so a rule that makes the loop stop EARLIER than it really
# did can be evaluated exactly: truncate the transcript and read off the Proposer answer
# that was standing at that point. This is a replay, not a model of what the Proposer
# would have said -- which is precisely the limit. A rule that changes what a turn
# *generates* (a different revision prompt, a bigger Critic) cannot be evaluated here and
# needs GPU time.
#
# Each rule is a function view -> bool (was the final answer correct under this rule).

def _rule_actual(view: dict) -> bool:
    """The run as it happened, the baseline every other rule is compared against."""
    return view["final_correct"]


def _rule_turn1(view: dict) -> bool:
    """Never debate: keep the first Proposer answer. The floor for the loop's value."""
    return view["turn1_correct"]


def _rule_at_most_one_revision(view: dict) -> bool:
    """Allow a single revision, then stop -- the loop's first step only."""
    first = view["verdicts"][0] if view["verdicts"] else None
    if first is None or not first["revise"] or first["next_correct"] is None:
        return view["turn1_correct"]  # AGREE, or no room to revise: nothing changed
    return first["next_correct"]


def _stop_at(view: dict, reject) -> bool:
    """Walk the verdicts, stopping the first time `reject` vetoes one.

    A vetoed REVISE is treated as the AGREE it would have become, so the answer standing
    at that point (`judged_correct`) is final. An AGREE already stopped the real loop, so
    it returns the same thing. Falling off the end means no verdict was vetoed and the
    replay agrees with the run.
    """
    for d in view["verdicts"]:
        if not d["revise"] or reject(d):
            return d["judged_correct"]
    return view["final_correct"]


def _rule_gate_hallucinated(view: dict) -> bool:
    """Reject a REVISE that cites a pair which is not an edge of the graph.

    The permissive gate: a REVISE citing no pair at all (the Critic wrote prose, which is
    most of `edge_existence`) is *not* vetoed, because nothing in it is refutable.
    Evidence we could not classify is likewise not vetoed.
    """
    return _stop_at(view, lambda d: bool(d["evidence"]) and d["evidence"]["hallucinated"] > 0)


def _rule_gate_must_cite(view: dict) -> bool:
    """Reject a REVISE unless it cites a real edge -- the strict reading of the prompt.

    Prose-only critiques are vetoed here, which is the whole difference from the
    permissive gate and is why both are worth measuring.
    """
    return _stop_at(view, lambda d: d["evidence"] is not None and d["evidence"]["real"] == 0)


STOPPING_RULES = {
    "actual": _rule_actual,
    "turn1_only": _rule_turn1,
    "at_most_one_revision": _rule_at_most_one_revision,
    "gate_hallucinated": _rule_gate_hallucinated,
    "gate_must_cite": _rule_gate_must_cite,
}


def replay_stopping_rules(
    views: list[dict], rules: dict | None = None
) -> dict[CELL, dict[str, dict]]:
    """Per-cell accuracy under each stopping rule, paired against the run as it happened.

    Returns `{cell: {rule_name: {accuracy, delta, b, c, p}}}`, where `delta` is the rule's
    accuracy minus the actual run's and the McNemar pairs the two on the same instances
    (`b` = actual-only right, `c` = rule-only right).

    Rules that need the evidence audit (`gate_*`) are only meaningful when the views were
    built with `edgelists`; without it every verdict's evidence is None, no REVISE is ever
    vetoed, and those rules collapse onto `actual`.
    """
    rules = rules or STOPPING_RULES
    out: dict[CELL, dict[str, dict]] = {}
    for cell, vs in sorted(_by_cell(views).items()):
        n = len(vs)
        actual = [v["final_correct"] for v in vs]
        cell_out: dict[str, dict] = {}
        for name, rule in rules.items():
            got = [rule(v) for v in vs]
            b = sum(a and not g for a, g in zip(actual, got))
            c = sum(g and not a for a, g in zip(actual, got))
            mc = mcnemar_from_bc(b, c)
            cell_out[name] = {
                "accuracy": sum(got) / n,
                "delta": (sum(got) - sum(actual)) / n,
                "b": b, "c": c, "p": mc["p"],
            }
        out[cell] = cell_out
    return out


# --- 7. what a wrong answer looks like ----------------------------------------

def error_shape(views: list[dict]) -> dict[CELL, dict]:
    """Per-cell shape of the turn-1 error, in the terms each task's answer is made of.

    Accuracy says how often the model is wrong; this says *how*, which is what
    separates two encodings that fail at the same rate for opposite reasons (adjacency
    undercounts a degree, friendship overcounts it). The metric set is task-specific
    because the answers are different types:

    - node_degree: `mean_signed_error` (+ means the model claims too high a degree),
      over/undercount rates, and how often it is off by exactly one
    - connected_nodes: mean Jaccard against the gold neighbour set, plus how often the
      answer contains a non-neighbour (`has_extra`) or drops a real one (`has_missing`)
    - edge_existence: the model's Yes rate against the gold Yes rate (its answer bias)

    Rates are over *parsed* answers; `n_unparsed` is reported alongside so a cell that
    is mostly parse failures cannot masquerade as a well-shaped error.
    """
    out: dict[CELL, dict] = {}
    for cell, vs in sorted(_by_cell(views).items()):
        task = cell[0]
        parsed = [v for v in vs if v["turn1_answer"] is not None]
        out[cell] = {
            "task": task, "n": len(vs), "n_parsed": len(parsed),
            "n_unparsed": len(vs) - len(parsed),
            "metrics": _shape_metrics(task, parsed),
        }
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _shape_metrics(task: str, parsed: list[dict]) -> dict[str, float]:
    if task == "node_degree":
        errs = [v["turn1_answer"] - v["ground_truth"] for v in parsed]
        return {
            "mean_signed_error": _mean(errs),
            "overcount_rate": _mean([float(e > 0) for e in errs]),
            "undercount_rate": _mean([float(e < 0) for e in errs]),
            "off_by_one_rate": _mean([float(abs(e) == 1) for e in errs]),
        }
    if task == "connected_nodes":
        jac, extra, missing = [], [], []
        for v in parsed:
            pred, gold = set(v["turn1_answer"]), set(v["ground_truth"])
            union = pred | gold
            jac.append(len(pred & gold) / len(union) if union else 1.0)  # both empty = exact
            extra.append(float(bool(pred - gold)))
            missing.append(float(bool(gold - pred)))
        return {"mean_jaccard": _mean(jac), "has_extra_rate": _mean(extra),
                "has_missing_rate": _mean(missing)}
    if task == "edge_existence":
        pred_yes = _mean([float(bool(v["turn1_answer"])) for v in parsed])
        gold_yes = _mean([float(bool(v["ground_truth"])) for v in parsed])
        return {"predicted_yes_rate": pred_yes, "gold_yes_rate": gold_yes,
                "yes_bias": pred_yes - gold_yes}
    raise NotImplementedError(f"no error shape defined for task {task!r}")


# --- formatting ---------------------------------------------------------------

def _cell(key: CELL) -> str:
    return f"{key[0]}/{key[1]}"


def format_turn_split(split: dict[CELL, dict]) -> str:
    """`baseline -> turn 1 -> final` with each step's paired McNemar p."""
    from gedebate.eval.report import _stars
    header = (f"{'task/encoding':<28}{'n':>4}{'base':>7}{'turn1':>7}{'final':>7}"
              f"{'CoT d':>8}{'p':>9}{'':>4}{'loop d':>9}{'p':>9}{'':>4}{'turns':>7}")
    lines = [header, "-" * len(header)]
    for key, s in split.items():
        lines.append(
            f"{_cell(key):<28}{s['n']:>4}{s['baseline_accuracy']:>7.3f}"
            f"{s['turn1_accuracy']:>7.3f}{s['final_accuracy']:>7.3f}"
            f"{s['cot_delta']:>+8.3f}{s['cot_p']:>9.4f} {_stars(s['cot_p']):<3}"
            f"{s['loop_delta']:>+9.3f}{s['loop_p']:>9.4f} {_stars(s['loop_p']):<3}"
            f"{s['turns_per_instance']:>7.2f}"
        )
    return "\n".join(lines)


def format_critic_confusion(conf: dict[CELL, dict], pooled: dict | None = None) -> str:
    from gedebate.eval.report import _stars
    header = (f"{'task/encoding':<28}{'verdicts':>9}{'FA|ok':>8}{'det|bad':>9}"
              f"{'gap':>8}{'phi':>8}{'p':>9}{'':>4}{'unparsed':>9}")
    lines = [header, "-" * len(header)]
    for key, s in list(conf.items()) + ([(("POOLED", ""), pooled)] if pooled else []):
        lines.append(
            f"{_cell(key).rstrip('/'):<28}{s['n_verdicts']:>9}{s['false_alarm']:>8.3f}"
            f"{s['detection']:>9.3f}{s['detection'] - s['false_alarm']:>+8.3f}"
            f"{s['phi']:>+8.3f}{s['p']:>9.4f} {_stars(s['p']):<3}{s['unparsed']:>9}"
        )
    return "\n".join(lines)


def format_critic_grounding(g: dict[CELL, dict]) -> str:
    header = (f"{'task/encoding':<28}{'REVISEs':>8}{'problems':>9}"
              f"{'real edge':>11}{'hallucinated':>14}{'no pair':>9}")
    lines = [header, "-" * len(header)]
    for key, s in g.items():
        lines.append(
            f"{_cell(key):<28}{s['revise_turns']:>8}{s['problems']:>9}"
            f"{s['real_edge']:>6} ({s['real_rate']:.2f}){s['hallucinated']:>7} "
            f"({s['hallucinated_rate']:.2f}){s['no_pair']:>9}"
        )
    return "\n".join(lines)


def format_revision_effect(rev: dict[CELL, dict]) -> str:
    header = (f"{'task/encoding':<28}{'revisions':>10}{'changed':>13}{'ok>bad':>8}"
              f"{'bad>ok':>8}{'net':>6}{'ok>ok':>7}{'bad>bad':>9}")
    lines = [header, "-" * len(header)]
    for key, s in rev.items():
        changed = f"{s['changed']} ({s['changed_rate']:.2f})"
        lines.append(
            f"{_cell(key):<28}{s['revisions']:>10}{changed:>13}{s['ok_to_bad']:>8}"
            f"{s['bad_to_ok']:>8}{s['net']:>+6}{s['ok_to_ok']:>7}{s['bad_to_bad']:>9}"
        )
    return "\n".join(lines)


def format_compliance(comp: dict[CELL, dict]) -> str:
    header = (f"{'task/encoding':<28}{'n':>4}{'t1 trunc':>9}{'t1 noANS':>9}"
              f"{'t1 unparsed':>12}{'prop trunc':>12}{'crit trunc':>12}{'crit noVERD':>12}")
    lines = [header, "-" * len(header)]
    for key, s in comp.items():
        prop = f"{s['proposer_truncated']}/{s['proposer_turns']}"
        crit = f"{s['critic_truncated']}/{s['critic_turns']}"
        unp = f"{s['turn1_unparsed']} ({s['turn1_unparsed_rate']:.2f})"
        lines.append(
            f"{_cell(key):<28}{s['n']:>4}{s['turn1_truncated']:>9}"
            f"{s['turn1_no_answer_line']:>9}{unp:>12}{prop:>12}{crit:>12}"
            f"{s['critic_no_verdict']:>12}"
        )
    return "\n".join(lines)


def format_stopping_rules(replay: dict[CELL, dict[str, dict]]) -> str:
    """One column per rule; each cell shows accuracy and its delta against the real run."""
    from gedebate.eval.report import _stars
    names = list(next(iter(replay.values())).keys()) if replay else []
    header = f"{'task/encoding':<28}" + "".join(f"{n:>22}" for n in names)
    lines = [header, "-" * len(header)]
    for key, per_rule in replay.items():
        row = f"{_cell(key):<28}"
        for name in names:
            s = per_rule[name]
            mark = "" if name == "actual" else f" {_stars(s['p'])}"
            cell = (f"{s['accuracy']:.3f}" if name == "actual"
                    else f"{s['accuracy']:.3f} ({s['delta']:+.3f}){mark}")
            row += f"{cell:>22}"
        lines.append(row)
    return "\n".join(lines)


def format_error_shape(shape: dict[CELL, dict]) -> str:
    """One line per cell; the metric names differ by task, so they are printed inline."""
    lines = [f"{'task/encoding':<28}{'n':>5}{'unparsed':>10}   metrics",
             "-" * 78]
    for key, s in shape.items():
        metrics = "  ".join(f"{k}={v:+.3f}" for k, v in s["metrics"].items())
        lines.append(f"{_cell(key):<28}{s['n']:>5}{s['n_unparsed']:>10}   {metrics}")
    return "\n".join(lines)


# --- CSV artifacts ------------------------------------------------------------

def turn_split_to_csv(split: dict[CELL, dict]) -> str:
    lines = ["task,encoding,n,baseline_acc,turn1_acc,final_acc,"
             "cot_delta,cot_b,cot_c,cot_p,loop_delta,loop_broke,loop_fixed,loop_p,"
             "turns_per_instance"]
    for (task, enc), s in split.items():
        lines.append(
            f"{task},{enc},{s['n']},{s['baseline_accuracy']:.4f},{s['turn1_accuracy']:.4f},"
            f"{s['final_accuracy']:.4f},{s['cot_delta']:+.4f},{s['cot_b']},{s['cot_c']},"
            f"{s['cot_p']:.4g},{s['loop_delta']:+.4f},{s['loop_broke']},{s['loop_fixed']},"
            f"{s['loop_p']:.4g},{s['turns_per_instance']:.2f}"
        )
    return "\n".join(lines) + "\n"


def critic_confusion_to_csv(conf: dict[CELL, dict], pooled: dict | None = None) -> str:
    lines = ["task,encoding,n_verdicts,ok_agree,ok_revise,bad_agree,bad_revise,"
             "false_alarm,detection,base_rate_wrong,revise_precision,chi2,p,phi,"
             "odds_ratio,unparsed"]
    for (task, enc), s in list(conf.items()) + ([(("POOLED", "all"), pooled)] if pooled else []):
        lines.append(
            f"{task},{enc},{s['n_verdicts']},{s['ok_agree']},{s['ok_revise']},"
            f"{s['bad_agree']},{s['bad_revise']},{s['false_alarm']:.4f},"
            f"{s['detection']:.4f},{s['base_rate_wrong']:.4f},{s['revise_precision']:.4f},"
            f"{s['chi2']:.4f},{s['p']:.4g},{s['phi']:+.4f},{s['odds_ratio']:.4f},"
            f"{s['unparsed']}"
        )
    return "\n".join(lines) + "\n"


def critic_grounding_to_csv(g: dict[CELL, dict]) -> str:
    lines = ["task,encoding,revise_turns,no_problem_line,problems,"
             "real_edge,hallucinated,no_pair,real_rate,hallucinated_rate,no_pair_rate"]
    for (task, enc), s in g.items():
        lines.append(
            f"{task},{enc},{s['revise_turns']},{s['no_problem_line']},{s['problems']},"
            f"{s['real_edge']},{s['hallucinated']},{s['no_pair']},{s['real_rate']:.4f},"
            f"{s['hallucinated_rate']:.4f},{s['no_pair_rate']:.4f}"
        )
    return "\n".join(lines) + "\n"


def revision_effect_to_csv(rev: dict[CELL, dict]) -> str:
    lines = ["task,encoding,revisions,changed,changed_rate,"
             "ok_to_ok,ok_to_bad,bad_to_ok,bad_to_bad,net"]
    for (task, enc), s in rev.items():
        lines.append(
            f"{task},{enc},{s['revisions']},{s['changed']},{s['changed_rate']:.4f},"
            f"{s['ok_to_ok']},{s['ok_to_bad']},{s['bad_to_ok']},{s['bad_to_bad']},{s['net']:+d}"
        )
    return "\n".join(lines) + "\n"


def stopping_rules_to_csv(replay: dict[CELL, dict[str, dict]]) -> str:
    """Long format (one row per cell per rule): the rule set is not fixed."""
    lines = ["task,encoding,rule,accuracy,delta,mcnemar_b,mcnemar_c,mcnemar_p"]
    for (task, enc), per_rule in replay.items():
        for name, s in per_rule.items():
            lines.append(f"{task},{enc},{name},{s['accuracy']:.4f},{s['delta']:+.4f},"
                         f"{s['b']},{s['c']},{s['p']:.4g}")
    return "\n".join(lines) + "\n"


def error_shape_to_csv(shape: dict[CELL, dict]) -> str:
    """Long format (one row per metric): the metric set is task-specific, so a wide
    table would be mostly empty columns."""
    lines = ["task,encoding,n,n_parsed,n_unparsed,metric,value"]
    for (task, enc), s in shape.items():
        for name, value in s["metrics"].items():
            lines.append(f"{task},{enc},{s['n']},{s['n_parsed']},{s['n_unparsed']},"
                         f"{name},{value:.4f}")
    return "\n".join(lines) + "\n"


def compliance_to_csv(comp: dict[CELL, dict]) -> str:
    lines = ["task,encoding,n,turn1_truncated,turn1_no_answer_line,turn1_unparsed,"
             "turn1_unparsed_rate,proposer_turns,proposer_truncated,"
             "critic_turns,critic_truncated,critic_no_verdict"]
    for (task, enc), s in comp.items():
        lines.append(
            f"{task},{enc},{s['n']},{s['turn1_truncated']},{s['turn1_no_answer_line']},"
            f"{s['turn1_unparsed']},{s['turn1_unparsed_rate']:.4f},{s['proposer_turns']},"
            f"{s['proposer_truncated']},{s['critic_turns']},{s['critic_truncated']},"
            f"{s['critic_no_verdict']}"
        )
    return "\n".join(lines) + "\n"
