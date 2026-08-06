"""Check the numbers quoted in docs/findings.md against the analysis CSVs.

findings.md is written by hand from the analysis output, so a transcription slip in it is
invisible -- it looks like a result. This re-reads `analysis/` and asserts every figure the
doc quotes, which caught three miscounts the first time it ran. Regenerate the CSVs first
(the commands are at the top of findings.md), then:

    python scripts/verify_findings.py

Exits non-zero and prints every mismatch. Update the expectations here when a number in
findings.md legitimately changes -- that is the point: the doc and the check move together.
"""
import csv, os, sys

A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
fails, checks = [], 0


def load(path):
    with open(os.path.join(A, path)) as f:
        return list(csv.DictReader(f))


def key(rows, *ks):
    out = {}
    for r in rows:
        out[tuple(r[k] for k in ks)] = r
    return out


def eq(label, got, want, tol=0.0006):
    global checks
    checks += 1
    try:
        if abs(float(got) - float(want)) > tol:
            fails.append(f"{label}: doc says {want}, data says {got}")
    except (TypeError, ValueError):
        if str(got) != str(want):
            fails.append(f"{label}: doc says {want}, data says {got}")


# ---- 1. baseline fragility ----
bs = key(load("pooled-v2/baseline_summary.csv"), "task", "encoding")
for (t, e), want in {
    ("node_degree", "adjacency"): 0.388, ("node_degree", "incident"): 0.750,
    ("node_degree", "friendship"): 0.458,
    ("connected_nodes", "adjacency"): 0.260, ("connected_nodes", "incident"): 0.343,
    ("connected_nodes", "friendship"): 0.217,
    ("edge_existence", "adjacency"): 0.703, ("edge_existence", "incident"): 0.690,
    ("edge_existence", "friendship"): 0.695,
}.items():
    eq(f"§1 baseline {t}/{e}", bs[(t, e)]["accuracy"], want)

bf = key(load("pooled-v2/baseline_fragility.csv"), "task")
for t, mm in {"node_degree": 0.362, "connected_nodes": 0.127, "edge_existence": 0.013}.items():
    eq(f"§1 baseline max-min {t}", bf[(t,)]["max_min"], mm)
for t, sd in {"connected_nodes": 0.0526, "edge_existence": 0.0055, "node_degree": 0.1566}.items():
    eq(f"§3a baseline std {t}", bf[(t,)]["std"], sd)

# ---- 3. debate vs baseline ----
dvb = key(load("pooled-v2/debate_vs_baseline.csv"), "task", "encoding")
for (t, e), (base, deb, d, b, c) in {
    ("connected_nodes", "adjacency"): (0.260, 0.160, -0.100, 100, 40),
    ("connected_nodes", "incident"): (0.343, 0.438, 0.095, 98, 155),
    ("edge_existence", "adjacency"): (0.703, 0.787, 0.083, 43, 93),
    ("connected_nodes", "friendship"): (0.217, 0.140, -0.077, 95, 49),
    ("node_degree", "friendship"): (0.458, 0.412, -0.047, 116, 88),
    ("edge_existence", "incident"): (0.690, 0.730, 0.040, 74, 98),
    ("edge_existence", "friendship"): (0.695, 0.655, -0.040, 74, 50),
    ("node_degree", "adjacency"): (0.388, 0.417, 0.028, 87, 104),
    ("node_degree", "incident"): (0.750, 0.723, -0.027, 81, 65),
}.items():
    r = dvb[(t, e)]
    eq(f"§3 {t}/{e} baseline", r["baseline_acc"], base)
    eq(f"§3 {t}/{e} debate", r["debate_acc"], deb)
    eq(f"§3 {t}/{e} delta", r["delta"], d, tol=0.0011)
    eq(f"§3 {t}/{e} b", r["mcnemar_b"], b)
    eq(f"§3 {t}/{e} c", r["mcnemar_c"], c)

sig = sum(1 for r in dvb.values() if float(r["mcnemar_p"]) < 0.05)
eq("§3 'five of nine significant'", sig, 5)

# means
eq("§3c mean baseline", sum(float(r["baseline_acc"]) for r in dvb.values()) / 9, 0.501, tol=0.001)
eq("§3c mean debate", sum(float(r["debate_acc"]) for r in dvb.values()) / 9, 0.496, tol=0.001)

# ---- 3a. debate fragility ----
df = key(load("pooled-v2/debate_fragility.csv"), "task")
for t, (sd, mm) in {"connected_nodes": (0.1362, 0.2983), "edge_existence": (0.0539, 0.1317),
                    "node_degree": (0.1458, 0.3117)}.items():
    eq(f"§3a debate std {t}", df[(t,)]["std"], sd)
    eq(f"§3a debate max-min {t}", df[(t,)]["max_min"], mm)

# ---- 3b. critic ----
cc = key(load("pooled-v2/debate_critic_confusion.csv"), "task", "encoding")
p = cc[("POOLED", "all")]
for f, want in [("n_verdicts", 6424), ("ok_agree", 1274), ("ok_revise", 1710),
                ("bad_agree", 1391), ("bad_revise", 2049), ("false_alarm", 0.573),
                ("detection", 0.596), ("base_rate_wrong", 0.535), ("revise_precision", 0.545),
                ("chi2", 3.36), ("p", 0.067), ("phi", 0.023), ("odds_ratio", 1.10)]:
    eq(f"§3b pooled {f}", p[f], want, tol=0.006)

for (t, e), (v, fa, det, phi) in {
    ("edge_existence", "adjacency"): (761, 0.705, 0.947, 0.266),
    ("edge_existence", "friendship"): (790, 0.816, 0.965, 0.221),
    ("edge_existence", "incident"): (757, 0.807, 0.948, 0.181),
    ("connected_nodes", "incident"): (689, 0.302, 0.464, 0.161),
    ("node_degree", "incident"): (615, 0.138, 0.218, 0.098),
    ("node_degree", "adjacency"): (681, 0.489, 0.412, -0.076),
    ("connected_nodes", "adjacency"): (741, 0.664, 0.586, -0.058),
}.items():
    r = cc[(t, e)]
    eq(f"§3b {t}/{e} verdicts", r["n_verdicts"], v)
    eq(f"§3b {t}/{e} FA", r["false_alarm"], fa)
    eq(f"§3b {t}/{e} det", r["detection"], det)
    eq(f"§3b {t}/{e} phi", r["phi"], phi)

gr = key(load("pooled-v2/debate_critic_grounding.csv"), "task", "encoding")
for (t, e), (rev, prob, real, hal, nop) in {
    ("node_degree", "adjacency"): (301, 326, 0.88, 0.10, 5),
    ("connected_nodes", "adjacency"): (443, 470, 0.73, 0.21, 27),
    ("edge_existence", "adjacency"): (591, 591, 0.29, 0.26, 264),
    ("edge_existence", "incident"): (643, 649, 0.11, 0.04, 551),
}.items():
    r = gr[(t, e)]
    eq(f"§3b grounding {t}/{e} revises", r["revise_turns"], rev)
    eq(f"§3b grounding {t}/{e} problems", r["problems"], prob)
    eq(f"§3b grounding {t}/{e} real", r["real_rate"], real, tol=0.006)
    eq(f"§3b grounding {t}/{e} halluc", r["hallucinated_rate"], hal, tol=0.006)
    eq(f"§3b grounding {t}/{e} nopair", r["no_pair"], nop)

re_ = load("pooled-v2/debate_revision_effect.csv")
eq("§3b net corrections", sum(int(r["net"]) for r in re_), 74)
eq("§3b total revisions", sum(int(r["revisions"]) for r in re_), 3759)
rates = [float(r["changed_rate"]) for r in re_]
eq("§3b changed-rate min", round(min(rates), 2), 0.15, tol=0.011)
eq("§3b changed-rate max", round(max(rates), 2), 0.44, tol=0.011)

# ---- 3c. turn split ----
ts = key(load("pooled-v2/debate_turn_split.csv"), "task", "encoding")
for (t, e), (t1, fin, cot, loop) in {
    ("connected_nodes", "adjacency"): (0.142, 0.160, -0.118, 0.018),
    ("connected_nodes", "friendship"): (0.127, 0.140, -0.090, 0.013),
    ("connected_nodes", "incident"): (0.378, 0.438, 0.035, 0.060),
    ("edge_existence", "adjacency"): (0.732, 0.787, 0.028, 0.055),
    ("edge_existence", "friendship"): (0.653, 0.655, -0.042, 0.002),
    ("edge_existence", "incident"): (0.767, 0.730, 0.077, -0.037),
    ("node_degree", "adjacency"): (0.402, 0.417, 0.013, 0.015),
    ("node_degree", "friendship"): (0.407, 0.412, -0.052, 0.005),
    ("node_degree", "incident"): (0.732, 0.723, -0.018, -0.008),
}.items():
    r = ts[(t, e)]
    eq(f"§3c {t}/{e} turn1", r["turn1_acc"], t1)
    eq(f"§3c {t}/{e} final", r["final_acc"], fin)
    eq(f"§3c {t}/{e} cot", r["cot_delta"], cot, tol=0.0011)
    eq(f"§3c {t}/{e} loop", r["loop_delta"], loop, tol=0.0011)

eq("§3c mean turn1", sum(float(r["turn1_acc"]) for r in ts.values()) / 9, 0.482, tol=0.001)
cot_sig = sum(1 for r in ts.values() if float(r["cot_p"]) < 0.05)
loop_sig = sum(1 for r in ts.values() if float(r["loop_p"]) < 0.05)
eq("§3c 'four of nine' CoT significant", cot_sig, 4)
eq("§3c 'two of nine' loop significant", loop_sig, 2)

# ---- 3e. compliance ----
comp = key(load("pooled-v2/debate_compliance.csv"), "task", "encoding")
r = comp[("edge_existence", "friendship")]
eq("§3e ee/fri truncated", r["turn1_truncated"], 53)
eq("§3e ee/fri unparsed", r["turn1_unparsed"], 52)
eq("§3e ee/fri unparsed rate", r["turn1_unparsed_rate"], 0.087, tol=0.001)
others = [float(v["turn1_unparsed_rate"]) for k, v in comp.items()
          if k != ("edge_existence", "friendship")]
if max(others) > 0.024:
    fails.append(f"§3e 'below 2.4 percent in eight of nine': max is {max(others)}")
checks += 1

# ---- 3f. stopping rules ----
sr = key(load("pooled-v2/debate_stopping_rules.csv"), "task", "encoding", "rule")
eq("§3f gate_must_cite ee/adj", sr[("edge_existence", "adjacency", "gate_must_cite")]["delta"],
   -0.0883, tol=0.0011)
eq("§3f gate_must_cite ee/adj p",
   sr[("edge_existence", "adjacency", "gate_must_cite")]["mcnemar_p"], 3.881e-07, tol=1e-8)
eq("§3f gate_hallucinated ee/adj",
   sr[("edge_existence", "adjacency", "gate_hallucinated")]["delta"], -0.0283, tol=0.0011)

# ---- 4. error shape ----
es = {(r["task"], r["encoding"], r["metric"]): r["value"]
      for r in load("pooled-v2/debate_error_shape.csv")}
def metric(t, e, name):
    return es.get((t, e, name))
for (t, e, name), want in {
    ("node_degree", "adjacency", "mean_signed_error"): -0.962,
    ("node_degree", "incident", "mean_signed_error"): -0.093,
    ("node_degree", "friendship", "mean_signed_error"): 0.148,
    ("connected_nodes", "adjacency", "mean_jaccard"): 0.502,
    ("connected_nodes", "incident", "mean_jaccard"): 0.711,
    ("connected_nodes", "friendship", "mean_jaccard"): 0.534,
    ("connected_nodes", "adjacency", "has_extra_rate"): 0.656,
    ("connected_nodes", "incident", "has_extra_rate"): 0.448,
    ("connected_nodes", "friendship", "has_extra_rate"): 0.759,
    ("node_degree", "adjacency", "undercount_rate"): 0.441,
    ("node_degree", "friendship", "overcount_rate"): 0.348,
    ("edge_existence", "adjacency", "predicted_yes_rate"): 0.613,
    ("edge_existence", "adjacency", "gold_yes_rate"): 0.501,
    ("edge_existence", "incident", "predicted_yes_rate"): 0.612,
    ("edge_existence", "incident", "gold_yes_rate"): 0.508,
}.items():
    v = metric(t, e, name)
    if v in (None, ""):
        fails.append(f"§4 {t}/{e} {name}: column missing from CSV")
        checks += 1
    else:
        eq(f"§4 {t}/{e} {name}", v, want, tol=0.0011)

# ---- 5. power ----
disc = sorted(int(r["mcnemar_b"]) + int(r["mcnemar_c"]) for r in dvb.values())
eq("§5 discordant min", disc[0], 124)
eq("§5 discordant max", disc[-1], 253)

# ---- 2. majority vote (seed 7) ----
mv = load("main/mv_vs_baseline.csv")
if max(abs(float(r["delta"])) for r in mv) > 0.0101:
    fails.append("§2 'every delta within +/-0.010' is false")
checks += 1
if min(float(r["mcnemar_p"]) for r in mv) < 0.625:
    fails.append("§2 'every McNemar p >= 0.625' is false")
checks += 1
dd = sorted(int(r["discordant"]) for r in mv)
eq("§2 discordance min", dd[0], 1)
eq("§2 discordance max", dd[-1], 8)
eq("§2 MV token mult", max(float(r["token_mult"]) for r in mv), 10.0, tol=0.02)

print(f"checked {checks} numeric claims in docs/findings.md")
if fails:
    print(f"\n{len(fails)} MISMATCH(ES):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all consistent with the regenerated analysis")
