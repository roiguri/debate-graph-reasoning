# Paper editing handoff (session continuation)

Written 2026-08-16. This continues the paper-polishing work on
`docs/paper/main.tex` (ACL-format LaTeX). For data provenance and the original
narrative, see **[paper-handoff.md](paper-handoff.md)** (the authoritative source
for canonical numbers). This file captures the *editing* state and conventions so a
fresh chat can continue section-by-section.

Last commit at handoff: `cb13704 docs(paper): add debate flow figure; tighten
§4.4–4.5, discussion, appendix` (pushed to `main`).

---

## How we work (conventions — follow these)

- **Rebuild the PDF after every edit.** From `docs/paper/`:
  `latexmk -pdf -interaction=nonstopmode main.tex` (exit 0 expected). The Bash cwd
  keeps resetting to repo root, so always `cd .../docs/paper &&` in the same command.
- **Verify visually** when layout matters: `pdftoppm -png -r 100 -f N -l N main.pdf out`
  then read the PNG. Scratchpad is the place for temp renders.
- **No fabricated numbers.** Every number in the paper must come from the real runs.
  Regenerate from the six run dirs (see below) rather than trusting memory or stale
  CSVs. Net effects must reconcile (e.g. fixed − broke = −113).
- **Style prefs the user has enforced repeatedly:** no italics/`\emph`; captions
  describe *only what we see*, not results/conclusions; move numbers into
  tables/figures where possible; simpler, more direct, less verbose; hedge claims as
  "our theory," don't be over-decisive; avoid the word "scaffold" and avoid talking
  about "cells" in body prose.
- **Commits:** short one-line subject, no `Co-Authored-By` trailer. Work is committed
  directly to `main` (that's the established pattern here).
- **If prompts change** (`src/gedebate/prompts/*`), regenerate `analysis/prompts/*.txt`
  in the same commit (not relevant unless you touch prompts).

## Two denominators — DON'T confuse them (this bit numbers)

The Critic/loop numbers come in two frames. Keep the paper internally consistent:

- **Per Critic verdict** (5,825 verdicts): base rate wrong 0.181 (≈82/18), detection
  0.502, false-alarm 0.132, φ +0.358 (+0.69 on `edge_existence`). fixed 130, broke 243
  (revision *events*). Conditional: 12% of wrong fixed, 5% of correct broke, ~5:1.
- **Per debate instance** (5,400 instances, Proposer's 1st answer → final): Proposer
  86.4% correct / 13.6% wrong; Final 84.3% / 15.7%; **fixed 96, broke 209, net −113**.
  Conditional: **13%** of wrong fixed, **4%** of correct broke, ~**6:1**. Error rate
  13.6% → 15.7% (+2.1%).

§4.5 and **Figure 2 use the instance frame** (86/14, 96 fixed / 209 broken). The
appendix per-cell Critic table uses the verdict frame (detection/FA/φ, pooled −113).
Both are correct; a "Proposer→Final" flow is inherently per-instance.

Regenerate instance-level flow with `scratchpad/flow.py` pattern:
`PYTHONPATH=. .venv/bin/python` loading `gedebate.eval.diagnostics.debate_views` over
the six dirs and summing `turn1_correct`×`final_correct`.

## Data / regeneration (from paper-handoff.md)

Six run dirs, always `--prompt-version v3`:
`results/llama70b-{main,seed11,seed13}` + `results/llama70b-v3-{main,seed11,seed13}`.
```
.venv/bin/python scripts/show_results.py <six dirs> --prompt-version v3 --fragility --by-seed
.venv/bin/python scripts/debate_diagnostics.py <six dirs> --prompt-version v3
```
`results/` is gitignored (local only).

---

## Current paper structure

- Abstract; 1 Introduction; 2 Related Work; 3 Setup
- 4 Results and Analysis: 4.1 fragility large · 4.2 debate raises accuracy/narrows gap
  · 4.3 gain is reasoning not loop · 4.4 compute-matched vote beats debate · 4.5 why
  debate fails: precision not detection
- 5 Discussion · 6 Limitations and future work · 7 Conclusion
- **Appendix A Prompts** (`sec:prompts`) · **Appendix B Per-cell results** (`sec:percell`)
  — headings prefixed via `\renewcommand{\thesection}{Appendix~\Alph{section}}`.

Figures/tables:
- Fig 1 `fig:fragility` — spread-by-stage line plot (§4).
- **Fig 2 `fig:flow`** — Proposer→Final Sankey (§4.5), instance-level, reference-style
  (thin node bars, pastel source-colored flows, "209 broken"/"96 fixed" on crossing
  bands, INITIAL/FINAL/NET error-rate summary row). `\resizebox{0.62\textwidth}` in a
  `figure*`. Colors: `cCorrect/cWrong/cFinalC/cFinalW` (defined in preamble).
- Table 1 `tab:example` (graph + serializations); Table 2 `tab:fragility`; Table 3
  `tab:main` (with/without edge-existence control).
- Appendix B: one `table*` float holding **Table 4 `tab:percell-critic`** then
  **Table 5 `tab:percell-acc`** (bundled so they stay together).
- NOTE: the old pooled "Critic diagnostic" Table 4 in §4.5 was **removed**; the
  per-cell Critic table's Pooled row now carries those numbers.

Key numbers already in the paper (verified): fragility 95.8%→48.8% (intro);
ordering baseline 0.793 < debate 0.843 < turn1 0.864 < vote 0.875; vote +0.032
(p<0.001) at 1.07× tokens; CoT +0.071 / loop −0.021.

## Done this session
- §4.4 rewritten (direct claims; "aggregation wins"; removed "destructive").
- §4.5 rewritten to precision-not-detection with instance-level numbers; removed the
  pooled Critic Table 4; a pie+bar figure was built then discarded in favor of the
  Sankey (Fig 2).
- Discussion: retitled "The asymmetry holds — the loop still loses"; dropped the
  detection×frac formula and the checkability/lesson closer; retitled the vote
  paragraph "Reasoning does the work; neither extra structure earns its compute."
- Limitations bullets 1–2 rephrased.
- Appendix: swapped order (Prompts=A, Per-cell=B), bundled the two per-cell tables,
  "Appendix A/B" headings.

## Open items / not yet done
- **Conclusion typos:** "introdduces" → "introduces"; comma splice "The Critic
  genuinely detects errors, it loses because…" → semicolon or "but". (Flagged, user
  hasn't asked to fix.)
- "scaffold" still appears in the Abstract ("reasoning scaffold") — earlier flagged.
- Related Work has a TODO comment: `% ADD: a GraphQA / LLM-graph-reasoning survey…`.
- Decision still nominally open: Fig 2 uses honest 86/14 (Proposer accuracy); user's
  reference image used 82/18 (verdict base rate). We kept 86/14. Can switch to 82/18
  relabeled "Answers judged" if the user prefers.

## Memory (persists across chats)
See `MEMORY.md`: wait-for-runs-to-finish, short-commit-messages,
regenerate-prompt-snapshots, paper-narrative-option-b, negative-result-framing.
