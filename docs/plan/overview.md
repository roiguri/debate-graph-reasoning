# Workplan

Living tracker for _Does Debate Improve Encoding-Fragile LLM Graph Reasoning?_
See [project_proposal.md](../project_proposal.md) for the research question and
[grading_criteria.md](../grading_criteria.md) for what the final writeup is judged on.

## Working principles

- **Vertical slices.** Every phase after P1 produces a runnable, scoreable
  result. Build helpers as they're needed, not upfront.
- **Baseline-first.** Conditions are added one at a time and proven before the
  next: baseline → majority-vote → debate.
- **Resumable by default.** The cluster's `studentkillable` partition kills
  jobs; write per-instance results incrementally and skip completed work on
  restart.
- **Matched compute is the core claim.** Track total generated tokens
  (including the Critic's) everywhere, so conditions compare at equal budget.
- **Share cluster resources.** Limited memory and limited concurrent jobs per
  student — keep jobs modest and shard the run matrix so the quota is used fully.

## Environment (from the sibling `nlp` repo's setup)

- **Inference:** in-process HuggingFace Transformers (not vLLM). Model loaded
  directly in the run process. Generated-token counts come from tokenizer output
  length.
- **Cluster:** TAU CS Slurm, `studentkillable` partition, RTX 2080 (11GB) nodes.
  Small open-weight instruct models (Qwen / Llama 7-8B, or smaller for smoke).
- **Paths:** conda env + HF cache live on a course netapp path, not home. Driven
  by a `NETAPP` variable via `slurm/_activate.sh` + `slurm/setup_env.sh`. The
  real netapp path (contains a username) stays out of git — placeholder/env only.
- **Results:** local JSON/CSV per run.

## Phases

| Phase | Slice | Delivers | GPU? | Status |
|---|---|---|---|---|
| [P0](p0-env.md) | Cluster env + minimal repo | Slurm setup for this course's netapp; a "model loads" smoke job passes | yes | ☐ |
| [P1](p1-data.md) | Data generation helpers | Graph generators + 3 encoders + NetworkX ground truth, unit-tested | no | ☐ |
| P2 | Baseline, one task × one encoding, end-to-end | Model wrapper + prompt + answer parser + scorer + JSON results + slurm run | yes | ☐ |
| P3 | Baseline across full 3×3 matrix + analysis | Reproduce encoding-fragility (checkpoint on the premise) | yes | ☐ |
| P4 | Majority-vote condition | N-sample + vote + token accounting; compare to baseline at matched compute | yes | ☐ |
| P5 | Debate condition (Proposer–Critic) | Structured trace, claim extraction, per-claim verification, revision loop, stopping rule | yes | ☐ |
| P6 | Matched-compute comparison + final analysis | Accuracy tables, cross-encoding variance (std, max-min), worst-encoding lift | no | ☐ |
| P7 | Writeup | Paper against grading criteria | no | ☐ |

Notes:
- **P2 is the keystone** — it fixes the results-file schema and answer-parsing
  contract that P4/P5 reuse. Do it carefully.
- Each GPU phase carries a tiny smoke/pilot pass (few graphs, 1 model) before
  spending real cluster time — mirrors the `nlp` repo's `smoke → pilot` habit.

## Tasks & encodings (scope, from proposal §2.2)

- **Tasks:** edge existence, connectivity, node degree.
- **Encodings:** adjacency (integer nodes, parenthesized edges), incident
  (integer nodes, NL neighbor lists), friendship (named nodes, relational
  sentences).
- **Conditions:** baseline (1 zero-shot), majority-vote (N samples + vote),
  debate (Proposer–Critic).
