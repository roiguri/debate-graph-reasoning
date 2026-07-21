# Does Debate Improve Encoding-Fragile LLM Graph Reasoning?

> **TL;DR.** We propose a Proposer-Critic debate framework to test whether active verification can improve the reliability of LLMs on graph reasoning tasks across different text encodings. By comparing this method against a majority-vote ensemble at a matched compute budget, we analyze if debate effectively mitigates the encoding-fragility of LLMs.

## 1 Background & Motivation

Large language models are increasingly asked to reason over graph-structured data, but a graph must first be serialized into text before an LLM can read it, and how it is serialized has an effect on accuracy. Fatemi et al. [2024] show that on basic tasks (e.g. edge existence, connectivity), zero-shot accuracy varies sharply with the encoding, the task, and the graph structure, with no single encoding dominating. For finding a node's connected neighbors, accuracy rises from 19.8% under an adjacency encoding to 53.8% under an incident encoding - a swing of over thirty points from wording alone. The learned-encoder successor GraphToken [Perozzi et al. 2024] raises absolute performance but does not eliminate this task-dependent sensitivity.

### 1.1 Problem Statement

LLM graph reasoning is encoding-fragile: on the same graph and task, accuracy can swing sharply depending on how the graph is serialized, so performance is unreliable. We ask whether a multi-agent Proposer–Critic debate can improve accuracy across encodings at matched compute, and whether any improvement comes from the framework of debate, rather than just from combining multiple model runs.

### 1.2 Why it Matters

As LLMs are deployed on relational data (knowledge graphs, code, networks), reliability that depends on serialization choices is a practical liability. Multi-agent debate is proposed as a general mechanism for improving and overseeing model reasoning, and its promise rests on an assumption of verification asymmetry. The assumption is that checking a claim is cheaper than producing one [Irving et al. 2018]. Yet this is almost always asserted, not measured, because in most reasoning domains (math word problems, open-ended QA) intermediate steps cannot be cheaply and objectively checked. Graph reasoning is a setting where the asymmetry clearly exists: an atomic edge claim is either present in the encoding or not. This makes graphs a clean case for testing whether debate makes graph reasoning robust to how the graph is written. Answering this question matters wherever relational data is serialized for an LLM. Even though we are tackling basic graph tasks, those tasks are essential for more complex reasoning tasks on graphs. For example, to determine the shortest path between two nodes in a graph, we must first be able to find all the nodes that are connected to a given node.

### 1.3 Prior Work Limits

Fixed-encoding studies [Fatemi et al. 2024] and learned encoders [Perozzi et al. 2024] improve or characterize performance on graph related tasks but leave the fragility in place. Multi-agent debate [Du et al. 2023] is a natural remedy, but its value is contested: at matched inference compute it is no better than self-consistency [Huang et al. 2024; Wang et al. 2023], and it was shown that majority voting alone accounts for most of the gains typically attributed to it [Choi et al. 2025]. Critically, those null results come from tasks (e.g., math word problems) whose intermediate steps are not cheaply checkable - exactly the condition graph reasoning removes. Whether debate adds anything over aggregation remains undetermined on graphs and settling it could show whether debate is a route to overcoming encoding fragility.

## 2 Proposed Approach

### 2.1 Core Idea

We test whether Proposer-Critic debate improves LLM graph reasoning, and isolate why. A Proposer answers each task and emits a structured trace - the atomic claims supporting its answer (e.g., a path as an ordered list of edge-existence claims). A Critic then verifies these atomic claims against the raw encoding, rather than judging the final conclusion, and feeds any refuted claims back to the Proposer. The design deliberately exploits the verification asymmetry: checking whether one edge is present is far cheaper than solving the task.

### 2.2 Technical Outline

1. **Setup:** We evaluate on GraphQA [Fatemi et al. 2024] using its released encoders, over three tasks (edge existence, connectivity, node degree) and three genuinely distinct encodings: adjacency (integer nodes, parenthesized edges), incident (integer nodes, natural-language neighbor lists), and friendship (named nodes, relational sentences). Graphs are generated with known ground-truth answers.
2. **LLM Critic:** The Critic is a separate LLM agent that verifies each atomic claim against the raw encoding - for each edge claim, it is only asked whether that specific edge appears in the given text, a strictly narrower task than solving the original problem. This keeps the Critic's role aligned with the verification asymmetry: it checks one claim at a time rather than re-deriving the answer.
3. **Debate Loop and Decision Rule:** The Proposer emits a structured trace → the Critic returns any refuted claims → the Proposer revises. This repeats until the Critic finds no errors (consensus) or a preset round/token cap is hit. The final answer is the Proposer's most recent one. Because debate is variable-length, we measure its cost as total generated tokens (including the Critic's) rather than rounds, so it can later be compared against baselines at matched compute.

### 2.3 Novelty

Debate and graph reasoning are each well studied, but their intersection - and specifically graphs as a setting where the verification asymmetry that debate assumes becomes directly measurable - is not. Our contribution is not "debate on graphs" but a controlled test of whether active verification, rather than mere aggregation, accounts for any improvement.

## 3 Experimental Plan

### 3.1 Datasets

GraphQA [Fatemi et al. 2024], using its released graph generators and encoding functions. Graphs carry ground-truth answers (NetworkX), so no labeled data is required.

### 3.2 Baselines/Comparisons

Three conditions, compared at a matched token budget (total generated tokens, including the critic's):

- **Baseline:** one zero-shot answer per task and encoding. Establishes baseline accuracy and the spread across encodings.
- **Majority Vote Ensemble** [Choi et al. 2025]: N samples of the plain prompt, resolved by majority vote. This is plain extra compute with no debate - to isolate the effect of debate over pure compute increase on the results.
- **Debate** [Du et al. 2023]: the Proposer-Critic method.

Comparing these conditions in order tells us what each ingredient contributes. If majority-voting beats the baseline, that gain is from spending more compute. If debate then beats majority-voting at the same compute, that additional gain is from the debate procedure itself, not from extra sampling.

All three conditions are evaluated on every task × encoding combination, so comparisons are made per-task and per-encoding rather than aggregated.

### 3.3 Metrics

- **Primary - Accuracy:** For each task and encoding, accuracy is the fraction of graphs where the parsed final answer exactly matches the NetworkX ground truth. We report mean accuracy across the three encodings per task, with per-encoding numbers in a table.
- **Secondary - Cross-Encoding Variance:** For each task and condition, we report the spread of accuracy across the three encodings (standard deviation and max-min gap), and whether debate reduces it by lifting the worst encoding rather than lowering the best.

A null result - debate is no better than majority-vote, consistent with Huang et al. [2024] and Choi et al. [2025] - is itself a notable finding.

### 3.4 Compute

The project is inference-only with short prompts and no training required, so it is lightweight. For both the Proposer and Critic, we will use small open-weight instruct models (e.g., Qwen or Llama-family 7-8B models), which will fit within the GPU's available on the university Slurm cluster.

## References

- H. K. Choi, X. Zhu, and S. Li. 2025. Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?. In NeurIPS.
- Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch. 2023. Improving Factuality and Reasoning in Language Models through Multiagent Debate. arXiv preprint arXiv:2305.14325 (2023).
- B. Fatemi, J. Halcrow, and B. Perozzi. 2024. Talk like a Graph: Encoding Graphs for Large Language Models. In ICLR.
- J. Huang, X. Chen, S. Mishra, H. S. Zheng, A. W. Yu, X. Song, and D. Zhou. 2024. Large Language Models Cannot Self-Correct Reasoning Yet. In ICLR.
- G. Irving, P. Christiano, and D. Amodei. 2018. AI Safety via Debate. arXiv preprint arXiv:1805.00899 (2018).
- B. Perozzi, B. Fatemi, D. Zelle, A. Tsitsulin, M. Kazemi, R. Al-Rfou, and J. Halcrow. 2024. Let Your Graph Do the Talking: Encoding Structured Data for LLMs. arXiv preprint arXiv:2402.05862 (2024).
- X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and D. Zhou. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In ICLR.
