# Research Strategy Experiment — Protocol Specification

Status: planned after Research Kernel MVP

## 1. Hypothesis

Under equal source, model, token, fetch, slot, and time budgets, structured multi-agent
research with an optimization policy can produce greater blind evidence-backed
understanding of a seed paper than random or greedy reference traversal.

## 2. Prerequisites

- Research Kernel MVP definition of done is satisfied.
- Frozen corpus execution is deterministic.
- Evaluator stability has been measured.
- Random and greedy baselines are implemented.
- The event schema and reconstruction checks are passing.

## 3. Primary task

Use one seed paper and one explicit question. Freeze the seed, first-level references,
selected second-level references, parsed chunks, and provider metadata into a versioned
experiment bundle.

The first recommended question is:

> How do extracellular fields originate and affect neuronal computation, according to
> the seed paper and its most relevant supporting or challenging references?

The task and corpus must be finalized before viewing comparative results.

## 4. Independent variables

### Controller study

- random
- greedy relevance
- diversity-aware greedy
- contextual UCB
- ACO

### Agent study

- single generalist
- four role-specialized agents with two active slots

Suggested roles are mechanism, methodology, skeptic, and synthesis. Prompts share the
same output contract and differ only in declared role and retrieval profile.

### Memory study

- structured notebook only
- shared retrieval plus structured notebook
- shared corpus with private retrieval profiles plus structured notebook

Do not vary controller, agent structure, and memory in one initial factorial run. Run
the studies sequentially to preserve interpretability and cost.

## 5. Controlled variables

- frozen corpus and chunking;
- model and model parameters;
- prompt/template versions;
- evaluator versions and rubric;
- total input/output tokens;
- provider-equivalent fetch budget;
- wall-clock ceiling;
- agent and provider slots;
- stop conditions;
- random-seed set;
- answer format.

## 6. Runs

- Use at least five reproducibility seeds per condition for the initial engineering
  experiment.
- Increase sample size only after measuring variance and evaluator agreement.
- Randomize condition and answer ordering presented to judges.
- Preserve failed and partial runs; do not silently rerun them as successes.

## 7. Evaluation

### Primary metric

Blind final evidence-backed understanding score using the versioned rubric.

### Secondary metrics

- delta quality per 1,000 tokens;
- delta quality per source fetch;
- time to quality threshold;
- supported claim coverage;
- unsupported claim rate;
- contradiction discovery and resolution;
- source and semantic diversity;
- duplicated work across agents;
- judge disagreement;
- run failure rate.

### Human check

At least one domain-capable human reviewer inspects a randomized subset for citation
correctness and obvious evaluator reward hacking before strong product claims are made.

## 8. Analysis

- Report every run, not only the best run.
- Report mean, median, dispersion, and paired differences where seeds align.
- Keep deterministic integrity metrics separate from subjective judge metrics.
- Inspect whether higher scores arise from verbosity, unsupported certainty, or leakage.
- Treat lack of improvement as a valid result.
- Document deviations from the preregistered protocol.

## 9. Memory ablation interpretation

The private-RAG hypothesis is supported only if private retrieval profiles improve blind
quality or efficiency beyond a shared corpus plus structured notebooks. Reduced overlap
alone is insufficient if evidence coverage or correctness falls.

## 10. Optimization interpretation

An optimization policy is useful only if it beats random and greedy controls under equal
budgets across multiple seeds. A higher single-run final score is insufficient.

## 11. Replay deliverable

Every experiment produces a replay bundle containing resolved configuration, frozen
corpus version, event log, snapshots, final answer, evaluations, policy decisions, costs,
and hashes. The first presentation may be static HTML. Interactive slider and video
rendering are later views over the same bundle.

