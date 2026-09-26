# Research Control System — Product and Architecture Specification

Status: proposed

## 1. Purpose

Research Explorer helps a user understand a scientific paper or answer a focused
question about it under finite budgets for tokens, provider calls, wall-clock time,
and context. The system must decide which evidence to inspect next, which research
agent should inspect it, and when further exploration has diminishing value.

The system is not primarily a narrative generator. It is a reproducible research
harness with pluggable exploration policies. Narratives and answers are generated
views over an evidence-backed knowledge state.

## 2. Problem statement

Given:

- a seed paper;
- an explicit research question;
- a finite or bounded source graph;
- execution budgets;
- one or more research agents;
- an evaluation process that is noisy and only approximates true understanding;

select research actions that maximize useful knowledge gained per unit of cost.

The latent objective is not directly observable. The operational objective is:

```text
maximize  evaluated knowledge gain
          ------------------------
          tokens + fetches + time

subject to evidence traceability, safety, and budget constraints.
```

Absolute quality and marginal gain must both be retained. Controllers learn from or
react to marginal gain, while users consume the best validated final state.

## 3. Product principles

1. Evidence before prose. Claims must retain links to exact evidence.
2. Structured state before narrative. Prose is a projection, not canonical memory.
3. Shared sources, differentiated viewpoints. Agents share immutable evidence while
   retaining private notebooks and retrieval profiles.
4. Evaluation is part of the experiment. Scores, judge prompts, and disagreements are
   versioned artifacts rather than ground truth.
5. Optimization is replaceable. ACO, greedy, random, and bandit policies use the same
   harness and action interface.
6. Every decision is replayable. The run is reconstructed from configuration, events,
   evidence hashes, and snapshots.
7. Budgets are first-class. Provider calls, tokens, time, context, and concurrency are
   explicit constraints.
8. Uncertainty is preserved. Consolidation must not erase disagreement or negative
   evidence.

## 4. Scope

### 4.1 In scope

- Understanding a seed paper in relation to an explicit question.
- Backward exploration of references and forward exploration of citing papers.
- Shared evidence storage and provenance.
- Structured claims, contradictions, and open questions.
- Private structured agent notebooks.
- Parameterized pipeline stages, roles, budgets, and provider slots.
- Deterministic and LLM-assisted evaluation.
- Pluggable exploration policies.
- Append-only event history and state snapshots.
- Human-readable run replay.
- Controlled comparisons between memory and optimization strategies.

### 4.2 Not initially in scope

- Unbounded autonomous web research.
- Treating an LLM judge as objective ground truth.
- One physical vector database per agent.
- Online training of model weights.
- Monte Carlo Tree Search before useful rollouts can be simulated.
- Fully parallel writes to a shared SQLite connection.
- A polished video editor in the first vertical slice.

## 5. System boundaries

The system has two planes.

### 5.1 Research plane

The research plane acquires and interprets evidence. It contains providers, the
evidence store, agent context assembly, notebooks, the claim ledger, evaluation, and
event recording.

### 5.2 Control plane

The control plane selects the next research actions. It sees candidate actions,
budgets, slots, prior evaluations, and a summarized research state. It must not own
provider-specific fetching or mutate evidence directly.

```text
                           ExplorationPolicy
                                  |
                            ResearchAction[]
                                  v
Sources -> EvidenceStore -> Scheduler -> ResearchAgent
                ^             |              |
                |             |              v
                |             |       PrivateNotebook
                |             v              |
                +------ EvaluationMeeting <--+
                              |
                      Evaluation + delta
                              |
                              v
                         ResearchEvent log
```

## 6. Canonical domain model

### 6.1 Research objective

A run must have a seed identifier and an explicit question. Broad topic exploration
may be supported later, but the first production contract requires a question.

Required fields:

- `seed_paper_id`
- `question`
- optional `scope_constraints`
- budgets and stop conditions
- pipeline, prompt, model, evaluator, and policy versions
- reproducibility seed

### 6.2 Evidence

`EvidenceDocument` represents a fetched source. `EvidenceChunk` represents an addressable
passage or metadata unit. Evidence is immutable by content hash. Later provider refreshes
create new versions rather than silently rewriting evidence used by a completed run.

Every evidence record includes:

- canonical paper identity and provider identity;
- source location, section, page, or structured field when available;
- content hash;
- acquisition event and timestamp;
- provider response status;
- extraction method and version;
- license/access metadata when available.

### 6.3 Knowledge state

The canonical state contains:

- `Claim`: a falsifiable or explanatory proposition.
- `ClaimEvidence`: support, contradiction, or qualification linked to evidence.
- `OpenQuestion`: a knowledge gap with priority and resolution status.
- `Contradiction`: incompatible claims or evidence requiring adjudication.
- `Term`: important terminology and definitions.
- `MethodNote`: assumptions, methods, limitations, and threats to validity.
- `CandidateAction`: a possible research operation and its predicted value/cost.
- `ResearchStateSnapshot`: a versioned projection at an event sequence.

Claims include confidence, author, lifecycle status, and provenance. Confidence is a
calibrated belief, not a substitute for evidence.

### 6.4 Agent notebook

Each agent has a private, structured notebook containing:

- current thesis;
- accepted and disputed claims;
- unresolved questions;
- contradictions noticed;
- intended next actions and their reasons;
- compacted episodic history;
- retrieval profile and role.

The notebook may be rendered to Markdown, but its stored representation is typed and
validated. Free-form notes may be attached without becoming canonical claims.

### 6.5 Agent brief

At the end of a research turn, every agent returns the same contract:

1. action performed;
2. new or changed claims;
3. evidence for each change;
4. effect on prior understanding;
5. contradictions or weaknesses found;
6. remaining uncertainty;
7. proposed next actions;
8. shareable colony knowledge.

Malformed fields degrade locally and are recorded. They must not terminate the run.

## 7. Memory and context

### 7.1 Storage model

The default architecture uses:

- one shared immutable evidence corpus;
- one shared claim ledger;
- one private notebook namespace per agent;
- one shared embedding/index layer where practical;
- agent-specific retrieval queries, filters, and reranking.

Agent isolation is logical, not a duplicated copy of every source.

### 7.2 Context assembly

Context is assembled per action rather than filled to a fixed percentage. The packer
prioritizes:

1. objective and output contract;
2. current agent thesis and open questions;
3. relevant claims and contradictions;
4. exact evidence passages for the action;
5. compacted decision history.

Input occupancy, reserved output tokens, tool-result allowance, truncation, and omitted
items are emitted as events. A 50–65% input occupancy target is an initial default, not
a product invariant.

### 7.3 Memory modes

The harness eventually supports comparable modes:

- `structured_notebook`
- `shared_rag`
- `shared_corpus_private_retrieval`

All modes implement one retrieval/context interface so they can be evaluated without
changing the research or evaluation contracts.

## 8. Evaluation

### 8.1 Evaluation dimensions

- question relevance;
- seed-paper coverage;
- mechanistic depth;
- methodological understanding;
- evidence traceability;
- counterevidence and alternative explanations;
- uncertainty calibration;
- novelty of the latest wave;
- redundancy;
- efficiency.

### 8.2 Evaluator composition

Evaluation combines:

- deterministic integrity checks;
- contextual LLM evaluation;
- one or more blind external judges;
- optional peer review;
- optional self-assessment.

Blind judges receive the objective, allowed evidence, rubric, and candidate state. They
do not receive agent identity, controller scores, or the exploration path.

### 8.3 Output contract

An evaluation contains dimension scores, overall quality, previous quality,
`delta_quality`, missing knowledge, unsupported claims, contradictions, recommended
questions, judge rationale, and evaluator version.

The controller-facing reward begins as:

```text
reward = delta_quality
         - token_cost_weight * tokens
         - fetch_cost_weight * provider_calls
         - time_cost_weight * elapsed
         - redundancy_weight * redundant_work
         - unsupported_weight * unsupported_claims
```

Every component and weight is configurable and recorded.

### 8.4 Evaluation safeguards

- Preserve raw judge outputs.
- Validate score ranges and shapes.
- Record inter-judge disagreement.
- Randomize ordering in comparisons.
- Require evidence-linked criticism where possible.
- Do not train or tune on the final test corpus.
- Report deterministic and LLM metrics separately.

## 9. Meeting protocol

Multi-agent pipelines use a structured meeting:

1. `exposition`: agents publish briefs independently.
2. `cross_examination`: assigned agents confirm, challenge, or qualify claims.
3. `consolidation`: a synthesizer updates the ledger without erasing disputes.
4. `blind_evaluation`: external evaluators score the consolidated state.
5. `control_feedback`: missing knowledge and marginal gain return to the controller.

The pipeline may parameterize the number of reviewers, aggregation rule, visibility of
private notes, and whether evaluation occurs per turn or per wave.

## 10. Scheduling and provider capacity

The scheduler enforces:

- total active agent slots;
- per-provider concurrency and rate limits;
- global and per-agent budgets;
- exclusive short-lived claims on active fetches;
- retry eligibility for transient failures;
- cancellation and timeout semantics.

A frontier item is removed only for a definitive absence, invalid identity, or explicit
policy decision. Circuit-open, rate-limit, timeout, and retry exhaustion preserve the
candidate for bounded retry.

## 11. Exploration policy interface

Policies consume the same state and return typed actions. The initial policy family is:

- random control;
- greedy relevance;
- diversity-aware best-first;
- contextual UCB;
- ACO;
- optional ACO/UCB hybrid.

Policy decisions include candidate scores, chosen actions, rejected alternatives, model
parameters, random seed, and reason codes in the event log.

The first hybrid candidate score is conceptually:

```text
expected_gain + exploration_bonus + pheromone_value
- redundancy_penalty - estimated_cost
```

## 12. Pipeline configuration

A run configuration must be sufficient to reproduce the pipeline shape:

```yaml
pipeline:
  name: paper-understanding-v1

objective:
  seed_paper_id: 10.1038/nrn3241
  question: How do extracellular fields originate and affect neuronal computation?

budgets:
  provider_fetches: 100
  input_tokens: 400000
  output_tokens: 100000
  wall_time_minutes: 90

slots:
  agents: 3
  providers:
    openalex: 2
    semantic_scholar: 1
    arxiv: 1

memory:
  mode: structured_notebook
  input_context_target: 0.60

controller:
  policy: greedy
  random_seed: 42

evaluation:
  interval: wave
  blind_judges: 1
  rubric_version: v1
```

Unknown configuration keys fail validation. Resolved defaults are stored with the run.

## 13. Event sourcing and replay

Every material action appends a `ResearchEvent` with:

- run and sequence identifiers;
- logical and wall-clock time;
- actor, wave, turn, and action;
- input and output artifact identifiers;
- reason and controller decision;
- token, fetch, and time cost;
- state-before and state-after hashes;
- schema and implementation versions.

Event types include run lifecycle, evidence acquisition, context assembly, agent action,
claim mutation, notebook mutation, evaluation, controller decision, budget mutation,
provider failure, retry, snapshot, and artifact generation.

Periodic snapshots accelerate playback but are derivable projections. Replaying events
to a sequence number must reproduce the same logical research state.

The initial replay may be a CLI or static HTML timeline. A later UI adds a slider showing
state, evidence, agent beliefs, costs, scores, and decisions at each sequence.

## 14. Reproducibility

Each run records:

- resolved configuration;
- repository revision;
- model/provider identifiers;
- prompt and rubric hashes;
- random seeds;
- retrieved content or immutable content hashes;
- agent briefs and raw judge responses;
- all policy decisions;
- environment and schema versions.

Live providers prevent perfect physical reproducibility. The system must support a frozen
corpus/replay mode for deterministic tests and policy comparisons.

## 15. Security and privacy

- Redact secrets and sensitive query parameters from logs and exceptions.
- Treat paper content and provider metadata as untrusted input.
- Preserve provenance for generated claims.
- Bind replay to loopback by default and warn or require authentication off-loopback.
- Do not embed credentials in event artifacts or exported experiment bundles.
- Apply retention controls to full provider responses and model transcripts.

## 16. Delivery roadmap

### Milestone 0 — trustworthy foundation

Resolve the verified Convoy findings Q-1 through Q-9, restore CI on `master`, and make
type checking executable.

### Milestone 1 — vertical research kernel

Implement structured objective, evidence references, claims, questions, notebook, brief,
evaluation, event contracts, snapshots, and a single-agent closed loop with a greedy
policy.

### Milestone 2 — multi-agent harness

Add roles, shared ledger, private notebooks, configurable slots, meeting stages, and
consolidation.

### Milestone 3 — policy laboratory

Add random, diversity-aware, contextual UCB, and ACO policies behind the common
interface. Add frozen-corpus comparisons.

### Milestone 4 — memory laboratory

Add shared retrieval and private retrieval profiles. Compare them against notebook-only
memory.

### Milestone 5 — human replay

Build the interactive slider/timeline and exportable research-process presentation.

## 17. Product success criteria

The system is successful when it can demonstrate, under a fixed budget and frozen corpus,
that a research configuration improves blind evidence-backed understanding over random
and greedy baselines, while the entire result can be audited and reconstructed.

