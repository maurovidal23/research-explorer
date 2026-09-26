# Research Kernel MVP — Implementation Specification

Status: ready for implementation

This specification defines the first Convoy implementation slice. It is intentionally
smaller than the complete architecture in `research-control-system.md`.

## 1. Outcome

Implement a reproducible single-agent research loop that:

1. accepts a seed paper and explicit research question;
2. acquires evidence without confusing temporary provider failure with absence;
3. maintains structured claims, evidence links, open questions, and a private notebook;
4. chooses the next reference with a replaceable greedy policy;
5. evaluates the state after every turn;
6. records sufficient events and snapshots to reconstruct the logical run;
7. emits a final evidence-backed answer as a view over the structured state.

The implementation must preserve current CLI behavior where compatible and provide a
migration path for the existing narrative-based ACO pipeline.

## 2. Required stabilization

The following verified findings are release blockers and are part of this slice.

### STAB-1 — DOI seed routing

- Detect normalized DOI and arXiv seed identifiers.
- Route DOI seeds to an enabled DOI-capable provider, preferring OpenAlex and then
  Semantic Scholar unless configuration defines an explicit order.
- Route arXiv identifiers to arXiv when enabled.
- Fail with a clear configuration error when no capable provider is enabled.
- Do not construct an unused fallback provider.

### STAB-2 — provider outcome semantics

Establish and test one explicit contract:

- a successful response returns parsed data;
- a definitive not-found/empty successful result becomes `None` or an empty collection;
- circuit open, rate limiting, server failure, timeout, network failure, and retry
  exhaustion raise a typed transient provider error at the boundary where the caller
  must decide whether to degrade or retry.

Ordinary 4xx responses must not open the circuit. Retry-exhausted 429, 5xx, and transport
failures must count toward it. Adapter search methods may degrade transient failures to
empty results only when their caller does not need to distinguish absence from failure.

### STAB-3 — frontier retention

- Definitive absence may remove a candidate.
- Transient provider failure releases the claim and retains the candidate.
- Track bounded attempt count and next eligibility or an equivalent bounded retry rule.
- Exception paths release frontier and shared-visited claims.

### STAB-4 — identity integrity

- Metadata-less candidates deduplicate by `provider:native_id`.
- Exact DOI or arXiv lookup supplies identifier evidence and can pass the confidence
  floor when no descriptive metadata exists.
- Contradictory descriptive metadata still triggers configured hard gates.

### STAB-5 — embedding preservation

Re-caching provider metadata without an embedding preserves the stored embedding.
An explicit non-null replacement may update it.

### STAB-6 — malformed LLM containment

- Validate top-level and nested JSON types before access.
- Ignore malformed individual collection entries.
- A failed agent turn is recorded and cannot terminate unrelated turns or the run.
- Claims acquired before an exception remain consistent, and active claims/locks are
  released.

### STAB-7 — secret redaction

Redact at least `api_key`, `key`, `token`, and `email` query parameters from logged URLs
and exception strings. Tests must use sentinel secrets and assert they never appear.

### STAB-8 — regression and CI coverage

- Add revert-failing tests for every stabilization requirement.
- Run CI on `master` pushes and pull requests.
- Add a mypy job or a separately visible type-check job.
- Resolve dependency/stub incompatibility so project source is actually checked on all
  supported Python versions or document and enforce a narrower supported matrix.

## 3. New domain contracts

Add typed models in a cohesive research-domain package. Names may follow repository
conventions, but the following semantics are mandatory.

### ResearchObjective

- run identifier
- seed paper identifier
- non-empty research question
- optional scope constraints
- resolved budgets
- random seed

### EvidenceRef

- canonical paper identifier
- optional chunk/section/page locator
- content hash when available
- acquisition event identifier

The MVP may continue storing full paper data in the existing graph store. It must not
duplicate complete provider documents merely to introduce `EvidenceRef`.

### Claim

- stable identifier
- proposition text
- status: proposed, supported, disputed, rejected, or superseded
- confidence in `[0, 1]`
- supporting and contradicting evidence references
- creator and creation/update sequence

A claim without evidence may exist only as `proposed`. It cannot be presented as a
supported final claim.

### OpenQuestion

- stable identifier
- question text
- priority in `[0, 1]`
- status: open, investigating, resolved, or abandoned
- related claim and paper identifiers
- resolution evidence when resolved

### AgentNotebook

- agent identifier and role
- current thesis
- claim identifiers accepted or disputed by the agent
- open-question identifiers
- planned actions with reasons
- compact history entries
- monotonically increasing revision

### AgentBrief

- action summary
- claim mutations
- evidence references
- changed understanding
- contradictions
- remaining uncertainty
- proposed next actions
- shareable findings

### ResearchEvaluation

- rubric version
- dimension scores
- overall and previous quality
- delta quality
- missing knowledge
- unsupported claim identifiers
- contradictions
- recommended next questions
- evaluator identity/version and raw-artifact reference

### ResearchEvent

- run identifier and monotonic sequence
- event type
- actor, wave, and turn where applicable
- input/output artifact identifiers
- structured payload
- token/fetch/time costs where known
- previous/current state hashes for snapshot-producing transitions
- wall-clock timestamp
- schema version

## 4. Storage and reconstruction

- Reuse SQLite and the existing trace facilities where practical.
- Store research events append-only with a uniqueness constraint on run and sequence.
- Persist structured knowledge state or mutations transactionally with the associated
  event.
- Create a snapshot at run initialization and after each evaluated turn.
- Implement `reconstruct(run_id, sequence=None)` returning the logical objective,
  knowledge state, notebook, budgets, and latest evaluation at that point.
- Reconstruction must not call external providers or LLMs.
- Reconstructing the same event range twice produces equal serialized state.
- Existing replay records remain readable or receive an explicit additive migration.

## 5. Single-agent closed loop

Add a versioned pipeline mode rather than silently replacing the existing ACO mode.
Suggested configuration name: `research_kernel`.

One turn performs:

1. inspect current objective, state, notebook, and budget;
2. enumerate eligible candidate references;
3. ask the configured policy for one action;
4. record the decision and predicted value/cost;
5. fetch and integrate evidence;
6. request a typed agent brief;
7. validate and apply allowed mutations;
8. evaluate the resulting research state;
9. record costs, evaluation, and snapshot;
10. stop or continue according to budget and convergence rules.

The loop stops on configured budgets, no eligible actions, user cancellation, or
convergence. A transient provider failure consumes only the configured attempt cost and
does not masquerade as successful research.

## 6. Policy interface and baseline

Introduce a policy protocol independent of provider and persistence implementations:

```python
class ExplorationPolicy(Protocol):
    async def select_actions(
        self,
        state: ResearchState,
        candidates: list[CandidateAction],
        budget: BudgetState,
        slots: SlotState,
    ) -> list[ResearchAction]: ...

    async def observe(
        self,
        actions: list[ResearchAction],
        evaluations: list[ResearchEvaluation],
    ) -> None: ...
```

Equivalent signatures consistent with existing types are acceptable. The important
constraint is that the controller does not fetch providers or write evidence itself.

The MVP implements deterministic greedy selection with explicit tie-breaking and an
injectable random seed. It records all considered candidates and reason codes.

Existing ACO logic need not be rewritten onto this interface in this slice, but the
interface must not prevent that migration.

## 7. Context and prompts

- Require the explicit research question in every research and evaluation prompt.
- Generate prompts from structured models using safe serialization.
- Provide the agent with the current thesis, relevant claims/questions, selected source,
  and output schema.
- Do not provide the complete event history when a compact notebook is sufficient.
- Record prompt template version, selected context item identifiers, approximate input
  tokens, and omitted/truncated items.
- Reserve configurable output capacity rather than filling a fixed percentage of the
  model window.

The MVP uses structured notebook memory only. Vector retrieval changes are outside this
slice.

## 8. Evaluation MVP

Implement a composite evaluator with two separable outputs.

### Deterministic integrity

- supported claims have at least one valid evidence reference;
- referenced papers/chunks exist;
- confidence and score ranges are valid;
- duplicate normalized claims are identified;
- final answer citations resolve;
- budget accounting is internally consistent.

### LLM rubric

Evaluate question relevance, coverage, mechanistic depth, methodological understanding,
evidence traceability, counterevidence, uncertainty calibration, novelty, and redundancy.

The evaluator receives no controller score or policy name. Invalid judge output produces
a recorded evaluator failure and deterministic results remain available. It does not
abort the research run.

The initial aggregate formula and weights are configuration. Both component scores and
aggregate are stored. `delta_quality` compares like-for-like rubric versions.

## 9. Final answer

Generate the final answer from the structured state. It must:

- directly address the research question;
- cite evidence identifiers that resolve through the store;
- distinguish supported conclusions, plausible interpretations, and unknowns;
- disclose important contradictions and limitations;
- avoid presenting unsupported proposed claims as fact.

Persist the answer as an artifact linked from the terminal run event.

## 10. Configuration

Additive typed configuration must cover:

- pipeline mode;
- objective/question requirement;
- policy and seed;
- fetch/token/time/turn budgets;
- context input target and output reserve;
- evaluator enablement, rubric version, weights, and interval;
- transient retry eligibility/attempts;
- snapshot interval.

Existing configuration files must continue loading with documented defaults. Unknown
keys follow the repository's chosen validation policy consistently.

## 11. Observability

Structured logs include run, turn, actor, event type, outcome, and costs but never secret
URLs or raw credentials. The following are visible in replay artifacts:

- controller choice and alternatives;
- provider outcome classification;
- knowledge-state mutations;
- notebook revision;
- evaluation components and delta;
- budget before and after;
- terminal reason.

## 12. Testing strategy

Tests use frozen fixtures and fake providers/LLMs by default. No unit or integration test
depends on live academic APIs.

Required test groups:

1. Provider outcome and redaction regression tests.
2. DOI/arXiv routing tests.
3. Frontier retention and bounded retry tests.
4. Candidate deduplication and identifier-only resolution tests.
5. Embedding preservation tests.
6. Malformed agent/evaluator JSON shape tests.
7. Domain model validation and serialization tests.
8. Event sequence, transaction, and deterministic reconstruction tests.
9. Greedy policy determinism and tie-breaking tests.
10. End-to-end frozen-corpus single-agent loop.
11. Final-answer evidence integrity test.
12. Migration/backward compatibility tests for existing stores and config.

The end-to-end fixture should contain one seed, several references, one transient fetch
failure, one contradictory source, and enough evidence to resolve at least one question.

## 13. CLI acceptance path

Provide a documented command or configuration profile that runs the vertical slice. A
representative interface is:

```bash
uv run research-explorer explore 10.1038/nrn3241 \
  "How do extracellular fields originate and affect neuronal computation?" \
  --pipeline research-kernel \
  --config config/profiles/kernel_quick.toml
```

Exact option placement may follow the current Typer interface. The checked-in quick
profile uses bounded frozen/fake dependencies for automated tests and a documented live
profile for manual smoke testing.

## 14. Definition of done

- The documented DOI command selects a capable provider.
- All verified Q-1 through Q-9 failure paths have regression tests and fixes.
- A frozen-corpus single-agent run completes from objective to final answer.
- Every supported final claim resolves to stored evidence.
- A transient failure retains its candidate and is visibly classified.
- Malformed agent or evaluator output cannot terminate the run.
- Reconstruction at every stored snapshot equals the serialized live logical state.
- Existing ACO mode and existing data remain usable.
- Ruff, pytest, and mypy pass in the supported CI matrix.
- CI triggers on `master`.
- No sentinel credential appears in logs, exceptions, trace events, or artifacts.

## 15. Deferred follow-up

The following belong to later milestones and must not be partially invented in this
slice:

- multi-agent exposition/cross-examination meetings;
- shared claim-ledger write arbitration between concurrent agents;
- contextual UCB and hybrid policy implementation;
- per-agent retrieval/RAG experiments;
- interactive replay slider/video rendering;
- authenticated remote replay service;
- parallel SQLite write support.

