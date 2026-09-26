# Research Kernel Evidence Reliability — Implementation Specification

Status: ready for implementation

This specification is a follow-on to `research-kernel-mvp.md`. It is based on the
observed live run `957a180ced33`, which completed operationally but published a
claim that its own evaluator identified as unsupported.

## 1. Outcome

Make the research-kernel answer trustworthy under incomplete retrieval and fallible
LLM output. A completed run must never promote a claim merely because cited paper
identifiers exist. Evaluator findings must affect claim state, evidence references
must resolve to acquired content, and an exhausted citation frontier must trigger a
bounded search expansion before the run gives up.

The change also makes NaN `glm-5.3-flash` the default judge model.

## 2. Release-blocking requirements

### REL-1 — Structured evaluator claim verdicts

- Replace free-text-only unsupported-claim reporting with structured claim verdicts
  keyed by an existing claim identifier.
- Each verdict records the claim id, an assessment of `supported`, `unsupported`, or
  `contradicted`, and a concise reason.
- Reject or ignore verdicts for unknown claim identifiers without aborting the run.
- Preserve aggregate missing-knowledge, contradiction, and recommended-question
  fields for reporting.
- Malformed evaluator output remains contained and produces an explicit evaluator
  failure rather than mutating claim state.

### REL-2 — Evaluator verdicts govern publishable state

- After a successful evaluation, apply structured verdicts deterministically.
- An `unsupported` verdict downgrades a supported claim to proposed and prevents it
  from appearing under supported conclusions.
- A `contradicted` verdict moves a claim to disputed only when contradiction evidence
  resolves; otherwise it downgrades the claim to proposed and records the reason.
- A `supported` verdict does not manufacture evidence or upgrade an evidence-free
  claim.
- Emit an auditable event describing every evaluator-driven status transition.
- Finalization must run a final integrity/evaluation gate over the answer or enforce
  the latest successful verdicts equivalently.

### REL-3 — Evidence provenance and entailment boundary

- Supporting and contradicting references on claims must resolve to an acquired
  `EvidenceRef` in research state, not merely to a paper row in the graph store.
- Canonicalize accepted claim references from matching acquired evidence so content
  hash and acquisition event are retained.
- Deterministic integrity fails supported/disputed claims whose references do not
  resolve to acquired evidence.
- Evidence references should support an optional locator or short source excerpt.
  Retain supplied provenance and expose it to evaluation and answer rendering without
  duplicating full documents.
- Citation integrity and claim support are separate checks. Passing citation
  resolution alone must not imply that the source entails the claim.

### REL-4 — Internally coherent final answers

- Do not emit a supported conclusion when the latest evaluation names the same claim
  as unsupported or when a limitation directly negates it.
- If no claim survives the gate, state that evidence is insufficient and render
  useful limitations, open questions, and acquired citations without inventing a
  conclusion.
- Mark conclusions provisional when evidence volume or cross-setting coverage is
  below a documented threshold.
- Deduplicate limitations and avoid limitations contradicted by stored source
  metadata or excerpts.

### REL-5 — Query-driven frontier expansion

- Add a bounded search action to the research policy/controller contract.
- When the citation frontier is empty, budget remains, and important missing
  knowledge or recommended questions remain, search enabled providers using the
  research question and evaluator-recommended questions.
- Search results enter the same deduplicated candidate pool and remain subject to
  normal budget, visited, provider-routing, and evidence rules.
- Avoid repeating an equivalent query in one run and cap queries/results through
  typed kernel options.
- Search failure is classified as transient or definitive using the same provider
  outcome semantics as evidence acquisition.
- `no_eligible_actions` is valid only after citation candidates and permitted search
  expansions are exhausted.

### REL-6 — Reference-mapping resilience

- Request schema-constrained reference-mapper output where the configured LLM API
  supports it.
- Retry malformed JSON only within a small explicit bound.
- Preserve explicit identifier extraction as a fallback and record mapping failure,
  fallback use, and recovered candidate count in events.
- A mapping failure must not silently imply that the bibliography was fully
  traversed.

### REL-7 — Correct arXiv outcome classification

- The arXiv XML request path must honor `strict_outcomes()`.
- Retry-exhausted 429/5xx, timeouts, network errors, circuit-open errors, and malformed
  successful payloads that cannot establish absence must surface as typed transient
  failures.
- Only a valid successful response with no matching entry, or a definitive not-found
  response, may become `ABSENT`.
- Prove transient arXiv failures remain eligible for bounded retry and do not become
  `evidence_absent`.

### REL-8 — Terminal reason and incomplete-research semantics

- Distinguish `frontier_exhausted` from policy stop, convergence, budget exhaustion,
  provider failure, and successful evidence sufficiency.
- Completion with low coverage or no supported claims is a completed run with an
  evidence-insufficient answer, not a successful substantive conclusion.
- Include terminal reason and evidence sufficiency in the rendered answer and
  structured final-answer event.

### REL-9 — Complete accounting and reconstructable final state

- Persist a final snapshot after success, absent, transient, and finalization
  transitions so reconstruction returns actual final fetch/turn/time totals.
- Count evaluator requests and outputs in token accounting, or expose agent and
  evaluator estimates separately when exact usage is unavailable.
- Remove duplicate ids from `context_assembled.selected_ids` while preserving order.
- Fix research-event console formatting so adjacent optional fields remain separated.
- Evaluate final-answer integrity with the actual final answer before marking the run
  complete.

### REL-10 — Default judge model

- Change application-wide `LLMConfig.judge_model` from `deepseek-v4-flash` to NaN
  model id `glm-5.3-flash`.
- Update shipped profiles and documented examples inheriting the old default.
- Explicit user configuration continues to override the default.
- Add configuration and evaluator tests for default and override resolution.

## 3. Required tests

Add revert-failing tests for at least these cases:

1. An evaluator `unsupported` verdict removes a claim from supported conclusions.
2. An unknown verdict claim id cannot mutate state or crash the run.
3. A graph-resident paper that was never acquired cannot support a claim.
4. A sparse run renders an evidence-insufficient answer without contradiction.
5. An empty citation frontier with budget schedules one search per normalized query.
6. Search results are deduplicated and become ordinary evidence candidates.
7. Transient arXiv XML failure is not classified as absence.
8. Malformed mapping JSON uses explicit-id fallback and records it.
9. Reconstruction after absent/transient terminal turns returns final budget totals.
10. Context event ids are unique and console fields are separated.
11. Final-answer integrity receives the rendered final answer.
12. The default judge is `glm-5.3-flash`; explicit configuration wins.

Include a frozen end-to-end run reproducing the original failure pattern: an agent
marks a broad claim supported, the evaluator rejects it, the frontier is exhausted,
and the final answer reports insufficient evidence instead of publishing the claim.

## 4. Verification

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/
uv run mypy src/
```

Run frozen research-kernel end-to-end tests without network access. Live API smoke
tests remain manual and must not be required by CI.

## 5. Compatibility and non-goals

- Preserve the existing `research-explorer explore` CLI and legacy ACO pipeline.
- Keep persisted research JSON readable with tolerant defaults when fields change.
- Do not claim general semantic entailment can be solved deterministically. The
  conservative boundary is acquired provenance plus explicit evaluator verdicts.
- Do not add a vector database, new external service, or unbounded search.
- Do not weaken rate limiting, secret redaction, or reproducibility.

## 6. Delivery evidence

The Convoy report must identify changed contracts for each REL requirement, regression
tests protecting every blocker, exact verification results, compatibility handling,
and any intentionally deferred requirement as an unresolved quality gap.
