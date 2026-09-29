# Research Survivor and Hidden Examination — Implementation Specification

Status: ready for implementation

This specification follows `docs/model.md`, `fulltext-reference-graph.md`, and the
existing wave-oriented ACO evaluation design. It changes the product outcome from a
short query-specific narrative into one evidence-grounded survivor that can answer
previously unseen questions about the seed paper and its research context.

The current ACO pipeline rewrites one narrative after every paper, limits the complete
narrative to 800 words, integrates only paper metadata, abstract, and TL;DR into that
narrative, and selects the historical peak of a subjective
`self + peers + virgin judge + structural` score. Those signals remain useful for
navigation, but they do not establish that the selected agent retained enough
knowledge to answer unseen questions. This specification adds durable structured
research memory and a leakage-resistant multiple-choice examination with a matched
naive baseline.

## 1. Outcome

A completed research run produces exactly one `SurvivorBundle` containing a frozen,
source-grounded research memory and a stable answering contract. The survivor is
selected primarily by performance on a hidden selection exam. Its research value is
then measured on a disjoint holdout exam against a naive agent using the same answer
model and inference settings but only the seed-paper evidence.

The primary experimental result is:

```text
research_uplift = survivor_holdout_accuracy - naive_holdout_accuracy
```

The initial CLI text is an optional research scope, not the question the survivor is
trained to answer. When the scope is blank, the system derives a bounded topic profile
from the seed paper. The final survivor must be usable for later questions without
rerunning the colony.

## 2. Product principles

1. Research memory is the product; prose is a view over that memory.
2. Full text is preferred when acquired. Abstract-only evidence is explicitly marked.
3. Every retained factual claim resolves to acquired evidence and provenance.
4. The exam author never reads agent narratives, memories, paths, identities, or
   quality scores.
5. Test-taking models never receive answer keys, evidence annotations, or examiner
   reasoning.
6. Selection questions and reported holdout questions are disjoint.
7. The survivor and naive baseline use the exact same answer model, prompt contract,
   decoding settings, question order, and tool restrictions.
8. Existing process evaluation guides exploration but cannot by itself prove topic
   competence.
9. A longer completion limit alone is not an acceptable substitute for structured
   memory.
10. All benchmark decisions are reconstructable from durable, redacted artifacts.

## 3. Release-blocking requirements

### SURV-1 — Optional research scope

- Preserve the positional CLI argument for backward compatibility, but treat it as
  `research_scope` internally rather than a required final-answer question.
- Accept a blank or omitted scope through the programmatic API. If the CLI cannot make
  the existing positional argument optional without ambiguity, add an explicit
  backward-compatible scope option and document the transition.
- Derive a deterministic bounded topic profile from the seed title, abstract, and
  identifiers when no scope is supplied.
- Use the scope to rank and bound graph exploration. Do not grade agents on whether
  they answer one literal prompt.
- Persist the effective scope and whether it was user-supplied or derived.

### SURV-2 — Structured per-paper research memory

- Replace narrative-only integration as the source of truth with a typed,
  persistible `PaperDossier` for each integrated paper.
- A dossier records at least:
  - canonical paper id, title, authors, year, and content availability;
  - research problem and stated contribution;
  - definitions and key concepts;
  - method, design, datasets, baselines, and assumptions;
  - principal results with qualifications;
  - limitations and threats to validity;
  - relevance to the effective research scope;
  - important relationships to other acquired papers; and
  - evidence references for every substantive extracted item.
- An evidence reference resolves to acquired content and retains paper id, content
  hash, acquisition event, content kind, and an optional section/page/paragraph
  locator plus a short source excerpt.
- Full text, when available, is included through a bounded chunking or staged
  extraction path. Do not silently reduce an acquired full paper to its abstract.
- Abstract-only and metadata-only dossiers are distinguishable. Metadata-only nodes
  cannot receive knowledge or examination evidence credit.
- Reuse and type the existing `paper_analyses` state boundary or migrate it additively;
  do not maintain two divergent stores of per-paper understanding.
- Persist dossier extraction failures without deleting earlier valid memory.

### SURV-3 — Claim ledger and concept relationships

- Each agent maintains typed atomic claims separate from rendered prose.
- A claim records text, status, confidence, supporting evidence, contradicting
  evidence, related concept ids, and creation/update provenance.
- Claim status supports at least `proposed`, `supported`, `disputed`, `superseded`,
  and `unknown`.
- A claim cannot become `supported` without at least one resolving acquired evidence
  reference. Cross-paper claims must identify every source needed for the relation.
- Maintain typed concept and paper relationships sufficient to represent foundation,
  extension, replication, disagreement, and application.
- Deduplicate equivalent concepts and claims deterministically without discarding
  additional provenance.
- Record open knowledge gaps separately from negative factual claims.

### SURV-4 — Narrative becomes a derived view

- Generate a readable long-form synthesis from structured memory for reports and the
  TUI. It is not the authoritative memory and is not constrained to 800 words.
- Make synthesis length configurable with a safe default appropriate for a detailed
  research briefing.
- Regeneration must not mutate dossiers, claims, evidence, or examination state.
- The synthesis distinguishes established findings, provisional interpretations,
  disputes, limitations, and unknowns.
- Existing consumers of `AgentState.narrative` remain readable during migration.
  Old saved state without structured memory must load with tolerant defaults.

### SURV-5 — Survivor bundle and later question answering

- Freeze the selected agent into a versioned `SurvivorBundle` containing its effective
  scope, dossiers, claims, relationships, evidence index, synthesis, source catalog,
  configuration fingerprint, and selection metadata.
- Store enough information to answer later questions without rerunning exploration.
- Provide an answer service or CLI boundary that assembles relevant survivor evidence
  for an arbitrary later question.
- Later answers cite canonical source ids and explicitly distinguish supported,
  plausible, disputed, and unknown statements.
- Retrieval may compact context but must never invent evidence absent from the frozen
  bundle.
- The bundle records model ids and prompt/schema versions used to construct it.

### EXAM-1 — Independent evidence pack

- Build the examination evidence pack from acquired, verified sources, never from
  agent-generated narratives or memory.
- Include the seed paper and eligible integrated full-text or abstract evidence from
  the union of the colony's acquired sources.
- Exclude metadata-only papers, unresolved identities, failed acquisitions, and text
  without durable provenance.
- Deduplicate identical source content by canonical id and content hash.
- Freeze and hash the evidence pack before exam generation.
- Persist source eligibility and exclusion reasons so the exam's knowledge boundary
  is auditable.

### EXAM-2 — Examiner configuration and isolation

- Add a separate typed examiner configuration: provider/base URL, API-key environment
  name, model id, reasoning effort where supported, temperature, token limits,
  question counts, and validation limits.
- Default the requested examiner model id to `gpt-5.6-sol`, while allowing explicit
  override and clean unavailability reporting.
- Do not assume the existing NaN-compatible endpoint serves the examiner model.
- Keep examiner credentials out of events, reports, prompts stored for replay, and
  survivor bundles.
- Exam generation failure cannot silently fall back to an explorer model. A configured
  fallback must be explicit in both configuration and the final report.
- The examiner has no access to any candidate identity, caste, path, narrative,
  memory, process score, or prior exam response.

### EXAM-3 — Grounded multiple-choice question schema

- Each `ExamItem` contains:
  - stable question id and exam version;
  - category and difficulty;
  - question text;
  - exactly four labeled options by default;
  - exactly one keyed correct option;
  - one or more evidence references supporting the key;
  - a concise examiner rationale retained privately;
  - source-distance metadata such as seed, direct reference, citant, or deeper path;
  - generation and validation status; and
  - rejection reasons when invalid.
- Support an `insufficient evidence` option when justified, but do not force it into
  every item.
- The default validated bank contains 50 items: 30 selection and 20 holdout. Counts
  are configurable, with both partitions required for a benchmark-valid run.
- Target category distribution:
  - 20 percent concepts and definitions;
  - 20 percent method and experimental design;
  - 20 percent results and interpretation;
  - 15 percent assumptions and limitations;
  - 15 percent cross-paper lineage, comparison, or contradiction; and
  - 10 percent transfer or application.
- Questions may test synthesis across sources, but every keyed answer must be entailed
  by the frozen evidence pack.

### EXAM-4 — Independent item validation

- Validate every generated item before partitioning.
- Validation checks at least:
  - schema and option-label integrity;
  - exactly one defensible correct option;
  - direct evidence support for the key;
  - distractors that are false, incomplete, or contradicted by the evidence;
  - no dependence on sources outside the evidence pack;
  - no answer leakage through wording, option length, formatting, or citations;
  - no duplicate or near-duplicate question; and
  - category and difficulty plausibility.
- Use a critic pass isolated from the generator's private rationale. The same model
  may be used with a separate prompt for the first implementation, but preserve a
  provider/model boundary for later independent validation.
- Discard ambiguous items rather than repairing their answer during scoring.
- Stop with an explicit exam-insufficient outcome if the configured validation budget
  cannot produce the minimum bank. Never reduce the holdout silently.
- Persist accepted and rejected counts and rejection reasons.

### EXAM-5 — Deterministic partition and leakage prevention

- Freeze the validated bank and answer key before any candidate answers a question.
- Partition question ids deterministically from a recorded random seed.
- Selection and holdout ids cannot overlap.
- Candidate agents receive only public question fields and options. They never receive
  keys, evidence annotations, rationales, validation results, or the other partition.
- Selection responses are not added to research memory and cannot trigger more graph
  exploration.
- Holdout questions remain inaccessible until one survivor has been frozen.
- Do not expose answer-key artifacts through normal TUI views while a test is active.

### EXAM-6 — Exact-option answer protocol

- Test takers return only a schema-constrained mapping of question id to option id.
- Explanations, chain of thought, confidence prose, citations, and free-form answers
  are neither requested nor scored.
- Reject unknown question ids, unknown option ids, duplicates, missing answers, and
  extra fields deterministically. Missing or invalid answers score as incorrect.
- Use temperature zero where supported, no tools, no network, and a bounded answer
  token budget.
- Batch size is configurable. Every test arm uses identical batching and public
  prompts.
- Record final option ids, validity, latency, and token usage, but no hidden reasoning.

### EXAM-7 — Terminal survivor selection

- Existing `S + P + J + R` becomes `Q_process`, a navigation and diagnostic signal.
- Compute selection accuracy `E_selection` from the frozen selection partition.
- Compute a deterministic grounding/integrity score `G` over the candidate's memory.
- Default terminal score:

  ```text
  Q_terminal = 0.70 * E_selection + 0.20 * Q_process + 0.10 * G
  ```

- Make weights typed and configurable, require non-negative values summing to one,
  and persist the effective formula.
- A candidate is ineligible if it has no evidence-bearing dossier, fails critical
  provenance integrity, lacks a valid selection response, or falls below configured
  minimum examination coverage.
- Rank all eligible candidates by `Q_terminal`. Break exact ties by selection
  accuracy, then grounding score, then process score, then stable agent id.
- Freeze exactly one survivor. Do not select a historical narrative snapshot whose
  structured memory no longer corresponds to the evaluated state.
- The report retains both process-peak history and terminal survivor selection so the
  distinction is visible.

### EXAM-8 — Matched naive baseline and holdout benchmark

- After survivor selection, run the hidden holdout exam twice:
  - survivor arm: the same answer model receives the frozen survivor memory through
    the normal bounded context assembly path;
  - naive arm: the same answer model receives only acquired seed-paper evidence.
- Both arms use the exact same model id/version, system prompt, public exam payload,
  batching, option order, temperature, reasoning effort, output schema, token limit,
  and tool restrictions.
- The examiner model does not answer either arm unless it is also explicitly the
  configured answer model for both arms.
- The naive arm cannot access citation-graph discoveries, shared caches beyond the
  seed evidence, survivor synthesis, or prior candidate answers.
- Compute and report survivor accuracy, naive accuracy, percentage-point uplift,
  paired item outcomes, and breakdowns by category, difficulty, content kind, and
  source distance.
- Label a single-run uplift as descriptive. Across repeated runs, provide a paired
  confidence interval and preserve run seeds/configuration fingerprints.

### EXAM-9 — Outcome semantics

- Distinguish at least:
  - `completed_benchmarked`;
  - `completed_survivor_unbenchmarked` when a valid survivor exists but the exam or
    baseline could not complete;
  - `completed_degraded` when research completes without an eligible survivor; and
  - `failed` for an operational failure that prevents a coherent terminal state.
- Examiner unavailability, insufficient validated questions, invalid candidate
  response, and baseline failure have stable reason codes.
- Never fabricate zero accuracy for an unavailable arm. Represent unavailable values
  as unavailable with a reason.
- A non-positive research uplift is a valid benchmark result, not an operational
  failure.

### OBS-1 — Durable events and artifacts

- Add typed, redacted events for evidence-pack freeze, exam generation, item
  validation summary, partition freeze, candidate test start/completion, survivor
  selection, baseline completion, and benchmark completion.
- Do not persist examiner hidden reasoning or answer keys in the ordinary event
  stream.
- Store the private answer key in a separate restricted artifact. Store a public exam
  artifact without keys for replay and reporting.
- Persist a machine-readable benchmark result and human-readable report.
- The final report includes:
  - effective research scope and its origin;
  - survivor id and terminal-score breakdown;
  - process score versus examination score;
  - source and dossier coverage;
  - selection and holdout bank statistics;
  - survivor and naive holdout results plus uplift;
  - category/difficulty/source-distance breakdowns;
  - invalid/rejected item counts;
  - model/configuration fingerprints, cost, tokens, and latency; and
  - every unavailable or degraded component with its reason.
- Replay of older traces remains supported with examination fields absent.

### TUI-1 — Examination and survivor visibility

- Preserve the existing wave-first research hierarchy and live/replay equivalence.
- Add terminal phases after research convergence: `Exam build`, `Selection`,
  `Survivor`, and `Benchmark`.
- During an active test, show counts and progress without revealing keys.
- The completed default view shows the survivor, survivor/naive accuracy, uplift,
  terminal-score breakdown, and any degraded reason.
- Detailed views expose public question outcomes and category summaries only after the
  benchmark completes.
- Keep the completed review screen open and preserve existing cancellation, report
  durability, event bounds, and memory-safety contracts.

### CFG-1 — Typed configuration

- Add typed configuration sections for memory, examination, terminal selection, and
  baseline execution.
- Validate weights, item counts, partition sizes, provider/model identifiers,
  reasoning settings, batching, retries, and evidence limits at startup.
- Ship a bounded test profile that uses deterministic fake examiner and answer clients
  without network access.
- Existing configurations continue to load with documented defaults.
- Secrets remain environment references and are never serialized into config dumps.

## 4. Required tests

Add revert-failing coverage for at least these cases:

1. Acquired full text reaches dossier extraction rather than being silently replaced
   by the abstract.
2. Abstract-only and metadata-only papers retain distinct evidence eligibility.
3. A supported claim cannot reference an unacquired graph-only paper.
4. Reintegrating a paper preserves prior valid evidence when a later extraction fails.
5. Equivalent claims merge without losing provenance.
6. A survivor bundle round-trips and can answer a later question without exploration.
7. The exam evidence pack contains no agent narrative or candidate identity.
8. An ambiguous item with two defensible answers is rejected.
9. A keyed answer without resolving evidence is rejected.
10. Selection and holdout partitions are deterministic and disjoint.
11. Test takers receive no keys, rationales, or evidence annotations.
12. Invalid or missing option ids score incorrect without crashing the run.
13. Terminal selection follows weights, eligibility gates, and deterministic ties.
14. The survivor snapshot and examined structured memory are the same state.
15. Survivor and naive holdout calls have identical answer-model settings.
16. The naive context contains seed evidence and no discovered-reference evidence.
17. Unavailable examiner or baseline values remain unavailable rather than numeric
    zero.
18. A negative uplift completes as a valid benchmark result.
19. Private answer keys never appear in ordinary trace events, logs, or TUI state.
20. Old agent state, old config, and old replay traces load with tolerant defaults.
21. The TUI projects exam phases live and replay reaches equivalent terminal state.
22. The final report reconstructs scores, partitions, models, and reason codes from
    durable state.

Include a frozen network-free end-to-end benchmark with at least two candidate agents,
a deterministic generated bank, one rejected ambiguous item, disjoint selection and
holdout partitions, a selected survivor, and a same-model naive baseline. Prove that
the expected survivor uplift and terminal score are reproduced from stored artifacts.

## 5. Verification

```bash
uv sync --extra dev
uv run pytest tests/ -q
uv run ruff check src/ tests/
uv run mypy src/
```

After deterministic checks pass, run a manual Orca TUI smoke test with the bounded
network-free profile. Confirm the visible lifecycle progresses through Research,
Exam build, Selection, Survivor, and Benchmark; no key appears before completion; the
completed screen remains reviewable; and replay reconstructs the same survivor and
uplift. A live examiner/provider smoke test may follow but must be bounded and is not
required by CI.

## 6. Compatibility and non-goals

- Preserve existing provider resilience, reference mapping, graph traversal,
  redaction, report, Obsidian export, and non-TUI behavior unless this specification
  explicitly extends them.
- Preserve the research-kernel pipeline boundary; this change targets the ACO survivor
  flow unless shared typed models can be reused without changing kernel semantics.
- Do not fine-tune or modify model weights. The survivor is a frozen state bundle used
  with the configured answer model.
- Do not treat multiple-choice accuracy as proof of unrestricted scientific expertise.
- Do not use the holdout set for exploration, prompt tuning, selection, retries, or
  memory repair.
- Do not generate questions from agent prose.
- Do not require live network calls in automated tests.
- Do not store chain of thought, raw credentials, or unrestricted full-text copies in
  trace events.
- A future fixed, versioned benchmark shared across runs is compatible with this
  design but is not required for the first per-run adaptive benchmark.

## 7. Delivery evidence

The implementation report must identify the changed contract for every `SURV`,
`EXAM`, `OBS`, `TUI`, and `CFG` requirement; document migrations and old-state
behavior; show the exact survivor and naive prompt/config equivalence proof; enumerate
all private/public artifact boundaries; include exact pytest, Ruff, mypy, and manual
visible-TUI smoke-test results; and list every unmet requirement as an explicit
unresolved quality gap.

## 8. Convoy execution contract

Use the repository's `reliability-full-cycle` pipeline. Treat this specification as
one release-blocking change and iterate until the independently verified score reaches
at least 90. The final quality report must verify the examination leakage boundary,
the identity of the examined and frozen survivor state, the same-model baseline
contract, and the full network-free end-to-end benchmark rather than accepting those
claims from implementation prose alone.
