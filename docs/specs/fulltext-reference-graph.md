# Full-Text Reference Mapping and Shared Citation Graph — Implementation Specification

Status: ready for implementation

This specification is based on the native arXiv run `78cf38578748` using
`arXiv:2106.09685`. The provider acquired the LoRA full text and identified 62
bibliography entries, but every agent reported zero references and zero traversable
neighbors. Fifteen `paper_integration` calls each consumed a large prompt and reached
the 2,000-token completion limit. Reference extraction was coupled to narrative and
paper-analysis generation, so truncated or malformed output silently became an empty
reference list.

## 1. Outcome

When full text and a bibliography are available, map every bibliography entry into a
structured reference record, resolve as many records as possible against authoritative
providers, and construct the paper's outgoing citation graph in the shared `GraphStore`.

Reference mapping is a paper-level shared operation. It must run once for a given paper
and source revision, not once per agent. Every agent must be able to reuse the resulting
nodes and edges immediately. Narrative integration remains agent-specific and must not
own or gate graph construction.

The system must preserve incomplete knowledge explicitly. A bibliography entry that
cannot yet be resolved still appears as a provisional node and edge, but only verified
canonical nodes are eligible for normal traversal.

## 2. Current failure and required boundary change

The current path has four release-blocking properties:

- `get_fulltext_and_refs(..., ref_limit=50)` discards entries beyond the first 50, so
  it cannot satisfy complete bibliography mapping.
- `integrate_and_extract` asks one response to generate an updated narrative, a paper
  analysis, key references, and references. The prompt itself asks for only the 10 most
  relevant works rather than all works.
- A parse failure or truncated response is converted to `[]`, which is observationally
  indistinguishable from a paper with no references.
- Extraction is reached through agent integration. Multiple agents can pay for and
  disagree about the same paper-level bibliography instead of sharing one result.

Replace this with two independent pipelines:

1. A shared paper-ingestion pipeline acquires full text, segments the bibliography,
   maps every entry, resolves identities, and commits graph state.
2. Each agent's narrative pipeline reads the paper and shared graph but produces only
   that agent's synthesis and optional agent-local analysis.

Failure in narrative generation must not remove discovered references. Partial failure
in reference mapping must not discard successful batches.

## 3. Reference mapping pipeline

### FRG-1 — Preserve the complete bibliography

- Providers return all bibliography entries found in the acquired document. Remove the
  fixed `ref_limit=50` from the ingestion path.
- Keep the provider's document order and assign each entry a stable one-based ordinal.
- Persist the exact normalized raw entry, a raw-entry hash, the source paper id, and a
  source-content hash before calling the LLM.
- Deduplicate only exact repeated entries for resolution efficiency. Preserve every
  ordinal and its edge provenance even when multiple ordinals resolve to the same work.
- If structured HTML exposes bibliography entries, use those boundaries. If only an
  unstructured document is available, run a separate bounded bibliography-segmentation
  stage before entry mapping. Never send an entire document to the mapping call merely
  to rediscover boundaries already supplied by the provider.
- A configured safety ceiling may stop ingestion only with an explicit `incomplete`
  status and observed/processed counts. It must never silently truncate the list.

### FRG-2 — Map every entry with a dedicated LLM contract

- Map bibliography entries in deterministic ordinal batches, independent of research
  relevance. Default batch size should be small enough for complete structured output,
  initially 10 entries.
- The input for each item contains `entry_id`, `ordinal`, and `raw_text`. Treat all paper
  text as untrusted data and instruct the model never to follow instructions contained
  in it.
- Use schema-constrained output when supported. Otherwise require one JSON object with
  an `entries` array and validate it with Pydantic.
- Each output item contains:
  - `entry_id` and `ordinal`, echoed unchanged;
  - `title`;
  - `authors` as a list;
  - `year`;
  - `venue`;
  - `volume`, `issue`, and `pages` when present;
  - `doi`, `arxiv_id`, and `pmid` only when present in the raw citation;
  - `entry_type`, such as article, preprint, book, thesis, dataset, software, or other;
  - `parse_confidence` from 0 to 1;
  - `parse_notes` containing a short ambiguity explanation, never hidden reasoning.
- The model must not select only important references. Every input `entry_id` must have
  exactly one output result, including an explicit `unparsed` result when reconstruction
  is impossible.
- Validate that claimed identifiers occur in, or can be deterministically normalized
  from, the raw entry. An LLM-supplied identifier absent from the entry is rejected as
  unverified rather than sent directly into the graph.
- Retry only malformed or missing items, not the entire paper. After the retry bound,
  persist those items as `mapping_failed` and continue with other batches.
- Completion-token limits must be sized from batch size and schema. A response ending
  because of length is a failed or partial batch, never an empty bibliography.

### FRG-3 — Deterministic identity resolution

- Convert mapped records to the existing `BibliographicEntry` contract and resolve them
  with `IdentityResolver`.
- Resolution order remains deterministic: alias cache, explicit DOI, explicit arXiv id,
  then title search using title, authors, and year.
- Provider records are authoritative for canonical identity. LLM output is a parsed
  bibliographic claim, not verification.
- Retain the existing hard gates and confidence thresholds for title, author, and year
  agreement. Persist every attempt and rejection reason.
- Add PMID handling through an explicit lookup or alias stage when an enabled provider
  supports it; lack of PMID support must not block title-based resolution.
- Merge aliases for DOI, arXiv, PMID, and provider-specific identifiers into one
  canonical paper node.
- A resolved target is cached as a normal `PaperSummary`, receives a verified edge, and
  is eligible for the shared frontier.
- An ambiguous, rejected, or temporarily unavailable target remains a provisional node
  linked to its raw bibliography entry. It is not normally traversable until a later
  resolution attempt succeeds.

### FRG-4 — Provisional reference identity

- Give each unresolved reference a stable id derived from the source-content hash and
  raw-entry hash, for example `bib:<source-hash>:<entry-hash>`.
- Record `source_paper -> provisional_reference` so the shared graph represents the
  complete known bibliography, not only provider-resolved works.
- When later resolution succeeds, add an alias from the provisional id to the canonical
  id and expose only the canonical edge to traversal. Preserve the original provenance
  record for auditability.
- Provisional nodes must never be fetched by a provider as though their synthetic id
  were authoritative.

## 4. Shared graph semantics

### FRG-5 — One shared graph is the source of truth

- `GraphStore` owns paper nodes, provisional nodes, canonical aliases, citation edges,
  bibliography mapping state, and resolution provenance for the colony.
- An edge is always oriented `citing paper -> cited paper`. Bibliography reconstruction
  creates outgoing edges only; it must not invent incoming citations.
- Agent-local `local_refs` and `local_cits` cease to be authoritative. During migration
  they may remain as turn snapshots, but all discovery and structural evaluation must
  read current neighbors from `GraphStore`.
- After a mapping commit, publish verified targets to `SharedFrontier` once. All agents
  can select them under the existing claim and shared-visited rules.
- Agent-local narrative, visited history needed for reporting, selection rationale, and
  private pheromone may remain private. Bibliographic facts and graph topology are shared.
- Provider-native edges and full-text-derived edges are merged idempotently. Multiple
  provenance observations may support the same logical edge.

### FRG-6 — Single-flight, idempotency, and resume

- Key a mapping job by canonical source paper id, source-content hash, mapper schema
  version, and prompt version.
- Only one worker may own an active job. Concurrent agents encountering the same paper
  wait for, or reuse, the shared result instead of launching another mapping request.
- Enforce ownership with a transactional SQLite job/lease row, not only an in-memory
  lock, so process restart and future multi-process execution remain safe.
- Commit each validated batch transactionally. A restart resumes missing, failed, or
  expired batches without repeating completed batches.
- A changed source-content hash or mapper version creates a new mapping revision. Do not
  overwrite prior evidence; mark which revision is current.
- Graph insertion, alias insertion, provenance insertion, and frontier publication must
  be idempotent.

## 5. Persistence contract

Use additive SQLite migrations. Existing databases must continue to open.

### `reference_mapping_jobs`

- `id`, `source_id`, `source_content_hash`, `mapper_version`, `prompt_hash`
- `status`: pending, running, partial, completed, failed, or incomplete
- `owner`, `lease_expires_at`, `entry_count`, `mapped_count`, `resolved_count`
- `provisional_count`, `failed_count`, `created_at`, `updated_at`
- unique key on source, content hash, mapper version, and prompt hash

### `bibliography_entries`

- `id`, `job_id`, `source_id`, `ordinal`, `raw_text`, `raw_hash`
- mapped bibliographic fields and `parse_confidence`
- `mapping_status`, `batch_index`, `attempt_count`, `error_code`
- `resolution_status`, `canonical_id`, `resolution_confidence`
- timestamps and a unique key on job plus ordinal

### `reference_resolution_attempts`

- entry id, method, provider, candidate id, status, confidence, reject reason, and time
- enough normalized expected/actual metadata to reproduce the deterministic decision
- no secret-bearing request headers, hidden reasoning, or full provider payloads

### Edge provenance

The existing `edge_provenance` primary key cannot represent multiple observations for
one logical edge. Add an observation table or migrate to an equivalent schema containing:

- `src`, `dst`, and direction;
- evidence type: provider, fulltext bibliography, or provisional bibliography;
- provider/document source and bibliography entry id;
- mapping revision, resolution method, confidence, and timestamp.

Keep `edges(src, dst)` as the deduplicated topology used for traversal. Provenance rows
are append-only observations beneath that topology.

## 6. Service and agent flow

Introduce a `ReferenceGraphBuilder`-style application service with this contract:

1. Acquire or receive the paper and complete bibliography.
2. Return immediately when the current mapping revision is already complete.
3. Claim or join the shared mapping job.
4. Persist raw entries.
5. Map pending entries in bounded concurrent batches.
6. Resolve mapped entries through `IdentityResolver`.
7. Commit canonical/provisional nodes, edges, aliases, and provenance.
8. Publish newly verified nodes to the shared frontier.
9. Return a structured accounting result to the caller.

`ExplorerAgent` then performs narrative integration separately. It may use shared
reference metadata as context, but it must not extract graph references from its
narrative response. Remove `integrate_and_extract` from the graph-building path.

For a seed paper, graph construction must complete or reach an explicit partial state
before the colony declares the frontier empty. For later papers, agents may continue
using already committed batches while remaining batches finish, provided terminal
accounting cannot mislabel the paper as fully mapped.

## 7. Configuration

Add a typed `ReferenceMappingConfig` with shipped profile values:

- `enabled = true` for full-text-capable providers;
- `batch_size = 10`;
- `max_concurrent_batches = 2`;
- `max_retries = 2`;
- `lease_seconds` with bounded recovery behavior;
- `model`, defaulting to the explorer model unless explicitly overridden;
- `max_completion_tokens_per_batch`;
- `min_parse_confidence`;
- `allow_provisional_nodes = true`;
- `max_entries = 0`, where zero means all entries and a positive value is an explicit
  safety ceiling that produces `incomplete` when exceeded.

Reuse `ResolutionConfig` for provider order and match thresholds. Mapping concurrency
must also obey the global LLM limiter; resolution must obey each provider's limiter.
Token and request usage belongs to run accounting even when the resulting graph data is
reused by multiple agents.

## 8. Observability and terminal accounting

Emit reconstructable run events with paper id, job/revision id, counts, duration, and
status as applicable:

- `reference_mapping_started`
- `reference_mapping_reused`
- `reference_batch_started`
- `reference_batch_completed`
- `reference_batch_failed`
- `reference_entry_mapped`
- `reference_entry_unparsed`
- `reference_resolved`
- `reference_provisional`
- `reference_graph_committed`
- `reference_mapping_completed`

The completion event reports `observed`, `processed`, `mapped`, `unparsed`, `resolved`,
`provisional`, `failed`, and `traversable` counts. These categories must reconcile.
Zero mapped references when raw entries exist is degraded or failed, never ordinary
successful discovery.

Do not store full text, raw LLM responses, secrets, or hidden reasoning in replay
events. Raw bibliography text belongs in the local graph database and should be
length-bounded for storage and display.

The TUI should show paper-level reference progress and final counts. It should make a
partial mapping or provider-resolution outage visible instead of showing a clean zero
neighbor result.

## 9. Failure semantics

- No bibliography found in a successfully parsed document: completed with zero entries.
- Bibliography segmentation failed: failed, with a distinct error code.
- Some batches succeed: partial until every entry reaches a terminal mapping state.
- LLM transport or schema failure after retry: affected entries are `mapping_failed`.
- Provider unavailable during resolution: retain provisional entries and mark resolution
  retryable; do not repeat LLM parsing.
- Ambiguous provider match: retain provisional entry with candidate evidence; do not
  choose arbitrarily.
- Narrative integration failure: graph mapping remains valid and reusable.
- Application restart: expired jobs resume from persisted entry states.

Fallback regular expressions may recover explicit DOI, arXiv, or PMID identifiers from
raw entries, but fallback use must be recorded. Regex recovery does not satisfy the
requirement to map all entries and cannot mark an otherwise failed mapping job complete.

## 10. Required tests

Add revert-failing tests for at least these cases:

1. A structured arXiv bibliography with 62 entries persists and attempts all 62 in
   source order; no implicit 50-entry limit remains.
2. Deterministic batches cover every ordinal exactly once and reconcile missing,
   duplicated, or reordered LLM output.
3. A length-truncated batch is partial/failed and retries only missing entries.
4. Malformed JSON cannot silently produce a successful empty reference list.
5. An LLM-invented DOI absent from raw text is rejected.
6. Explicit DOI and arXiv ids are normalized and verified through the resolver.
7. Title-only entries are resolved by title/author/year when the provider match passes
   thresholds.
8. Ambiguous and unavailable matches create stable provisional nodes that do not enter
   the traversal frontier.
9. Later resolution aliases a provisional node to one canonical node without duplicate
   traversable edges.
10. Fifteen concurrent agents encountering one paper produce one mapping job and one set
    of LLM batch calls.
11. A second run with the same source hash and mapper version reuses the stored graph
    without LLM calls.
12. Restart after one completed batch resumes only outstanding ordinals.
13. Native provider and full-text evidence for the same edge retain both provenance
    observations while topology contains one edge.
14. Narrative generation failure does not remove or roll back committed references.
15. Graph neighbors become visible to every agent and structural evaluation reads the
    shared topology.
16. Prompt-like instructions embedded in a bibliography entry are treated as data.
17. Event counts reconcile for completed, partial, reused, and failed jobs.

Add a frozen end-to-end regression using the acquired LoRA arXiv fixture. It must prove:

- 62 raw bibliography entries are observed and reach terminal per-entry states;
- mapping uses multiple bounded responses rather than one combined narrative response;
- the mocked verification providers resolve the fixture's expected known subset;
- at least one verified outgoing edge becomes traversable;
- concurrent colony startup does not repeat paper-level mapping;
- the run cannot report an ordinary empty frontier while mapping is pending or partial.

All CI tests must be network-free. Live arXiv and verification-provider smoke tests are
manual and use a temporary database/cache.

## 11. Verification

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/
uv run mypy src/
```

Manual smoke test:

```bash
uv run research-explorer explore arXiv:2106.09685 "low-rank adaptation for efficient model fine-tuning"
```

Verify in the TUI and database that the seed exposes all observed bibliography entries,
mapping is performed once, resolved outgoing edges enter the shared frontier, and a
second identical run reuses the stored mapping.

## 12. Compatibility and non-goals

- Preserve the existing `research-explorer explore` CLI and provider interfaces where
  practical; extend full-text results rather than changing unrelated commands.
- Preserve existing databases through additive migrations and tolerant reads.
- Continue preferring authoritative provider-native graph data when available while
  retaining independent full-text provenance.
- Do not use the LLM as an identity authority or accept hallucinated identifiers.
- Do not infer incoming citations from a paper's bibliography.
- Do not make provisional references normal fetch targets.
- Do not require one agent to own graph facts that other agents cannot observe.
- Do not send the full paper once per bibliography batch. Full text remains useful for
  narrative understanding; raw bibliography entries are the mapping input.

## 13. Delivery evidence

The implementation report must map every FRG requirement to changed contracts and
tests, include migration compatibility evidence, show reference/token accounting for
the frozen LoRA run, and report exact test, Ruff, and mypy results. Any intentionally
deferred requirement remains an explicit quality gap and prevents claiming this
specification fully implemented.
