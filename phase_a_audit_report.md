# Phase A Audit Report

Scope: audit the full Phase A working-tree diff against every stated Phase A
requirement, fix issues found, preserve unrelated changes, no commit.

## Outcome

- Full test suite: **264 passed**
- Focused suites (tests/resolution + tests/agents/test_explorer.py +
  tests/graph/test_store.py): **144 passed**
- Ruff (src/ + tests/): **All checks passed**
- mypy: Phase A files clean. Full-project `mypy src/` is blocked by a
  pre-existing environment incompatibility (numpy 3.14 stubs use `type` line
  syntax, but `pyproject.toml` sets `mypy python_version = "3.10"`), and it
  errors inside `numpy/__init__.pyi` before Phase A code is reached. This is
  unrelated to the Phase A diff (verified: it predates it). I validated Phase A
  typing with `--follow-imports=skip --ignore-missing-imports --python-version 3.12`.

## Issues found and fixed (2 real type errors in Phase A code)

1. `src/research_explorer/resolution/traversal.py:87` — `expand()` passed
   `None` for the `extracted: list[PaperSummary]` parameter of `_expand(...)` on
   the incoming direction. Mypy: incompatible type None vs list[Any]. Fixed to
   `[]` (incoming carries no extracted full-text refs).

2. `src/research_explorer/agents/explorer.py:602` — in `_metadata_transit`, the
   tracer payload `provider=summary.provider if summary is not None else paper.provider`
   dereferenced a possibly-`None` `paper`. Mypy: union-attr. Fixed by computing
   `provider` defensively with an explicit `paper is not None` guard (the call
   site static type is `Paper | None`).

Both edits changed behavior only in the sense of making the static type correct;
runtime behavior is unchanged (the prior `summary is None and paper is None`
guard already guaranteed at least one was set).

## Requirement-by-requirement verification

- **Deterministic provider-backed DOI/arXiv/title-author-year resolution, no
  invented-ID trust.** `IdentityResolver.resolve` runs a strict DOI stage →
  arXiv stage → title-search stage, each provider-backed, with deterministic
  `score_candidate` + `hard_gate_failures` gates and a deterministic tie-break.
  Malformed identifiers (e.g. a DOI given the `10.<4-9 digits>` guard, or a
  bogus arXiv id) are rejected as `UNVERIFIABLE_IDENTIFIER` and fall through to
  title verification; a title-search hit for a sibling work replaces the
  invented id with the provider's real id (resolver + traversal_tests).
- **Typed expected/actual evidence, attempts/confidence/rejection.**
  `ResolutionResult`/`ResolutionAttempt`/`ResolutionEvidence` carry typed
  `expected`/`actual` metadata, per-provider attempts, reason + confidence
  (`tests/resolution/test_resolver.py`).
- **Canonical aliases/dedup + legacy-safe idempotent SQLite migration.**
  `paper_aliases` (first-mapping-wins, alias-chain resolution), `candidate_dedup_key`
  collapse, `_migrate()` additive + idempotent (adds `arxiv_id`, `integrated`),
  tested for legacy-DB preservation and reopen persistence.
- **Edge provenance.** `edge_provenance` table + `record_edge`/`get_edge_provenance`
  carrying direction + provider + timestamp, idempotent.
- **Both directions, OpenAlex primary + Semantic Scholar fallback.** `NeighborExpander`
  expands incoming (cited-by) and outgoing (references); empty/unavailable
  primary triggers fallback (tracer `provider_fallback`, `fallback_used`);
  OpenAlex `referenced_works` batch-hydration and `cited_by` filtering; verified.
- **Metadata-only traversal independent of full text, no credit.**
  `ExplorerAgent._metadata_transit` expands summary-only / fetch-failed nodes
  without `_integrate`, without recording a visit or pheromone edge, and without
  `mark_integrated`; only consumes a budget slot (`record_transit`), emits
  `metadata_transit`. `mark_integrated` fires only when `fulltext or abstract`.
- **Rejection pollution prevention.** Rejected/unverifiable resolution candidates
  never touch store: no node insert, no alias registration, no edge/provenance
  record (test `test_rejected_candidates_never_pollute_graph`,
  `test_unhydrated_stub_skipped_without_calling_resolver`).
- **Configuration.** `ResolutionConfig` + `[resolution]` TOML parsing + defaults;
  `build_neighbor_expander` short-circuits to `None` when disabled or no source
  available (opt-in, default off). `opencode.json` change (GLM 5.3 Flash model
  entry) is unrelated and untouched.
- **All required replay events.** `neighbors_expanded`, `provider_fallback`,
  `resolution_started` / `resolution_resolved` / `resolution_rejected`, and
  `metadata_transit` verified with exact payload ordering/fields.
- **Mocked arXiv:1905.07697v2 scenario.**
  `test_arxiv_1905_07697v2_resolves_to_versionless_canonical`,
  `test_arxiv_1905_07697v2_title_mismatch_rejects`, and traversal
  `test_arxiv_candidate_resolves_to_versionless_canonical`; also exercised an
  arxiv-seed full expansion (incoming via S2 `arXiv:` key, outgoing via OpenAlex
  DataCite `10.48550/arxiv.*` DOI) successfully to versionless `arxiv:<id>`.

## Integration verification (beyond unit helpers)

Built a throwaway harness driving `NeighborExpander` + real `IdentityResolver`
against a mocked OpenAlex (primary) / Semantic Scholar (fallback) pair seeded
from an arXiv node. Confirmed canonical IDs, per-direction edges with
provenance, correct direction keys (`arXiv:` / `10.48550/arxiv.*`), and
cross-provider resolution into `openalex:W...` / `arxiv:<versionless>`.

Also confirmed no dead code / debug (no TODO/FIXME/print/pdb) and correct
module wiring (`Colony.__init__` -> `build_neighbor_expander` -> agents; no
import cycle; `TYPE_CHECKING` import used in explorer.py).

## Notes / non-blocking observations (no code change made)

- When the expander is active, `cache_paper(paper)` still records the fetched
  paper's raw provider-native reference/citation edges (legacy behavior that
  predates this diff; the diff only added provenance). Verified/recognized as
  intended for the structure-only provider graph; resolution candidates are
  added separately via the expander and are the pollution-guarded path.
- Fallback triggers only on empty/unavailable primary, not on resolution
  rejection (a present-but-unverifiable OpenAlex work stays rejected rather
  than re-attempted via Semantic Scholar). Reasonable, deterministic; unchanged.
- `store.py:92` (`np.frombuffer` unannotated `arr`) is a pre-existing mypy note
  from the initial snapshot, untouched.

## Modified files (Phase A unchanged files preserved; my edits)

- `src/research_explorer/resolution/traversal.py` (extracted `[]` fix)
- `src/research_explorer/agents/explorer.py` (defensive provider calc)

No commit made.
