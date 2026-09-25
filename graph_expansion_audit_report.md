# Graph-Expansion Objective — Final Audit & Fixes

Scope: independent audit of the complete graph-expansion objective across Phase A
(identity resolution, provider traversal, canonical aliasing, metadata-only
expansion, provenance, replay) and Phase B (ACO selection, castes, probabilistic
transition, private pheromone, concurrency claims, work-budget, convergence).
Verified the controlled arXiv:1905.07697v2 mock end-to-end path and replay data
for the run_id frontend. Fixed the defects found; preserved unrelated work; did
not commit.

## Validation commands & results

| Check | Command | Result |
|-------|---------|--------|
| Full tests | `uv run pytest tests/ -q` | **312 passed** (baseline 305 + 7 new) |
| Focused (resolution/aco/explorer/graph) | `uv run pytest tests/resolution/ tests/aco/ tests/agents/test_explorer.py tests/graph/ -q` | **208 passed** |
| Lint | `uv run ruff check src/ tests/` | **All checks passed** |
| Diff hygiene | `git diff --check` | **Clean** |
| mypy (project-wide, 3.12) | `uv run mypy --follow-imports=skip --ignore-missing-imports --python-version 3.12 src/` | **2 errors** (both pre-existing, see below) |

New test file: `tests/aco/test_e2e_arxiv.py` (7 tests).

## Defects found & fixed

1. **`src/research_explorer/resolution/providers.py` — 2 real mypy `arg-type` errors**
   in new Phase A code. `build_verification_providers` passed the registry type
   `ResilientProvider` into adapters whose constructors require
   `OpenAlexProvider` / `SemanticScholarProvider`. Mypy could not narrow by name.
   Fixed by explicitly dispatching on adapter type and `cast()`-ing the provider
   to the concrete adapter's expected type. Runtime behavior unchanged.

   Before: `src/research_explorer/resolution/providers.py:114` (x2).

Other mypy findings are **pre-existing and unrelated** (verified against `HEAD`,
unchanged by the diff):
   - `graph/store.py:92` `arr = np.frombuffer(...)` unannotated (`var-annotated`).
   - `evaluation/peer_vote.py:73` mypy fails to narrow `gather`'s
     `float | BaseException` despite the `isinstance(s, Exception)` guard.

The known NumPy-stub / `python_version="3.10"` incompatibility in
`numpy/__init__.pyi` (reported by Phase A) is outside the diff; checking under
`--python-version 3.12` isolates the new Phase A/B code (no errors remain there
after the cast fix).

## Requirement-by-requirement verification

### Phase A
- **Deterministic provider-backed resolution.** `IdentityResolver` runs strict
  DOI → arXiv → title stages with deterministic scoring/tie-break and
  `hard_gate_failures`; malformed identifiers are rejected
  (`UNVERIFIABLE_IDENTIFIER`) and fall through to title verification, never
  poisoning the graph (verified: `test_malformed_and_duplicate_identifiers_handled`,
  `test_rejected_candidates_never_pollute_graph`).
- **Typed expected/actual evidence, attempts/confidence/rejection.**
  `ResolutionResult`/`ResolutionAttempt`/`ResolutionEvidence` ✓ (test_resolver).
- **Canonical aliases/dedup + idempotent migration.** `paper_aliases`
  first-mapping-wins, chain resolution, `candidate_dedup_key` collapse, additive
  `_migrate()` ✓ (test_store).
- **Edge provenance.** `edge_provenance` + `record_edge`/`get_edge_provenance`
  with direction/provider ✓; idempotent ✓.
- **Both directions, OpenAlex primary + S2 fallback.** Verified: outgoing via
  OpenAlex, incoming via Semantic Scholar fallback when primary is empty or
  *raises* (new `test_fallback_on_unusable_provider_result`).
- **Metadata-only traversal, no credit.** `_metadata_transit` expands without
  `_integrate`/visit/`mark_integrated`; consumes a budget slot; emits
  `metadata_transit` ✓.
- **Rejection pollution prevention** ✓.
- **Config parsing** `ResolutionConfig` + `[resolution]` ✓.
- **Replay events all emitted with exact field ordering:** `neighbors_expanded`,
  `provider_fallback`, `resolution_started`/`_resolved`/`_rejected`,
  `metadata_transit` ✓.

### Phase B
- **ACO selection / transition.** `select_candidate` pure, deterministic,
  injectable RNG; `transition_weight = tau^alpha * eta^beta * dir_modifier`;
  probabilities normalized ✓ (test_transition, test_selection_integration).
- **Castes.** `caste_direction_weights` normalized; fundaciones→ref,
  impacto→cites, mixto→baseline; direction modifier applied per candidate mode ✓.
- **Private pheromone.** Per-agent `local_pheromone`; deposit/evaporation/
  elitism/MMAS clip operate on each agent's map; seeding boosts a candidate's
  probability for that agent only ✓.
- **Concurrency claims.** `SharedFrontier.claim_for`/`release` block a node only
  during the concurrent window, preserve discovered sibling paths, and leaves a
  released node selectable ✓. (Scheduler runs K agents sequentially within an
  oleada for peer-voting; the shared-frontier claim still guards cross-agent
  re-selection.)
- **Work-budget & convergence.** Metadata transits decrement budget and count
  toward `delta_work`/`scheduler.total_fetches`; `ConvergenceChecker` stops on
  budget/plateau/pheromone-concentration ✓.
- **Seeded RNG.** `Colony._agent_rng(seed + i)` reproducible; new
  `test_colony_seed_rng_is_reproducible` for agent-RNG determinism +
  `test_deterministic_selection_uses_seeded_rng`.

### arXiv:1905.07697v2 end-to-end (controlled/mock) — `test_e2e_arxiv.py`
Proves the full chain with mocks only:
- Versioned seed `arxiv:1905.07697v2` normalizes to versionless
  `arxiv:1905.07697` (`normalize_arxiv` strip `vN`; provider `_extract_id` also
  strips the version — verified by scratch harness).
- `NeighborExpander` expands **references** (outgoing, OpenAlex primary →
  `openalex:Wref1`, `openalex:Wref2`) and **cited-by** (incoming, Semantic Scholar
  fallback → `s2:S0`) even without full text (expansion is metadata-driven).
- Directed edges recorded with provenance in correct orientation; `provider_fallback`
  + `resolution_*` + `neighbors_expanded` events emitted in order.
- The canonical candidates feed into deterministic ACO selection with normalized
  probabilities (`test_arxax` main test) and, separately, a full-text-unavailable
  metadata transit expands refs+cites into selectable candidates
  (`test_metadata_only_transit_feeds_deterministic_selection`).

### Replay → run_id frontend
All frontend events are appended to the ordered `events` stream fetchable at
`/api/runs/{run_id}/events`: `resolution_started`/`resolution_resolved`/
`resolution_rejected`, `provider_fallback`, `metadata_transit`,
`neighbors_expanded`, `candidate_score` (components+weights+eta+llm_used),
`candidate_selected` (tau, alpha/beta, dir_modifier, final_weight, probability,
epsilon_branch, rationale). `CandidateScore`/`CandidateSelection`/`DetailedEvaluation`
are typed and exposed via `/api/runs/{run_id}/evaluations`. Confirmed the payload
shapes via the mock run and test_replay tests.

## Notes (no change)
- The prior scratch harness confirmed real-provider shape: OpenAlex
  DataCite-DOI (`10.48550/arxiv.<id>`) → W-id mapping feeds `get_references`;
  S2 `arXiv:<id>` direct lookup feeds `get_citations`. Fallback fires on empty +
  on transport-exception (both covered).
- `opencode.json` (GLM model entry) and other unrelated working-tree changes are
  preserved untouched.

## Files touched (this session)
- `src/research_explorer/resolution/providers.py` (cast fix)
- `tests/aco/test_e2e_arxiv.py` (new, 7 tests)

No commit made.
