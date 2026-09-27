# Research TUI Live Reliability — Implementation Specification

Status: ready for implementation

This specification follows `research-tui.md` and is based on the observed live run
`7a67d30b7f39`. The backend completed in 23 seconds with a durable `no_winner`
event, while the visible Textual application remained at `initializing` because its
live-event consumer stopped when the exclusive research worker started.

## 1. Outcome

Make a live ACO run accurately observable and safely reviewable from the TUI. Live
events must continue to project while research executes, terminal state must agree
with the durable replay trace, and an empty initial frontier must finish as a
completed but degraded run with an actionable explanation. When `--output` is used,
the report must be durable before the completed review screen waits for the user.

## 2. Release-blocking requirements

### TUI-REL-1 — Independent worker lifecycles

- Run the live-event consumer and research runner in distinct Textual worker groups.
- Starting the exclusive research runner must never cancel the consumer.
- Keep the consumer alive until application shutdown, including during runner
  completion, failure, and cancellation.
- Do not block orchestration on terminal rendering or remove bounded-channel
  backpressure behavior.

### TUI-REL-2 — Ordered terminal-state delivery

- A runner result is ready only after all events published before runner completion
  have been applied to `RunProjection`.
- Add queue acknowledgement or an equivalent bounded synchronization mechanism;
  avoid sleeps and timing-dependent polling.
- Durable failure and cancellation events take precedence over synthetic UI
  fallbacks. Do not duplicate a failure already received from the event stream.
- Live projection and replay projection of the same ordered event stream must reach
  equivalent terminal state.

### TUI-REL-3 — Stable review and exit behavior

- After completion, keep the TUI open so the user can inspect agents, timeline,
  warnings, metadata, and events.
- `q` in a terminal state exits immediately without an active-run confirmation.
- `q` during an active run retains the existing keep-running/cancel confirmation.
- Cancellation targets the runner, leaves the consumer able to project
  `run_cancelled`, closes resources, and persists `cancelled` rather than `failed`.
- Never let a stale UI cancel or relabel a run already completed in the trace store.

### TUI-REL-4 — Structured seed-discovery telemetry

- Attach the run tracer to explorer agents before colony seed discovery starts.
- Emit seed neighbor-discovery started, completed, and failed semantics per agent,
  carrying `agent_id`, seed paper id, wave `0`, turn `0`, discovered reference and
  citation counts, and traversable-candidate count.
- Contain concurrent initialization failures without silently discarding them.
- Classify an empty initial frontier with exactly one primary reason:
  `no_neighbors_discovered`, `no_traversable_identifiers`,
  `reference_extraction_failed`, or `seed_discovery_failed`.
- Do not persist raw LLM responses, hidden reasoning, credentials, or authorization
  data in diagnostic events.

### TUI-REL-5 — Completed degraded `no_winner` runs

- A valid seed with no traversable frontier remains `completed`, with
  `outcome=degraded`; it is not a failed run and the seed is not promoted to winner.
- Preserve the durable legacy `no_winner` event for existing replay consumers.
- Add structured fields for status, outcome, reason code, concise reason, elapsed
  time, total fetches, and total waves.
- Emit a preceding warning event with the same reason so the final timeline and
  warning-filtered event view explain the stop.
- Old traces containing an unenriched `no_winner` event must still project as
  completed and receive a generic non-duplicated explanation.

### TUI-REL-6 — Degraded result presentation

- Show the terminal reason in the final TUI timeline, event view, and metadata.
- Include outcome and terminal reason in the Markdown report summary.
- A no-winner report must retain seed metadata, configuration, elapsed time, agent
  diagnostics, and discovered-paper information, while explicitly stating that no
  winning narrative was produced.
- Warning presentation must be textual and must not rely on color alone.

### TUI-REL-7 — Durable `--output` before review

- When `--output` is supplied, write the completed report once when backend
  execution finishes, before waiting on the final review screen.
- Use an atomic same-directory replacement so interruption cannot leave a partially
  written report. Preserve the current behavior that a missing parent directory is
  an error rather than creating directories implicitly.
- After the user exits the TUI, print the report-path confirmation without rewriting
  the file.
- A report-write failure produces a redacted stable command failure and nonzero exit
  without changing the already-persisted research run from `completed`.
- Preserve normal stdout report behavior when `--output` is absent and preserve
  Obsidian export behavior.

### TUI-REL-8 — Compatibility and boundaries

- Keep the `research-explorer explore ... --tui` CLI and its flags unchanged.
- Preserve replay database readability and deterministic reconstruction for old and
  new events.
- Do not change provider routing, enable automatic provider fallback, or modify the
  shipped quick profile in this change.
- Preserve non-TUI ACO behavior and the research-kernel pipeline.

## 3. Required tests

Add revert-failing tests for at least these cases:

1. A real `queue` and `runner` used together under Textual's test pilot keep both
   workers alive and update the screen before the runner finishes.
2. Runner completion waits for queued events, yielding populated agents and a
   terminal `completed` projection rather than `initializing`.
3. `q` after completion exits directly; `q` while active still confirms.
4. Cancellation allows the consumer to project `run_cancelled` and persists the
   distinct cancelled state.
5. Each empty-frontier reason produces its expected structured initialization and
   completion telemetry.
6. New degraded traces display one warning and replay as completed with no winner.
7. An old unenriched `no_winner` trace remains readable and displays one generic
   warning.
8. Live-channel and replay projections of the same fixture have equivalent run id,
   status, agents, timeline, warnings, and winner state.
9. `--output` exists before the completed review screen is dismissed and is not
   rewritten after exit.
10. Atomic report-write failure is redacted, exits nonzero, and leaves the durable
    research run completed.
11. Non-TUI ACO output, unsupported-pipeline rejection, and research-kernel behavior
    remain unchanged.

Include a frozen end-to-end regression reproducing run `7a67d30b7f39`: three agents
initialize from an arXiv seed, no traversable neighbors survive, the backend emits a
completed degraded outcome, the TUI renders the reason, and the report is preserved.

## 4. Verification

```bash
uv sync --extra dev
uv run pytest tests/ -q
uv run ruff check src/ tests/
uv run mypy src/
```

After deterministic checks pass, run a manual Orca smoke test with the bounded quick
profile. Confirm that the screen advances beyond `initializing`, reaches a stable
terminal state, `q` exits without cancellation, and the report and replay trace both
remain available. The live network smoke test is manual and must not be required by
CI.

## 5. Convoy execution contract

Use the repository's `reliability-full-cycle` pipeline. The implementation may be
divided into independent worker-lifecycle, diagnostics/projection, and output-
durability work, but the final integration must satisfy every TUI-REL requirement as
one coherent change.

The Convoy delivery report must include:

- the worker grouping and terminal-event synchronization mechanism;
- the additive event payload contract and old-trace compatibility behavior;
- the four empty-frontier classifications and their regression tests;
- proof that completed review cannot cancel an already-completed durable run;
- proof of pre-review atomic report persistence;
- exact pytest, Ruff, mypy, and manual smoke-test results; and
- any unmet requirement as an explicit unresolved quality gap.
