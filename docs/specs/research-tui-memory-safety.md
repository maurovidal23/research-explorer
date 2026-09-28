# Research TUI Memory Safety and Crash Recovery — Implementation Specification

Status: ready for implementation

This specification follows `research-tui.md` and
`research-tui-live-reliability.md`. It is based on the observed live arXiv run
`b8ab967667ba`, which Linux killed after the `research-explorer` process reached
approximately 5.3 GiB resident memory on an 8 GiB WSL host with exhausted swap.

The durable trace contained only 4,658 events and 1.9 MiB of JSON payload. A
one-shot replay projection of the same trace used approximately 81 MiB. The live
run emitted 3,877 `candidate_score` events; the TUI applied and fully rendered
every event independently, retained all event and candidate objects, and rebuilt
an Events-tab Markdown document that had reached 844 KiB. The kernel terminated
the process with `SIGKILL`, so no terminal event was emitted and the run remained
incorrectly recorded as `running`.

## 1. Outcome

Make long-running ACO research observable without memory growth proportional to
render count or unbounded event history. Preserve the complete durable replay
trace while keeping the live Textual process within a predictable memory bound.
If the process is forcibly terminated, later inspection must identify the run as
interrupted rather than indefinitely running.

## 2. Release-blocking requirements

### TUI-MEM-1 — Coalesced rendering

- Continue applying live events in durable sequence order.
- Do not call a complete dashboard refresh once per event.
- Coalesce event bursts into a bounded render cadence, configurable in one place
  and no faster than 10 refreshes per second by default.
- A refresh must render the newest fully applied projection, not an intermediate
  snapshot captured when the refresh was scheduled.
- Terminal events, user navigation, resize, and explicit follow-live actions may
  request an immediate refresh, but pending scheduled refreshes must collapse into
  that refresh rather than execute afterward.
- Rendering must never block the research runner or remove the bounded event
  channel's drop-on-overflow behavior.
- Before runner completion is presented, drain or reconcile the channel so the
  terminal screen includes every event published before completion.

### TUI-MEM-2 — Bounded live projection

- SQLite remains the authoritative, complete event history. Do not truncate or
  delete durable events, candidate scores, evaluations, artifacts, or graph data.
- Replace unbounded live retention of `RunViewState.events` and
  `RunViewState.candidate_scores` with explicit bounded windows.
- Centralize and document the window sizes. Defaults must be sufficient for useful
  inspection while remaining small enough for a multi-hour run; use at most 1,000
  recent events and at most 1,000 recent candidate scores unless a test-backed
  smaller bound is selected.
- Maintain total-seen and dropped-from-live-window counters so the UI never implies
  the visible window is the complete history.
- Preserve selection details, final evaluation data, warnings, failures, terminal
  state, agent summaries, and navigation entries independently of eviction from a
  recent-event window.
- Live and replay projection must use the same deterministic compaction rules for
  equivalent inputs.

### TUI-MEM-3 — Bounded Events and candidate presentation

- The Events tab must render a bounded page, not all retained or durable events.
- Display the visible range and total durable/seen count, including a clear notice
  when older events are omitted from the live window.
- Provide bounded older/newer page navigation for replay-backed sessions. Live
  sessions may show a recent window and defer older history to `replay tui` after
  it is durable.
- Filtering by agent and outcome must not create an unbounded intermediate list or
  repeatedly stringify every historical payload on each live update.
- Candidate-score presentation must preserve the selected candidate and useful top
  alternatives while avoiding one permanent UI object per scored frontier item.
- The rendered Markdown body and live navigation widget count must remain bounded
  as the durable event count grows.

### TUI-MEM-4 — Stable widget lifecycle

- Reuse existing tree rows and content widgets when topology has not changed.
- Avoid clearing and rebuilding the complete Textual tree for ordinary telemetry
  or candidate-score events.
- If topology changes require rebuilding, ensure removed widgets and Markdown
  documents are detached and collectible before subsequent rebuilds accumulate.
- Do not leave unbounded Textual messages, workers, timers, renderables, or detached
  widgets queued behind the live-event consumer.
- Shutdown must cancel timers/workers and release projection, queue, and renderer
  references without waiting indefinitely for dropped events.

### TUI-MEM-5 — Durable run heartbeat and stale-run recovery

- Add an additive, backward-compatible persistence mechanism that distinguishes an
  active run from a `running` row left by `SIGKILL`, OOM, host loss, or terminal
  destruction.
- Record process identity, host identity, and a periodically refreshed heartbeat
  timestamp without emitting high-volume replay events.
- Normal completion, failure, and cancellation must clear/finalize heartbeat state.
- When a local process is provably absent or a heartbeat is stale beyond a documented
  threshold, replay/listing must present the run as `interrupted` with a concise
  reason. Do not relabel a potentially active remote run based only on local PID
  absence.
- Reconciliation must be idempotent, preserve the last durable event, and never
  fabricate successful completion or a winner.
- Old databases without heartbeat columns must migrate additively and remain
  readable.

### TUI-MEM-6 — OOM incident observability

- Show the last durable activity timestamp and last durable operation for running
  and interrupted runs.
- A recovered interrupted run must display that its final in-memory work may have
  been lost and that durable trace/graph data remains available.
- The report/replay metadata must distinguish `interrupted` from `failed`,
  `cancelled`, and `completed`.
- Token totals are historical API-reported usage. The UI must not imply that tokens
  continue accumulating when no new LLM completion telemetry has arrived.

### TUI-MEM-7 — Compatibility and scope boundaries

- Preserve `research-explorer explore ... --tui`, `replay tui`, and non-TUI CLI
  interfaces.
- Preserve complete SQLite trace data and old-trace replay compatibility.
- Do not change provider routing, ACO selection semantics, reference mapping,
  quality scoring, model prompts, or token budgets in this change.
- Do not add comments to production code.
- Keep all public functions typed and all I/O asynchronous.

## 3. Required tests

Add revert-failing coverage for at least these cases:

1. A burst of 10,000 candidate-score events is applied in order while causing a
   bounded number of full TUI refreshes, not 10,000 refreshes.
2. After 50,000 mixed events, live event and candidate windows remain at their
   configured bounds while total-seen counters remain exact.
3. Every event in the stress fixture remains present in the durable trace despite
   live-window compaction and channel overflow.
4. Terminal delivery flushes the newest projection immediately and does not leave a
   delayed scheduled refresh.
5. The Events tab renders only one bounded page and truthfully reports its range,
   total, filters, and omitted history.
6. Agent/outcome filtering and newest/oldest page navigation remain deterministic.
7. Thousands of candidate-score updates do not rebuild the tree when its topology
   is unchanged; topology changes still appear correctly.
8. A Textual pilot stress test drives at least 10,000 events through the real live
   consumer and asserts bounded queue size, rendered row count, Markdown size, and
   refresh count.
9. A subprocess memory regression replays a representative high-volume trace
   through the live TUI update path and stays below a documented generous RSS or
   `tracemalloc` growth ceiling. Avoid timing-sensitive or platform-fragile limits.
10. A normally active heartbeat remains active; clean completion finalizes it.
11. A stale local heartbeat with an absent process becomes `interrupted` exactly
    once and preserves its last durable event and graph data.
12. A remote or otherwise unverifiable heartbeat is not falsely relabeled.
13. Old replay databases migrate and project without data loss.
14. Existing live/replay equivalence, cancellation, terminal-review, output,
    provider, ACO, and research-kernel tests remain green.

Use run `b8ab967667ba` as the shape of the frozen regression fixture: 15 agents,
large frontier scoring bursts, more than 3,800 candidate scores, reference mapping,
multiple waves, and an Events-tab view. The fixture must be synthetic and
network-free; do not commit the user's live database.

## 4. Acceptance criteria

- The synthetic incident trace no longer produces memory growth proportional to
  the number of refreshes.
- At 50,000 events, retained live collections, rendered Markdown, tree rows, widget
  count, queued messages, and measured memory stay within their documented bounds.
- Durable event count remains exact and replay remains deterministic.
- A forced subprocess termination is subsequently shown as interrupted, never as
  actively running or completed.
- The existing arXiv full-text reference workflow remains behaviorally unchanged.
- No secrets, raw model responses, hidden reasoning, or authorization data enter
  events, reports, fixtures, or logs.

## 5. Verification

```bash
uv sync --extra dev
uv run pytest tests/ -q
uv run ruff check src/ tests/
uv run mypy src/
```

After deterministic checks pass, run a manual Orca smoke test using a bounded local
or synthetic profile. Keep the TUI visible, switch repeatedly between Research and
Events, and verify from a separate shell that RSS stabilizes rather than growing
with every event. A live network smoke test is optional and must not be required by
CI.

## 6. Convoy execution contract

Use the repository's `reliability-full-cycle` pipeline. Treat the specification as
one release-blocking change and iterate until the verified quality score reaches at
least 90.

The delivery report must include:

- the chosen refresh cadence and coalescing mechanism;
- every live-state and rendered-page bound;
- proof that the durable trace remains complete;
- before/after memory measurements for the synthetic incident workload;
- the heartbeat schema and stale-run reconciliation rules;
- exact pytest, Ruff, and mypy results;
- the manual visible-TUI smoke-test result; and
- every unmet requirement as an explicit unresolved quality gap.
