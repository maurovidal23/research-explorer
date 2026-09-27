# Research Explorer TUI — Implementation Specification

Status: implemented

This document is the implementation source of truth for the Research Explorer
terminal dashboard. It replaces the first-release header-tab + text-modal design
with a Convoy-style dashboard while preserving the UI-independent event,
projection, and durable-trace architecture.

## 1. Outcome

The ACO exploration pipeline runs behind an opt-in terminal dashboard that makes
long runs understandable while they execute and fully inspectable after they
finish. The dashboard reproduces Convoy's visual hierarchy and interaction
behavior in Textual with Research Explorer branding; it does not port Convoy's
TypeScript/OpenTUI implementation or terminology.

A user can:

1. launch an ACO exploration from the existing CLI with `--tui`;
2. see run identity, status, budget, wave, best quality, and winner at a glance;
3. navigate an agent-first execution tree that truthfully represents the actual
   sequential execution;
4. inspect a focused agent/activity card and four content tabs (Research, Paper,
   Evaluation, Events) for the selected scope;
5. open focused evaluation, frontier, paper, narrative, event, and metadata views;
6. open the active tab in a fullscreen reader, use a command palette and complete
   help, and never have footer/palette/help drift apart;
7. revisit historical steps without pausing the live run and restore follow-live;
8. see a stable completed, cancelled, or failed state;
9. reopen a recorded run with `research-explorer replay tui RUN_ID` in a
   read-only dashboard reconstructed from durable events only.

## 2. Product principles

1. Observability is a contract. Every displayed fact comes from a typed event,
   persisted evaluation, or persisted artifact.
2. Live and replay share one projection. The same reducer consumes stored and
   in-memory events.
3. The engine does not depend on Textual. UI dependencies point inward through an
   event-sink boundary.
4. Research decisions are explained with structured, recorded rationale; private
   chain-of-thought is never requested, exposed, or persisted.
5. The screen reflects actual execution. Sequential turns are not presented as
   concurrent work.
6. The terminal remains usable at common widths and degrades clearly at ≤ 84
   columns.
7. Existing report, Obsidian export, tracing, redaction, and non-TUI CLI behavior
   remain valid.

## 3. Invocation and lifecycle

```bash
research-explorer explore <PAPER_ID> "<QUESTION>" --tui
research-explorer replay tui <RUN_ID> [--db PATH]
```

- `--tui/--no-tui` defaults to off; non-TUI remains the default.
- `--tui` with a pipeline other than `aco` fails before providers or databases
  are opened, with a concise message.
- `--output` still writes the normal report after a successful TUI run, exactly
  once, atomically, before the review screen waits.
- The normal replay database and Obsidian export behavior are preserved.
- Exploration runs in an asynchronous worker and publishes through an event
  sink; the TUI never blocks rendering.
- The TUI owns the foreground terminal and restores it after completion,
  cancellation, and uncaught failure.

### 3.1 Exit behavior

- `q` while a run is active opens a confirmation overlay offering **detach view**
  and **cancel run** only when the process can outlive the UI, otherwise **keep
  running** and **cancel run**.
- `q` after a terminal run state exits the application.
- `Ctrl+C` requests graceful cancellation and reports while the operation is
  settling.
- Cancellation closes LLM clients, providers, graph storage, and trace storage,
  and persists a distinct `cancelled` status (never `failed`/`completed`).
- An uncaught error produces a stable failure screen with a redacted summary and
  the run ID; the terminal is not left in an alternate-screen state.
- Replay is read-only: elapsed time is frozen and cancel/detach actions are
  inert.

## 4. Screen layout

```text
┌ compact context line                                                        ┐
├ two-row run header                                                          ┤
├──────────────────────────────┬──────────────────────────────────────────────┤
│ agent-first execution tree   │ focused agent/activity card                  │
│ (selectable, expandable)     ├──────────────────────────────────────────────┤
│                              │ Research │ Paper │ Evaluation │ Events        │
│                              ├──────────────────────────────────────────────┤
│                              │ tabbed content pane                          │
├──────────────────────────────┴──────────────────────────────────────────────┤
│ bordered adaptive footer: branding + prioritized key hints                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

- Panels are border-only on the terminal background with dim borders; exactly one
  panel carries the accent focus border.
- At ≤ 84 columns the tree is stacked above the right dashboard in a bounded
  region. Selection, active tab, expansion, and scroll state survive resize.
- The chosen content tab stays stable while tree navigation changes its scope.

### 4.1 Compact context line and header

The context line shows the Research Explorer brand, run ID, seed, pipeline, and
explorer/judge models.

The first header row shows status, elapsed wall-clock time, fetch work consumed
and maximum, current wave, current turn, best quality, winner, and known token
usage or cost. Unknown tokens/cost are shown as `unavailable`, never `0`.

The second header row shows the ellipsized question plus colony size, active-slot
configuration, and papers per evaluation (`aco.k_per_turn`). The complete
question and identifiers are available from the metadata view. A historical
selection annotates the header.

### 4.2 Agent-first execution tree

The left pane is a selectable, scrollable tree whose roots are colony agents (in
colony order). Each agent root shows label, caste, Q, delta, winner state, and a
lifecycle glyph + label. Under each agent, waves nest turns, and turns nest paper
steps, discoveries, frontier evaluations, evaluation results, and warnings.

- Wave `0` (seed discovery) is labeled `Seed`.
- Completed historical waves remain available.
- Selecting a node freezes the right-hand scope on it and leaves follow-live
  mode; `Home` restores follow-live and jumps to the newest active node.
- `Enter` expands or collapses the selected node.
- The tree derives its shape from the projection only. Old traces that omit
  `agent_id`/`wave`/`turn` on some events derive the missing context from the
  surrounding turn events; duplicate delivery never duplicates nodes.

### 4.3 Focused agent/activity card

The upper-right card renders the selected agent and step: lifecycle state, current
action, paper title/ID, operation, model, elapsed time, Q and delta, budget,
frontier, and whether the scope is live or a historical snapshot.

### 4.4 Tabbed content pane

Exactly four primary tabs:

- **Research** — the evolving Markdown research line plus the recorded
  understanding delta (explicit delta event when present, otherwise the latest
  evaluation's Q change).
- **Paper** — metadata (title, year, authors, source paper, traversal mode,
  provider), structured analysis, selection rationale, and the recorded
  decision-time frontier ranking.
- **Evaluation** — Q history, S/P/J/R and weights, self and peer rationales,
  judge coverage/gaps, and structural metrics. A skipped evaluation is shown as
  skipped, never as a zero score.
- **Events** — reverse-chronological structured events with agent and outcome
  filtering; content payloads are omitted and secrets stay redacted.

Missing data is marked unavailable; malformed/partial analysis degrades field by
field and never crashes rendering.

## 5. Interaction model

One action registry (`research_explorer.tui.actions`) is the single source for
Textual bindings, footer hints, the command palette, and the shortcut help.

| Key | Action |
|---|---|
| `1`–`4` | select Research / Paper / Evaluation / Events |
| `Tab` / `Shift+Tab` | next / previous content tab |
| `←` / `→` | move focus between tree and content |
| `↑` / `↓` / `PgUp` / `PgDn` | navigate the focused panel |
| `Enter` | expand/collapse the selected tree node |
| `Home` | restore follow-live and jump to the newest node |
| `e` / `f` / `p` / `n` | evaluation / frontier / paper / narrative |
| `l` / `r` | events / metadata |
| `o` / `a` | cycle the Events outcome / agent filter |
| `v` | open the active tab in a fullscreen styled reader |
| `Ctrl+P` | command palette |
| `?` | complete shortcut help |
| `Escape` | close the topmost view |
| `t` | toggle the visible pane on compact terminals |
| `q` / `Ctrl+C` | quit flow / graceful cancellation |

Tree rows and tab labels are clickable and panels support wheel scrolling.

## 6. Projection and replay

- `RunEvent`, `EventSink`/`ChannelSink`/`CompositeSink`, and `RunProjection`
  remain UI-independent; the Textual layer never reads `Orchestrator`, `Colony`,
  `ExplorerAgent`, or `GraphStore` internals.
- The durable SQLite trace store remains authoritative. The UI channel is
  bounded and never stalls research; the projection is idempotent by durable
  sequence and tolerates unknown/future event types.
- `research_explorer.events.navigation` derives the agent-first tree from the
  projected state without durable events.
- Transient interaction state (active tab, focus, expansion, selection, scroll,
  follow, filters) lives in `UISession`, never in durable run events.
- Candidate, LLM, paper, frontier, and evaluation events are enriched with
  consistent `agent_id`, `oleada`, and `turn` fields where missing. The ACO
  scheduler sets an ambient turn context on the tracer, and the explorer adds
  defaults; explicit payload keys still win.
- `replay tui` validates the run, projects stored events, hydrates missing
  evaluation and narrative detail from `evaluation_results` and `artifacts`
  without duplication, and opens the same dashboard read-only. It never opens a
  competing TUI write connection.

## 7. Module boundaries

- `research_explorer.events` — UI-independent event models, sinks, projection,
  and navigation derivation.
- `research_explorer.tui.app` — the Textual application and widgets.
- `research_explorer.tui.actions` — the action registry.
- `research_explorer.tui.session` — transient UI session state.
- `research_explorer.tui.theme` — semantic colors and status glyphs.
- `research_explorer.tui.text` — Textual-free renderers (including the legacy
  helpers still shared with reports/tests).
- `research_explorer.tui.replay` — replay loading and hydration.
- No provider, graph, evaluation, or ACO module imports Textual.

## 8. Testing requirements

- Projection/navigation tests: run and budget progression, sequential execution
  representation, wave/turn grouping, best-agent changes, skipped evaluation,
  transient/fatal failures, duplicate idempotence, unknown events, partial
  analysis, missing context derivation, and redaction.
- Textual pilot tests without real providers/LLM: initial layout, agent-first
  navigation, all four tabs, expansion, follow/manual/Home, focused views,
  fullscreen reader, command palette, help, filters, mouse selection, wheel
  scrolling, compact layout, resize preservation, completed/failed/cancelled
  states, quit/cancel flows, and terminal-safe shutdown.
- Integration: live channel updates the projection; runner failure is redacted;
  replay hydrates and freezes; `replay tui` validates missing runs.
- Required checks: `uv run pytest tests/ -q`, `uv run ruff check src/ tests/`,
  `uv run mypy src/`.

## 9. Out of scope

- A run-history picker, attaching to another process, runtime config mutation.
- TUI support for the single-agent research-kernel pipeline.
- A graphical node-link citation graph or pixel-perfect Convoy reproduction.
- Display or storage of hidden model reasoning.
