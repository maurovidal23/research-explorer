# Research Explorer TUI — Implementation Specification

Status: ready for implementation

## 1. Outcome

Implement a live terminal interface for the ACO research pipeline that makes long
exploration runs understandable while they execute and debuggable afterward. The
interface should use Convoy's compact visual hierarchy as a design reference without
copying its terminology or implementation.

The first release must let a user:

1. launch an ACO exploration from the existing CLI with a TUI;
2. understand the run identity, budget, current wave, and best quality at a glance;
3. switch between agents from a tab strip in the header;
4. follow discovery and evaluation steps in a left-side timeline;
5. inspect the selected agent's current paper, selection rationale, paper contribution,
   and evolving research line;
6. inspect failures and detailed evaluation results without leaving the TUI;
7. revisit earlier steps while a run is active;
8. see a stable completed or failed state when execution ends; and
9. reconstruct the same views from persisted run events rather than relying on mutable
   orchestrator objects.

The first release targets the ACO pipeline. It must not regress the existing
`research-kernel` pipeline or the non-TUI CLI.

## 2. Product principles

1. Observability is a contract. Every displayed fact comes from a typed event,
   persisted evaluation, or persisted artifact.
2. Live and replay share one projection. The same reducer/view model consumes stored
   and in-memory events.
3. The engine does not depend on Textual. UI dependencies point inward through an
   event-sink boundary.
4. Research decisions are explained with structured, recorded rationale. The product
   does not request, expose, or persist private chain-of-thought.
5. The screen reflects actual execution. Sequential turns must not be presented as
   concurrent work.
6. The terminal remains usable at common widths and degrades clearly at small widths.
7. Existing report, Obsidian export, tracing, redaction, and CLI behavior remain valid.

## 3. Scope

### 3.1 Required

- A Textual-based terminal application.
- A `--tui/--no-tui` option on `research-explorer explore`.
- Non-TUI behavior remains the default unless project conventions or existing tests
  establish that an interactive-terminal default is safe and backward compatible.
- An async-safe live event stream from orchestration to the interface.
- Durable event recording remains enabled during TUI runs.
- Global header, agent tabs, execution timeline, agent detail pane, focused overlays,
  status/footer help, completion state, and failure state.
- Keyboard-only navigation.
- Unit tests for event projection and view-model behavior.
- Textual pilot tests for critical interaction paths.
- CLI regression tests for TUI selection and non-TUI compatibility.
- Documentation in the README or CLI help.

### 3.2 Not required in the first release

- Runtime mutation of research configuration.
- Pausing in-flight HTTP or LLM requests.
- Attaching to a run owned by another process.
- A graphical node-link citation graph.
- Mouse-first interaction.
- TUI support for the single-agent research-kernel pipeline.
- Pixel-perfect reproduction of Convoy.
- Display or storage of hidden model reasoning.

## 4. Invocation and lifecycle

The supported entry point is:

```bash
uv run research-explorer explore <PAPER_ID> "<QUESTION>" --tui
```

All existing exploration options remain available, including `--config`, `--output`,
and `--pipeline`. Requesting `--tui` with a pipeline that is not yet supported must
fail before providers or databases are opened, with a concise actionable message.

The TUI owns the foreground terminal while the exploration runs in an asynchronous
worker. It must restore the terminal after normal completion, cancellation, and
uncaught failure.

When `--output` is supplied, the normal final report is still written after a
successful TUI run. A TUI run must also preserve the normal replay database and
Obsidian export behavior.

### 4.1 Exit behavior

- `q` while a run is active opens a confirmation overlay offering **detach view** and
  **cancel run** only when detaching can safely leave the process running. If the
  current process cannot outlive the UI, offer **keep running** and **cancel run**.
- `q` after a terminal run state exits the application.
- `Ctrl+C` requests graceful cancellation and clearly reports whether the current
  operation is still settling.
- Cancellation closes LLM clients, providers, graph storage, and trace storage.
- A cancelled run is persisted with a distinct `cancelled` status rather than
  `failed` or `completed`.
- An uncaught error produces a stable failure screen with a redacted error summary and
  the run ID; it must not leave the terminal in an alternate-screen state.

## 5. Screen layout

The normal layout has five regions:

```text
┌ global run header                                                        ┐
├ horizontally scrollable agent tabs                                      ┤
├──────────────────────────────┬───────────────────────────────────────────┤
│ budget + execution timeline  │ selected-agent detail                     │
│                              │                                           │
│                              │                                           │
├──────────────────────────────┴───────────────────────────────────────────┤
│ contextual keys, status, transient notifications                        │
└──────────────────────────────────────────────────────────────────────────┘
```

At widths below the two-column breakpoint, the application switches to one visible
pane at a time rather than rendering clipped or overlapping content. The user can
toggle timeline and detail panes. The exact breakpoint may follow Textual layout
behavior but must be covered by a test.

### 5.1 Global header

The first header row displays:

- run status: initializing, running, evaluating, converged, exhausted, completed,
  cancelled, or failed;
- elapsed wall-clock time;
- global fetch work consumed and maximum;
- current wave;
- current turn when one is active;
- best quality score;
- winning agent when one exists; and
- known token usage or cost only when backed by recorded data.

Unknown token usage or cost is shown as unavailable, never as zero.

The second header row displays:

- seed paper identifier;
- research question, ellipsized to available width;
- pipeline;
- explorer model and judge model;
- colony size; and
- active-slot configuration.

The complete question and identifiers must be available from a details overlay.

### 5.2 Agent tabs

Agent tabs appear directly below the global header and are the primary navigation for
the detail pane. They scroll horizontally and use short stable display labels such as
`A01`, while preserving the full agent ID in details.

Each tab communicates:

- short agent label;
- caste: foundations, impact, or mixed, using the existing stored values even if they
  are currently Spanish identifiers;
- current Q;
- delta Q when available;
- lifecycle state; and
- whether it owns the best recorded narrative.

Lifecycle indicators must distinguish at least active, evaluating, waiting,
exhausted/completed, and failed. Color cannot be the only distinction.

Changing the selected tab changes the detail pane and optionally follows that agent's
latest timeline item. It never changes scheduler behavior.

### 5.3 Budget and execution timeline

The left pane starts with a global progress bar based on authoritative consumed and
maximum work. It includes an exact textual fraction so information is not color-only.
When the configured stopping rule is time or convergence, the pane presents the
relevant primary measure and still shows known fetch work as a secondary value.

Below the bar is a selectable, scrollable hierarchy:

```text
Wave 6
├─ A01 foundations
│  ├─ completed  read  Paper A
│  ├─ completed  read  Paper B
│  ├─ active     read  Paper C
│  ├─ pending    read  4/4
│  └─ pending    evaluate
├─ A02 impact
│  └─ waiting
└─ A03 mixed
   └─ waiting
```

The hierarchy contains waves, agent turns, paper discovery/read steps, evaluation,
and significant warnings. Completed historical waves remain available. Selecting an
item freezes the detail pane on that event until follow mode is restored.

The number of papers grouped before an ACO evaluation is the existing
`aco.k_per_turn` value. The interface labels it **papers per evaluation**. The TUI must
not introduce a second source of configuration truth.

The current scheduler processes selected agents sequentially for peer-voting
semantics. The timeline must therefore show the actual active agent and the queued or
waiting state of the others; `max_concurrent` alone is not evidence that turns ran
simultaneously.

### 5.4 Selected-agent detail

The main pane shows four sections for the selected live or historical step.

#### Current activity

- action state;
- paper title and normalized ID;
- year and authors when available;
- source paper;
- traversal mode (`ref` or `cites`);
- provider;
- explorer or judge model used for the active operation;
- operation elapsed time when known; and
- budget/frontier state after the latest completed step.

#### Selection rationale

- chosen candidate score or probability;
- eta total and available eta components: semantic similarity, citations, recency,
  provider confidence, and optional LLM priority;
- pheromone value;
- direction weight;
- exploration versus exploitation selection; and
- a ranked view of recorded alternatives.

Missing components are marked unavailable. They are never inferred from unrelated
state.

#### Paper contribution

Render the structured per-paper analysis already represented by `paper_analyses`:

- summary;
- key concepts;
- methods;
- findings;
- relevance;
- limitations; and
- key references.

Malformed or partial analysis must degrade field by field and must not crash the TUI.

#### Research line

Render the selected agent's accumulated narrative as a living research synthesis. If
an event contains an explicit delta or changed-understanding summary, show it before
the full narrative. Preserve Markdown semantics where Textual supports them.

The detail pane clearly indicates when it displays a historical snapshot instead of
the live head.

## 6. Focused views and key bindings

The following bindings are required:

| Key | Action |
|---|---|
| `Left` / `Right` | Select previous or next agent tab |
| `Up` / `Down` | Select timeline item |
| `Enter` | Open details for selected item |
| `Home` | Restore follow-live mode and jump to the newest item |
| `e` | Open latest detailed evaluation for the selected agent |
| `f` | Open the recorded frontier/candidate ranking |
| `p` | Open the full selected-paper analysis |
| `n` | Open the complete agent narrative |
| `l` | Open run logs/events filtered to the selected agent where possible |
| `r` | Open run metadata and resolved configuration |
| `?` | Open complete key help |
| `q` | Exit flow described in section 4.1 |

Focused views may use modals, overlays, or replaceable content panes. They must be
scrollable and must preserve the prior timeline/agent selection when closed.

### 6.1 Evaluation view

The evaluation view displays:

- final Q, old Q, and delta Q;
- configured weights;
- S/P/J/R component scores;
- self-assessment rationale;
- each peer vote with voter and rationale;
- virgin-judge coverage and gaps; and
- structural coverage, diversity, depth, coherence, and aggregate R.

An evaluation skipped for lack of new evidence is shown as skipped, not as a zero
score.

### 6.2 Frontier view

The frontier view displays recorded candidates ordered by the decision-time ranking,
including paper ID, source, traversal mode, eta components, pheromone, total weight,
and chosen status. It is a historical decision record, not a recomputation using
current graph state.

### 6.3 Event/log view

The event view displays structured events rather than reading application log files.
It supports filtering by severity-like outcome and agent. Secrets and sensitive query
parameters remain redacted through the existing redaction boundary.

## 7. Event and projection architecture

Introduce a UI-independent event publishing contract. Names may follow repository
conventions, but the semantics must be equivalent to:

```python
class EventSink(Protocol):
    def publish(self, event: RunEvent) -> None: ...
```

An implementation may be synchronous at the publish boundary as long as it never
blocks orchestration on terminal rendering. A composite sink fans out to:

1. the durable SQLite trace store; and
2. a bounded, async-safe in-memory channel consumed by the TUI.

The durable store remains authoritative for completed history. Backpressure in the UI
channel must not stall research. If intermediate display-only notifications are
coalesced, durable semantic events must still be written.

A pure projection component reduces ordered events into `RunViewState`. The Textual
widgets render that state and do not read `Orchestrator`, `Colony`, `ExplorerAgent`, or
`GraphStore` internals directly.

At minimum, projected state contains:

- run metadata and lifecycle;
- resolved budget and consumption;
- current wave and turn;
- best score and winning agent;
- ordered agent summaries;
- hierarchical timeline entries;
- current and historical paper activity;
- candidate decision records;
- evaluation records;
- narrative/paper-analysis artifact references; and
- redacted failures.

Projection must tolerate duplicate delivery without duplicating timeline nodes. Events
are ordered by their durable per-run sequence. Gaps or unrecognized future event types
are retained in the event view and do not crash rendering.

## 8. Required event coverage

Reuse compatible replay records and extend tracing where necessary. Every agent event
must consistently include `agent_id`, wave, turn, and relevant paper ID.

Required lifecycle events or equivalent typed semantics:

- run started;
- seed routing started/completed/failed;
- colony initialization started/completed;
- wave started/completed;
- agent turn queued/started/completed/failed;
- candidate set scored;
- candidate selected, including selection mode and decision-time alternatives;
- paper fetch started/completed/failed;
- paper integration started/completed/failed;
- neighbor discovery started/completed/failed;
- frontier reference evaluation started/completed/failed;
- quality evaluation started/completed/skipped/failed;
- new best recorded;
- budget snapshot;
- run completed/cancelled/failed.

LLM operation events record purpose, model, start/end state, elapsed time, and token
usage when the API provides it. They must not record API keys, authorization headers,
or hidden reasoning. Existing explicit evaluation rationales and structured paper
analysis remain valid display data.

The migration must preserve compatibility with existing trace databases. Additive
payload fields and new event types are preferred over destructive schema changes.

## 9. Dependency and module boundaries

- Add Textual as a bounded compatible project dependency.
- Place TUI code in a cohesive package such as `research_explorer.tui`.
- Keep event models and publishing abstractions outside Textual-specific modules when
  they are also used by orchestration or replay.
- Keep provider, graph, evaluation, and ACO modules free of imports from Textual.
- Public functions and methods require type hints.
- Follow the repository rule against code comments unless a comment is explicitly
  necessary to explain a non-obvious safety invariant.

## 10. Rendering and accessibility

- Target the existing dark terminal aesthetic; do not assume a specific terminal
  palette.
- Use status text or symbols in addition to color.
- Long paper titles, IDs, and questions ellipsize in compact views and remain
  accessible in full views.
- Empty states explain whether data is pending, unavailable, skipped, or absent.
- Resizing must not raise or corrupt selection.
- The UI remains functional without mouse support.
- Animation is restrained and disabled or coalesced when it would make tests or remote
  terminals unstable.

## 11. Failure and data-integrity behavior

- Provider transient failures remain distinct from definitive absence.
- Agent-turn containment remains intact: one failed agent cannot terminate unrelated
  turns.
- Partial and malformed LLM payloads render safely.
- Trace-write failure is surfaced as a durability failure and handled according to the
  engine's existing correctness guarantees; the TUI must not claim a replayable run
  when persistence failed.
- UI rendering failure must request orderly application shutdown without corrupting
  the recorded run.
- All displayed error strings pass through secret redaction.
- The TUI never opens its own competing write connection for graph or trace mutation.

## 12. Testing requirements

### 12.1 Projection tests

Use deterministic event fixtures to verify:

- run and budget progression;
- agent tab ordering and status transitions;
- sequential agent execution representation;
- discovery/evaluation grouping by wave and turn;
- best-agent changes;
- skipped evaluation semantics;
- transient and fatal failures;
- historical selection versus follow-live mode;
- duplicate delivery idempotence;
- unknown event compatibility; and
- partial paper-analysis payloads.

### 12.2 Textual application tests

Use Textual's test pilot without real providers or LLM calls to verify:

- initial layout;
- agent navigation;
- timeline navigation;
- evaluation, frontier, paper, narrative, event, metadata, and help views;
- restoration of selection after closing a focused view;
- narrow-terminal fallback;
- completed and failed screens;
- quit/cancellation confirmation; and
- terminal-safe shutdown.

Assertions should target semantic widget state and text rather than brittle full-screen
snapshots.

### 12.3 Integration and regression tests

- Run a deterministic fake ACO exploration through the live event channel and assert
  the final projection.
- Verify every required semantic event persists and replays to an equivalent final
  projection.
- Verify `explore --tui` selects the TUI path.
- Verify existing `explore` output behavior without `--tui`.
- Verify unsupported-pipeline rejection occurs before resource construction.
- Verify `--output` still writes the report for a completed TUI run.
- Verify cancellation persists `cancelled` and closes resources.
- Verify sentinel secrets never appear in projected failures or event views.

The standard repository checks must pass:

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/
uv run mypy src/
```

## 13. Acceptance criteria

The implementation is complete when all of the following are true:

1. `uv run research-explorer explore <id> "<question>" --tui` launches a real Textual
   application and executes the ACO run without blocking rendering.
2. The header shows authoritative run identity, elapsed time, budget, wave, models,
   and best-quality information.
3. Every colony agent has a navigable header tab with caste, score, delta, state, and
   winner indication.
4. The left pane groups actual paper steps and evaluation under their wave and agent
   turn, using `aco.k_per_turn` as papers per evaluation.
5. The main pane exposes structured selection rationale, paper contribution, and the
   evolving research line without exposing hidden chain-of-thought.
6. Detailed evaluation and decision-time frontier data are inspectable.
7. Selecting historical steps does not stop the live run, and follow-live mode can be
   restored.
8. Completed, cancelled, and failed runs produce distinct stable states.
9. The persisted event stream reconstructs the same terminal view state produced by
   the live channel for deterministic fixtures.
10. The original non-TUI command and research-kernel path retain their tested
    behavior.
11. Tests, Ruff, and mypy pass.

## 14. Implementation order

Implement in this order to keep the system usable throughout the change:

1. Define normalized run events, event sink, and pure projection/view-state models.
2. Adapt existing trace emission and add missing operation/budget lifecycle events.
3. Add deterministic projection and replay tests.
4. Build the Textual static layout and fixture-driven interaction tests.
5. Connect the bounded live channel and background exploration worker.
6. Integrate the CLI lifecycle, output generation, cancellation, and cleanup.
7. Add focused evaluation/frontier/artifact/event views.
8. Complete responsive behavior, documentation, and full regression checks.

