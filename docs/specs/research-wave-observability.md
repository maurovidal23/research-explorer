# Wave-Oriented Research Lifecycle and TUI — Implementation Specification

Status: ready for implementation

This specification follows `research-tui.md`, `research-tui-live-reliability.md`,
and `research-tui-memory-safety.md`. It is based on the observed arXiv-only run
`1c7cc5e0b6da`, which completed after 14 minutes and 340 events with a visible
quality score of `0.0809`, but made evaluation difficult to find and accepted an
empty winning narrative as a normal successful result.

## 1. Outcome

Make the research lifecycle understandable at wave level. A user must be able to
see, without reading provider or reference-mapping events, what every selected
agent researched, whether it was evaluated, how its score changed, which agent
leads, and why the system continues or stops.

Change an ACO wave into three explicit barriers: Research, Evaluation, and
Decision. The default TUI must use waves as its primary hierarchy while retaining
agent, paper, mapping, and raw-event detail on demand. A run with no usable final
narrative must never be presented as an ordinary successful completion.

## 2. Lifecycle contract

### WAVE-1 — Explicit wave phases

Every non-seed wave has exactly three ordered phases:

1. **Research** — select at most `max_concurrent` active agents and let each
   selected agent fetch and integrate up to `k_per_turn` papers.
2. **Evaluation** — after every selected research turn has settled, evaluate every
   eligible selected agent against the same completed research-phase snapshot.
3. **Decision** — update pheromones, rank agents, select the current winner, update
   convergence state, and decide whether another wave starts.

Seed discovery remains a separate Setup phase identified as wave `0`; it is not
presented as an agent turn or evaluation wave.

### WAVE-2 — Barrier and concurrency semantics

- No evaluation may start while any selected agent's research turn is running.
- Research turns may run concurrently up to `max_concurrent` when shared graph and
  agent-state safety permits. If the implementation retains sequential research,
  it must preserve the same phase barrier and document the reason in the delivery
  report.
- Evaluation inputs are frozen after the Research barrier. Peer voting must not
  depend on evaluation order within the wave.
- Evaluations may run concurrently when they only consume the frozen snapshot.
- A failure in one agent is contained, represented in that agent's phase status,
  and does not erase successful work from other agents.
- An agent with no new evaluable evidence receives `skipped` with a stable reason;
  it must not silently disappear from the Evaluation phase.
- Pheromone, ranking, winner, and convergence mutations happen only in Decision.

### WAVE-3 — Durable phase telemetry

Add backward-compatible events or payload fields sufficient to reconstruct:

- wave and phase started/completed/failed state;
- selected, completed, failed, and skipped agent counts;
- per-agent research status, papers attempted/integrated, and evidence added;
- per-agent evaluation status, skip/failure reason, S/P/J/R/Q, and Q delta;
- Decision ranking, leader, budget use, convergence counters, and continue/stop
  reason;
- phase and wave elapsed time.

Events must preserve durable ordering and remain readable by old replay databases.
Do not persist raw model responses, hidden reasoning, credentials, authorization
data, or full paper text in telemetry.

### WAVE-4 — Evaluation invariants

- Each selected agent reaches one terminal Evaluation state: `complete`, `skipped`,
  or `failed`.
- `complete` requires all configured score components to have an explicit result;
  unavailable components carry a reason rather than looking like an unexplained
  zero.
- The TUI and report distinguish a genuine numeric zero from unavailable, skipped,
  timed out, parse-failed, or model-failed evaluation.
- Q is calculated only from valid component states according to the documented
  quality policy. Any fallback is explicit and replayable.
- Evaluation totals in the wave summary reconcile with the selected-agent roster.

### WAVE-5 — Honest terminal outcomes

A normal successful result requires:

- a non-empty, normalized winning narrative;
- at least one evidence-bearing evaluated turn;
- a terminal evaluation for the winner; and
- a durable stop reason.

If exploration completed but these conditions are not met, persist and render
`completed` with `outcome=degraded` and one stable reason such as
`empty_winner_narrative`, `no_evaluated_evidence`, or `winner_evaluation_missing`.
Do not fabricate a winner or score. Provider, orchestration, persistence, and
unexpected model failures retain their existing failed-run semantics.

The Markdown report must use the same outcome, reason, winner, and evaluation data
as the durable trace.

## 3. TUI information architecture

### TUI-WAVE-1 — Wave-first navigation

The primary navigation hierarchy is:

```text
Setup
Wave 1
  Research
  Evaluation
  Decision
Wave 2
  Research
  Evaluation
  Decision
Final result
Debug
```

Do not repeat the same wave independently beneath every agent. Agent, paper,
reference-mapping, provider-call, and individual event detail remains reachable
from the selected phase or Debug view.

### TUI-WAVE-2 — Always-visible run summary

The top summary remains visible across navigation and includes:

- run status and outcome;
- current wave and phase;
- phase progress such as `2/3 agents`;
- elapsed time, fetch budget, and API-reported token total;
- current leader and best Q when available; and
- warning/failure count.

It must never imply that tokens are actively increasing when no completion usage
has arrived.

### TUI-WAVE-3 — Compact wave rows

Each wave row shows, without expansion:

- Research completion and paper/evidence totals;
- Evaluation completion/skipped/failed counts and best Q;
- Decision leader, Q delta, and continue/stop reason; and
- active, complete, degraded, or failed state in text, not color alone.

The active wave follows live progress by default. Completed waves remain stable
and inspectable.

### TUI-WAVE-4 — Phase detail

The selected wave presents a compact agent table:

```text
Agent  Paper/current work  Research    Evaluation  Q      Delta
A01    Paper title         complete    complete    0.38   +0.06
A02    Paper title         running     pending     --     --
A03    No new evidence     complete    skipped     --     --
```

Selecting an agent reveals its narrative/evidence, papers, component scores, and
diagnostics. Evaluation detail always exposes S/P/J/R, component availability,
final Q, Q delta, and skip/failure reason.

### TUI-WAVE-5 — Progressive disclosure

- Default views contain research meaning, evaluation, and decisions rather than
  low-level event volume.
- Reference parsing and mapping are summarized as counts with failures and
  traversable results; individual attempts are drill-down detail.
- Provider operations and raw ordered events live under Debug.
- Existing event paging and memory bounds remain enforced; this feature must not
  reintroduce per-event full rendering or unbounded widget growth.

### TUI-WAVE-6 — Final result

On terminal state, Final result becomes the default selection and shows:

- outcome and stop reason;
- winning agent and final S/P/J/R/Q;
- the winning narrative, or an explicit statement that none was produced;
- papers and evidence supporting the result;
- waves, fetches, elapsed time, and token usage; and
- warnings, failed/skipped evaluations, and unresolved quality gaps.

The user can still inspect every completed wave and Debug detail before exiting.

## 4. Compatibility and boundaries

- Preserve the `research-explorer explore ... --tui` CLI and non-TUI behavior.
- Preserve ACO selection size, transition probabilities, graph integration,
  provider routing, reference mapping, and quality weights unless explicitly
  required by the phase barrier.
- Preserve bounded live projection, coalesced rendering, heartbeat recovery,
  cancellation behavior, complete durable traces, and old-trace replay.
- Old traces without phase events are projected into a clearly labeled legacy
  best-effort wave view; do not invent evaluation states that were not recorded.
- Preserve the unsupported research-kernel TUI boundary.
- Do not add comments to production code. Public functions remain typed and all
  I/O remains asynchronous.

## 5. Required tests

Add revert-failing coverage for at least these cases:

1. Two selected agents finish Research before either Evaluation starts.
2. Evaluation order cannot change peer inputs or final wave rankings.
3. Every selected agent obtains exactly one terminal Evaluation state.
4. No-evidence, model failure, parse failure, and timeout are visibly distinct from
   a genuine zero score.
5. Agent failure is contained and Decision uses only valid completed results.
6. Pheromone, ranking, winner, and convergence updates occur only after the
   Evaluation barrier.
7. Projection reconstructs Setup and Research/Evaluation/Decision phases from a
   new trace and handles an old trace deterministically.
8. The default TUI tree is wave-first and does not duplicate waves per agent.
9. Live progress shows current phase, selected-agent progress, evaluation status,
   leader, Q delta, budgets, and stop reason.
10. Research/provider/reference event bursts do not hide or evict durable wave and
    evaluation summaries.
11. Terminal navigation selects Final result while retaining wave drill-down.
12. Empty winning narrative, missing evaluated evidence, and missing winner
    evaluation each produce the expected completed-degraded outcome.
13. A valid narrative and terminal evaluation still produce a normal completed
    outcome and unchanged Markdown report content.
14. Cancellation, failure, replay equivalence, memory bounds, old databases,
    non-TUI ACO, and research-kernel regressions remain green.

Include a frozen, network-free regression shaped like run `1c7cc5e0b6da`: arXiv
seed setup, visible reference-mapping summaries, one evaluated turn with
S/P/J unavailable or zero, structural score present, and an empty narrative. It
must end completed-degraded and make the reason unmistakable in both TUI and
report.

## 6. Verification

```bash
uv sync --extra dev
uv run pytest tests/ -q
uv run ruff check src/ tests/
uv run mypy src/
```

After deterministic checks pass, run a manual Orca smoke test using a bounded,
network-free fixture first. Confirm that Setup, Research, Evaluation, Decision,
and Final result are visible and that the completed screen remains open. A live
arXiv-only smoke run may follow, but it must be bounded and is not required by CI.

## 7. Convoy execution contract

Use the repository's `reliability-full-cycle` pipeline. Treat this specification
as one release-blocking change and iterate until the independently verified score
reaches at least 90.

The Convoy delivery report must include:

- the implemented Research/Evaluation/Decision barrier semantics;
- whether research and evaluation are sequential or concurrent and why;
- the durable phase-event and evaluation-state contracts;
- old-trace projection behavior;
- the terminal outcome invariants and empty-narrative regression proof;
- before/after TUI hierarchy captures or equivalent pilot assertions;
- exact pytest, Ruff, mypy, and manual visible-TUI smoke-test results; and
- every unmet requirement as an explicit unresolved quality gap.
