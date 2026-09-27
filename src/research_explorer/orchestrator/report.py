"""Exploration report generator — produces a full markdown report after a run.

Captures: run configuration, summary stats, the winning narrative, quality
evolution per oleada, per-agent details (path, private graph, quality), all
discovered papers, and the colony's aggregate graph.
"""

from __future__ import annotations

from collections import defaultdict

from research_explorer.aco.colony import Colony
from research_explorer.aco.convergence import ConvergenceChecker
from research_explorer.aco.scheduler import Scheduler
from research_explorer.config import Config
from research_explorer.graph.store import GraphStore


def _fmt_papers_list(ids: list[str], graph: GraphStore, limit: int = 200) -> str:
    lines: list[str] = []
    for nid in ids[:limit]:
        s = graph.get_paper_summary(nid)
        if s is not None:
            title = s.title or "(no title)"
            year = s.year or "?"
            lines.append(f"  - `{nid}` — {title} ({year})")
        else:
            lines.append(f"  - `{nid}` — (metadata not fetched)")
    if len(ids) > limit:
        lines.append(f"  - ... and {len(ids) - limit} more")
    return "\n".join(lines)


def build_report(
    config: Config,
    colony: Colony,
    scheduler: Scheduler,
    convergence: ConvergenceChecker,
    graph: GraphStore,
    seed_paper_id: str,
    seed_query: str,
    elapsed: float,
    outcome: str = "",
    terminal_reason: str = "",
) -> str:
    parts: list[str] = []

    # ---- Header ----------------------------------------------------------
    parts.append("# Research Explorer — Exploration Report\n")
    parts.append(f"- **Seed paper:** `{seed_paper_id}`")
    parts.append(f"- **Research line:** {seed_query}")
    parts.append(f"- **Default provider:** {config.providers.default}")
    parts.append(f"- **Active providers:** {', '.join(config.providers.active)}")
    parts.append(f"- **Colony size:** {config.aco.colony_size}")
    parts.append(f"- **Concurrent slots (K):** {config.aco.max_concurrent}")
    parts.append(f"- **Fetches per turn (k):** {config.aco.k_per_turn}")
    parts.append(f"- **Budget:** {config.budget.max_fetches} fetches / {config.budget.max_time_seconds}s")
    parts.append(f"- **LLM model:** {config.llm.explorer_model}")
    parts.append("")

    # ---- Summary ---------------------------------------------------------
    parts.append("## Summary\n")
    parts.append(f"- **Outcome:** {outcome or '(unspecified)'}")
    if terminal_reason:
        parts.append(f"- **Terminal reason:** {terminal_reason}")
    parts.append(f"- **Winner:** {colony.best_snapshot_agent or '(none)'}")
    parts.append(f"- **Peak Q:** {colony.best_quality:.4f}")
    parts.append(f"- **Peak at oleada:** {colony.best_snapshot_oleada}")
    parts.append(f"- **Total oleadas:** {scheduler.oleada_count}")
    parts.append(f"- **Total fetches:** {scheduler.total_fetches}")
    parts.append(f"- **Elapsed:** {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    parts.append(f"- **Active agents at end:** {len(colony.active_candidates())}")
    if colony.best_agent is None:
        reason = terminal_reason or "the initial frontier had no traversable candidates"
        parts.append(
            "- **No winning narrative was produced.** The run completed as a "
            f"degraded result: {reason}."
        )
    parts.append("")

    # ---- Winning narrative ----------------------------------------------
    parts.append("## Winning Narrative\n")
    parts.append(colony.best_narrative or "(no narrative produced)")
    parts.append("")

    # ---- Quality evolution ----------------------------------------------
    parts.append("## Quality Evolution\n")
    parts.append("| Oleada | Best Q | Fetches | Elapsed (s) | Top agents |")
    parts.append("|--------|--------|---------|-------------|------------|")
    for h in scheduler.history:
        ranking_str = ", ".join(f"{aid}={q:.2f}" for aid, q in h["ranking"][:3])
        parts.append(
            f"| {h['oleada']} | {h['best_Q']:.4f} | {h['fetches']} | {h['elapsed']} | {ranking_str} |"
        )
    parts.append("")

    # ---- Per-agent details ----------------------------------------------
    parts.append("## Agent Details\n")
    for agent in colony.agents:
        s = agent.state
        parts.append(f"### {s.id} (caste: {s.caste})\n")
        parts.append(f"- **Final Q:** {s.quality:.4f}")
        parts.append(f"- **Turns:** {s.turn_count}")
        parts.append(f"- **Budget remaining:** {s.budget}")
        parts.append(f"- **Papers visited:** {len(s.visited)}")
        parts.append(f"- **Frontier remaining:** {len(s.frontier)}")

        n_refs = sum(len(v) for v in s.local_refs.values())
        n_cits = sum(len(v) for v in s.local_cits.values())
        parts.append(f"- **Private graph:** {len(s.local_refs)} nodes with refs ({n_refs} ref-edges), "
                     f"{len(s.local_cits)} nodes with cits ({n_cits} cit-edges)")
        parts.append(f"- **Private pheromone entries:** {len(s.local_pheromone)}")
        parts.append(f"- **Pheromone concentration:** {s.pheromone_concentration():.3f}")

        if s.full_path:
            path_str = " → ".join(f"`{pid}` ({mode})" for pid, mode in s.full_path)
            parts.append(f"- **Path:** {path_str}")

        if s.visited:
            parts.append(f"- **Visited papers:**\n{_fmt_papers_list(s.visited, graph)}")
        parts.append("")

    # ---- All discovered papers ------------------------------------------
    parts.append("## All Discovered Papers\n")
    paper_agents: dict[str, list[str]] = defaultdict(list)
    for agent in colony.agents:
        for pid in agent.state.visited:
            if agent.state.id not in paper_agents[pid]:
                paper_agents[pid].append(agent.state.id)
    for pid in colony.shared_frontier.papers:
        for agent in colony.agents:
            if agent.state.id not in paper_agents.get(pid, []) and pid in agent.state.visited:
                paper_agents[pid].append(agent.state.id)

    all_summaries = graph.get_all_paper_summaries()
    parts.append(f"Total nodes in shared store: **{len(all_summaries)}**\n")
    parts.append("| ID | Title | Year | Provider | Visited by |")
    parts.append("|----|-------|------|----------|-----------|")
    for sm in all_summaries:
        title = (sm.title or "(no title)")[:80]
        year = sm.year or "?"
        visitors = ", ".join(paper_agents.get(sm.id, [])) or "—"
        parts.append(f"| `{sm.id}` | {title} | {year} | {sm.provider} | {visitors} |")
    parts.append("")

    # ---- Aggregate colony graph -----------------------------------------
    parts.append("## Aggregate Colony Graph\n")
    all_nodes: set[str] = set()
    ref_edges: set[tuple[str, str]] = set()
    cit_edges: set[tuple[str, str]] = set()
    for agent in colony.agents:
        for src, refs in agent.state.local_refs.items():
            all_nodes.add(src)
            for dst in refs:
                all_nodes.add(dst)
                ref_edges.add((src, dst))
        for dst, cits in agent.state.local_cits.items():
            all_nodes.add(dst)
            for src in cits:
                all_nodes.add(src)
                cit_edges.add((src, dst))
    parts.append(f"- **Total nodes (union across agents):** {len(all_nodes)}")
    parts.append(f"- **Reference edges (union):** {len(ref_edges)}")
    parts.append(f"- **Citation edges (union):** {len(cit_edges)}")
    parts.append(f"- **Total edges:** {len(ref_edges) + len(cit_edges)}")
    parts.append("")

    # ---- Convergence ----------------------------------------------------
    parts.append("## Convergence\n")
    qh = convergence.state.quality_history
    parts.append(f"- **Quality history length:** {len(qh)}")
    if qh:
        parts.append(f"- **First Q:** {qh[0]:.4f}")
        parts.append(f"- **Last Q:** {qh[-1]:.4f}")
        parts.append(f"- **Max Q:** {max(qh):.4f}")
        if len(qh) >= 2:
            parts.append(f"- **Improvement:** {qh[-1] - qh[0]:+.4f}")
    parts.append("")

    return "\n".join(parts)
