"""Obsidian graph generator — produces a folder of markdown notes with wikilinks.

For the winner agent, generates:
  - One .md file per visited paper (with metadata, key ideas, citation links)
  - A _MOC.md (Map of Content) with the exploration path and full narrative

The [[wikilinks]] between notes create a visualizable graph in Obsidian.
"""

from __future__ import annotations

from pathlib import Path

from research_explorer.aco.frontier import SharedFrontier
from research_explorer.agents.state import AgentState
from research_explorer.graph.store import GraphStore
from research_explorer.logging_setup import get_logger

log = get_logger("obsidian")


def _safe_filename(paper_id: str) -> str:
    """Convert a paper ID to a valid filename (and wikilink target)."""
    return paper_id.replace(":", "_").replace("/", "_").replace("\\", "_")


def _safe_wikilink(paper_id: str) -> str:
    return f"[[{_safe_filename(paper_id)}]]"


def generate_obsidian_graph(
    agent_state: AgentState,
    graph: GraphStore,
    seed_query: str,
    output_dir: str = "obsidian",
    shared_frontier: SharedFrontier | None = None,
) -> str:
    """Generate an Obsidian-compatible graph for an agent.

    Returns the path to the generated folder.
    """
    folder = Path(output_dir) / _safe_filename(agent_state.id)
    folder.mkdir(parents=True, exist_ok=True)

    # Generate one note per visited paper
    for pid in agent_state.visited:
        _write_paper_note(folder, pid, agent_state, graph, shared_frontier)

    # Generate the MOC (Map of Content)
    _write_moc(folder, agent_state, graph, seed_query, shared_frontier)

    log.info(
        "obsidian_graph_generated",
        agent=agent_state.id,
        folder=str(folder),
        papers=len(agent_state.visited),
    )
    return str(folder)


def _write_paper_note(
    folder: Path,
    paper_id: str,
    state: AgentState,
    graph: GraphStore,
    shared_frontier: SharedFrontier | None = None,
) -> None:
    """Write a single paper note with metadata, ideas, and citation links."""
    safe_name = _safe_filename(paper_id)
    summary = graph.get_paper_summary(paper_id)
    paper = graph.get_paper(paper_id)

    title = summary.title if summary else paper_id
    year = summary.year if summary else None
    authors = summary.authors if summary else []
    abstract = summary.abstract if summary else None
    provider = summary.provider if summary else "unknown"
    score = (shared_frontier.scores.get(paper_id, 0.0) if shared_frontier
             else state.frontier_scores.get(paper_id, 0.0))

    # Outgoing references (from agent's private graph)
    ref_ids = state.local_references(paper_id)
    cit_ids = state.local_citants(paper_id)

    ref_links = "\n".join(f"- [[{_safe_filename(r)}]]" for r in ref_ids if r != paper_id) or "(none discovered)"
    cit_links = "\n".join(f"- [[{_safe_filename(c)}]]" for c in cit_ids if c != paper_id) or "(none discovered)"

    # Which papers link TO this one (reverse lookup in agent's private graph)
    referenced_by = []
    for src_id in state.visited:
        if paper_id in state.local_references(src_id):
            referenced_by.append(src_id)
    cited_by = []
    for src_id in state.visited:
        if paper_id in state.local_citants(src_id):
            cited_by.append(src_id)

    ref_by_links = "\n".join(f"- [[{_safe_filename(s)}]]" for s in referenced_by) or "(none)"
    cit_by_links = "\n".join(f"- [[{_safe_filename(s)}]]" for s in cited_by) or "(none)"

    # Full text snippet (if available)
    fulltext_snippet = ""
    if paper and paper.fulltext:
        snippet = paper.fulltext[:500].replace("\n", " ")
        fulltext_snippet = f"## Full Text Snippet\n{snippet}...\n\n"

    # Abstract
    abstract_section = ""
    if abstract:
        abstract_section = f"## Abstract\n{abstract}\n\n"

    # Per-paper analysis from the LLM
    analysis = state.paper_analyses.get(paper_id, {})
    analysis_section = ""
    if analysis:
        summary = analysis.get("summary", "")
        key_concepts = analysis.get("key_concepts", [])
        methods = analysis.get("methods", "")
        findings = analysis.get("findings", "")
        relevance = analysis.get("relevance", "")
        limitations = analysis.get("limitations", "")
        key_refs = analysis.get("key_references", [])

        concepts_str = "\n".join(f"- {c}" for c in key_concepts) if key_concepts else "(none)"
        key_refs_str = "\n".join(
            f"- {r.get('title', '?')}: {r.get('why', '')}"
            for r in key_refs if isinstance(r, dict)
        ) if key_refs else "(none)"

        analysis_section = f"""## Paper Analysis

### Summary
{summary or "(not available)"}

### Key Concepts
{concepts_str}

### Methods
{methods or "(not available)"}

### Findings
{findings or "(not available)"}

### Relevance to Research Line
{relevance or "(not available)"}

### Limitations
{limitations or "(not available)"}

### Key References (from this paper)
{key_refs_str}

"""

    content = f"""---
title: "{title}"
year: {year or "null"}
authors: {authors}
provider: {provider}
exploration_score: {score}
---

# {title}

## Metadata
- **Year:** {year or "Unknown"}
- **Authors:** {", ".join(authors) if authors else "Unknown"}
- **Provider:** {provider}
- **Exploration score:** {score:.3f}

{abstract_section}{analysis_section}{fulltext_snippet}## References (outgoing)
{ref_links}

## Cited by (incoming)
{cit_links}

## Referenced by (in agent's graph)
{ref_by_links}

## Cited this paper (in agent's graph)
{cit_by_links}
"""

    (folder / f"{safe_name}.md").write_text(content, encoding="utf-8")


def _write_moc(
    folder: Path,
    state: AgentState,
    graph: GraphStore,
    seed_query: str,
    shared_frontier: SharedFrontier | None = None,
) -> None:
    """Write the Map of Content (overview of the agent's exploration)."""

    # Exploration path
    path_lines: list[str] = []
    for i, (pid, mode) in enumerate(state.full_path, 1):
        summary = graph.get_paper_summary(pid)
        title = summary.title if summary else pid
        path_lines.append(f"{i}. [[{_safe_filename(pid)}]] ({mode}) — {title}")

    path_section = "\n".join(path_lines) or "(no path)"

    # All visited papers
    visited_lines: list[str] = []
    for pid in state.visited:
        summary = graph.get_paper_summary(pid)
        title = summary.title if summary else pid
        score = (shared_frontier.scores.get(pid, 0.0) if shared_frontier
                 else state.frontier_scores.get(pid, 0.0))
        visited_lines.append(f"- [[{_safe_filename(pid)}]] (score: {score:.3f}) — {title}")

    visited_section = "\n".join(visited_lines) or "(none)"

    # Private graph stats
    total_ref_edges = sum(len(refs) for refs in state.local_refs.values())
    total_cit_edges = sum(len(cits) for cits in state.local_cits.values())

    frontier_len = len(shared_frontier) if shared_frontier else len(state.frontier)

    content = f"""---
type: moc
agent: {state.id}
caste: {state.caste}
quality: {state.quality}
---

# Exploration Map — {state.id}

## Overview
- **Agent:** {state.id}
- **Caste:** {state.caste}
- **Final Q:** {state.quality:.4f}
- **Turns:** {state.turn_count}
- **Papers visited:** {len(state.visited)}
- **Frontier remaining:** {frontier_len}
- **Private graph:** {total_ref_edges} ref-edges, {total_cit_edges} cit-edges
- **Research line:** {seed_query}

## Exploration Path
{path_section}

## All Visited Papers
{visited_section}

## Full Narrative

{state.narrative or "(no narrative produced)"}
"""

    (folder / "_MOC.md").write_text(content, encoding="utf-8")
