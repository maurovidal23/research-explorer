"""Command-line interface for Research Explorer."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from research_explorer.config import load_config
from research_explorer.logging_setup import configure_logging, get_logger

app = typer.Typer(
    name="research-explorer",
    help="ACO multi-agent heuristic exploration of scientific citation graphs.",
    no_args_is_help=True,
)

mcp_app = typer.Typer(help="Manage MCP servers.", no_args_is_help=True)
app.add_typer(mcp_app, name="mcp")

replay_app = typer.Typer(help="Inspect and replay recorded evaluation runs.", no_args_is_help=True)
app.add_typer(replay_app, name="replay")


@app.command()
def explore(
    seed_paper_id: str = typer.Argument(help="Seed paper ID (DOI, S2 ID, PMID, etc.)"),
    seed_query: str = typer.Argument(help="Research line description to explore"),
    config_path: str = typer.Option(
        "config/default.toml", "--config", "-c", help="Path to config TOML file"
    ),
    output: str = typer.Option(None, "--output", "-o", help="Write narrative to file"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON log lines"),
) -> None:
    """Explore a research line starting from a seed paper."""
    cfg = load_config(config_path)
    configure_logging(cfg.log_level, json_logs=json_logs)
    log = get_logger("cli")

    from research_explorer.orchestrator.runner import Orchestrator

    async def _run() -> tuple[str, str | None]:
        orch = Orchestrator(cfg)
        try:
            await orch.run(seed_paper_id, seed_query)
            report = orch.generate_report(seed_paper_id, seed_query)
            obsidian_dir = orch.generate_obsidian(seed_query)
        finally:
            await orch.aclose()
        return report, obsidian_dir

    log.info("starting_exploration", seed=seed_paper_id, query=seed_query)
    report, obsidian_dir = asyncio.run(_run())

    if output:
        Path(output).write_text(report, encoding="utf-8")
        typer.echo(f"Report written to {output}")
    else:
        typer.echo("\n" + "=" * 80)
        typer.echo("EXPLORATION REPORT")
        typer.echo("=" * 80)
        typer.echo(report)

    if obsidian_dir:
        typer.echo(f"Obsidian graph written to {obsidian_dir}/")


@mcp_app.command("list")
def mcp_list() -> None:
    """List available MCP servers."""
    servers = [
        ("semantic-scholar", "Semantic Scholar API", "semantic_scholar_server"),
        ("openalex", "OpenAlex API", "openalex_server"),
        ("pubmed", "PubMed E-utilities", "pubmed_server"),
        ("arxiv", "arXiv preprint repository", "arxiv_server"),
    ]
    typer.echo("Available MCP servers:\n")
    for name, desc, module in servers:
        typer.echo(f"  {name:20s} {desc}")
        typer.echo(f"  {'':20s} Run: python -m research_explorer.mcp.{module}")
        typer.echo()


@mcp_app.command("run")
def mcp_run(
    server: str = typer.Argument(help="Server name: semantic-scholar | openalex | pubmed | arxiv"),
) -> None:
    """Run an MCP server (stdio transport)."""
    import importlib

    module_map = {
        "semantic-scholar": "research_explorer.mcp.semantic_scholar_server",
        "openalex": "research_explorer.mcp.openalex_server",
        "pubmed": "research_explorer.mcp.pubmed_server",
        "arxiv": "research_explorer.mcp.arxiv_server",
    }
    module_name = module_map.get(server)
    if module_name is None:
        typer.echo(f"Unknown server: {server}", err=True)
        raise typer.Exit(1)

    mod = importlib.import_module(module_name)
    mod.mcp.run(transport="stdio")


@app.command()
def config(
    config_path: str = typer.Option(
        "config/default.toml", "--config", "-c", help="Path to config TOML file"
    ),
) -> None:
    """Show the loaded configuration."""
    cfg = load_config(config_path)
    typer.echo(f"LLM: {cfg.llm.base_url} (model: {cfg.llm.explorer_model})")
    typer.echo(f"ACO: colony={cfg.aco.colony_size}, K={cfg.aco.max_concurrent}, k={cfg.aco.k_per_turn}")
    typer.echo(f"Providers: {cfg.providers.active} (default: {cfg.providers.default})")
    typer.echo(f"Budget: {cfg.budget.type}={cfg.budget.max_fetches}")
    typer.echo(f"Quality weights: S={cfg.quality.w_self} P={cfg.quality.w_peers} J={cfg.quality.w_virgin} R={cfg.quality.w_structural}")


@replay_app.command("list")
def replay_list(
    db: str = typer.Option("data/replay.db", "--db", help="Path to the replay database"),
) -> None:
    """List recorded runs."""
    from research_explorer.replay.trace import RunTraceStore

    store = RunTraceStore(db)
    runs = store.list_runs()
    store.close()
    if not runs:
        typer.echo("No runs recorded.")
        return
    for r in runs:
        q = f"  Q={r['best_quality']:.4f}" if r["best_quality"] is not None else ""
        typer.echo(
            f"{r['run_id']}  [{r['status']}]  {r['seed_paper_id']}{q}  "
            f"{r['event_count']} events  {r['evaluation_count']} evals  {r['started_at']}"
        )


@replay_app.command("show")
def replay_show(
    run_id: str = typer.Argument(help="Run ID"),
    db: str = typer.Option("data/replay.db", "--db", help="Path to the replay database"),
    timeline: bool = typer.Option(False, "--timeline", help="Print the full event timeline"),
    evaluations: bool = typer.Option(False, "--evaluations", help="Print detailed evaluation records"),
) -> None:
    """Show a recorded run's summary, timeline, and/or evaluations."""
    from research_explorer.replay.trace import RunTraceStore

    store = RunTraceStore(db)
    run = store.get_run(run_id)
    if run is None:
        typer.echo(f"Run not found: {run_id}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"Run {run['run_id']} [{run['status']}]  seed={run['seed_paper_id']}  "
        f"query={run['seed_query']!r}"
    )
    typer.echo(
        f"started={run['started_at']}  completed={run['completed_at']}  "
        f"best_Q={run['best_quality']}  events={run['event_count']}"
    )
    if timeline:
        typer.echo("\n=== Timeline ===")
        for ev in store.list_events(run_id):
            payload = "  ".join(f"{k}={v}" for k, v in ev["payload"].items())
            typer.echo(f"[{ev['seq']}] {ev['type']:<24} {payload}")
    if evaluations:
        typer.echo("\n=== Evaluations ===")
        for rec in store.list_evaluations(run_id):
            typer.echo(
                f"oleada={rec.oleada} agent={rec.agent_id} Q={rec.q:.4f} "
                f"S={rec.self_assessment.score:.4f} P={rec.peers.aggregated_score:.4f} "
                f"J={rec.virgin_judge.score:.4f} R={rec.structural.r:.4f}"
            )
            typer.echo(f"  self: {rec.self_assessment.reasoning}")
            for v in rec.peers.votes:
                typer.echo(f"  peer {v.voter_id}: {v.score:.4f} -- {v.reasoning}")
            typer.echo(f"  virgin coverage: {rec.virgin_judge.coverage}")
            typer.echo(f"  virgin gaps: {rec.virgin_judge.gaps}")
    store.close()


@replay_app.command("export")
def replay_export(
    run_id: str = typer.Argument(help="Run ID"),
    out_dir: str = typer.Option("replay_out", "--out", help="Destination directory"),
    db: str = typer.Option("data/replay.db", "--db", help="Path to the replay database"),
) -> None:
    """Export the run's narrative snapshot(s) to disk (safe filenames only)."""
    from research_explorer.replay.trace import RunTraceStore

    store = RunTraceStore(db)
    artifacts = store.list_artifacts(run_id)
    narratives = [a for a in artifacts if a["kind"] == "narrative"]
    if not narratives:
        typer.echo(f"No narrative snapshots for run {run_id}", err=True)
        raise typer.Exit(1)
    for a in narratives:
        target = store.export_artifact(a["artifact_id"], out_dir)
        typer.echo(f"Exported {a['name']} -> {target}")
    store.close()


@replay_app.command("serve")
def replay_serve(
    db: str = typer.Option("data/replay.db", "--db", help="Path to the replay database"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    run_id: str = typer.Option(None, "--run-id", help="Validate and bind to a specific run"),
) -> None:
    """Serve the replay web UI (FastAPI)."""
    import uvicorn

    from research_explorer.replay.trace import RunTraceStore

    if run_id is not None:
        store = RunTraceStore(db)
        run = store.get_run(run_id)
        store.close()
        if run is None:
            typer.echo(f"Run not found: {run_id}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Replay UI on http://{host}:{port}  (db: {db}, run: {run_id})")
    else:
        typer.echo(f"Replay UI on http://{host}:{port}  (db: {db})")

    from research_explorer.replay.server import build_app

    uvicorn.run(build_app(db, default_run_id=run_id), host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
