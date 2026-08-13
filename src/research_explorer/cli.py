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

    async def _run() -> str:
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
