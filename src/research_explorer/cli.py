"""Command-line interface for Research Explorer."""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path

import typer

from research_explorer.config import get_api_key, load_config
from research_explorer.logging_setup import configure_logging, get_logger
from research_explorer.redaction import redact_secrets

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
    output: str = typer.Option(None, "--output", "-o", help="Write result to file"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON log lines"),
    pipeline: str = typer.Option(
        None, "--pipeline", help="Pipeline mode: aco | research-kernel"
    ),
    tui: bool = typer.Option(
        False,
        "--tui/--no-tui",
        help="Launch the live terminal interface (ACO pipeline only)",
    ),
) -> None:
    """Explore a research line starting from a seed paper."""
    cfg = load_config(config_path)
    configure_logging(cfg.log_level, json_logs=json_logs)
    log = get_logger("cli")

    mode = (pipeline or cfg.pipeline or "aco").replace("_", "-").lower()
    if tui and mode != "aco":
        raise typer.BadParameter(
            f"--tui is not supported for the '{mode}' pipeline yet; "
            "run without --tui or use --pipeline aco."
        )
    if mode == "research-kernel":
        _run_research_kernel(cfg, seed_paper_id, seed_query, output)
        return

    from research_explorer.orchestrator.runner import Orchestrator

    if tui:
        _run_aco_tui(cfg, seed_paper_id, seed_query, output)
        return

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
    _emit_report(report, output, obsidian_dir)


def _emit_report(report: str, output: str | None, obsidian_dir: str | None) -> None:
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


def _write_report_atomic(report: str, output: str) -> None:
    """Write ``report`` to ``output`` via an atomic same-directory replace.

    A missing parent directory is an error (no directories are created
    implicitly). Interruption can never leave a partially written report
    because the destination only ever sees a fully written temp file.
    """
    target = Path(output)
    parent = target.parent if str(target.parent) else Path(".")
    fd, tmp_path = tempfile.mkstemp(
        dir=str(parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(report)
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _run_aco_tui(cfg, seed_paper_id: str, seed_query: str, output: str | None) -> None:
    """Run the ACO exploration under the Textual terminal interface."""
    from research_explorer.orchestrator.runner import Orchestrator
    from research_explorer.tui import TUIController, build_app

    controller = TUIController()
    orch = Orchestrator(cfg, event_sink=controller.event_sink)
    report_written = False
    report_error: str | None = None

    async def runner() -> tuple[str, str | None]:
        try:
            await orch.run(seed_paper_id, seed_query)
            report = orch.generate_report(seed_paper_id, seed_query)
            obsidian_dir = orch.generate_obsidian(seed_query)
            return report, obsidian_dir
        except asyncio.CancelledError:
            orch.mark_cancelled()
            raise
        finally:
            await orch.aclose()

    def persist_report(result: tuple[str, str | None]) -> None:
        """Persist the report once the backend finishes, before review waits."""
        nonlocal report_written, report_error
        if not output or report_written:
            return
        report, _obsidian = result
        try:
            _write_report_atomic(report, output)
        except Exception as exc:
            report_error = redact_secrets(str(exc))
            return
        report_written = True

    app = build_app(
        controller.projection,
        queue=controller.queue,
        runner=runner,
        on_result=persist_report,
    )
    app.run()
    controller.drain()

    if app.cancelled or app.state.status == "cancelled":
        typer.echo("Run cancelled.")
        return
    if app.run_result is None:
        typer.echo("Run did not complete; see the failure summary above.", err=True)
        raise typer.Exit(1)
    report, obsidian_dir = app.run_result
    # Fallback for apps that do not invoke the completion hook.
    if output and not report_written and report_error is None:
        persist_report(app.run_result)
    if report_error is not None:
        typer.echo(f"Report write failed: {report_error}", err=True)
        raise typer.Exit(1)
    if output:
        typer.echo(f"Report written to {output}")
        if obsidian_dir:
            typer.echo(f"Obsidian graph written to {obsidian_dir}/")
        return
    _emit_report(report, output, obsidian_dir)


def _run_research_kernel(cfg, seed_paper_id: str, seed_query: str, output: str | None) -> None:
    """Run the single-agent research-kernel vertical slice."""
    import uuid

    from research_explorer.agents.llm_client import LLMClient
    from research_explorer.graph.store import GraphStore
    from research_explorer.providers.factory import build_all_providers
    from research_explorer.providers.routing import route_seed_provider
    from research_explorer.research import (
        BudgetState,
        CompositeEvaluator,
        DeterministicIntegrity,
        GraphEvidenceGateway,
        GreedyPolicy,
        KernelOptions,
        LLMReferenceMapper,
        LLMResearchAgent,
        LLMRubricEvaluator,
        ResearchKernel,
        ResearchObjective,
        ResearchStore,
    )

    rk = cfg.research_kernel
    providers = build_all_providers(cfg)
    if not providers:
        raise typer.BadParameter("No providers enabled; set providers.active in the config.")
    seed_provider, seed_ref = route_seed_provider(
        seed_paper_id, providers, cfg.providers.seed_routing
    )

    question = seed_query.strip()
    if not question:
        if rk.require_question:
            raise typer.BadParameter(
                "A non-empty research question is required; set "
                "research_kernel.require_question = false to allow a default."
            )
        question = f"Explore the research context of {seed_ref.value}."

    graph = GraphStore(cfg.storage.db_path)
    store = ResearchStore(cfg.storage.research_db_path)
    llm = LLMClient(
        base_url=cfg.llm.base_url,
        api_key=get_api_key(cfg.llm.api_key_env),
        max_concurrent=cfg.llm.max_concurrent,
        rpm=cfg.llm.rpm,
    )
    doi_provider = next(
        (name for name in ("openalex", "semantic_scholar") if name in providers),
        None,
    )
    reference_mapper = LLMReferenceMapper(
        llm,
        model=cfg.llm.explorer_model,
        max_tokens=rk.output_reserve,
        doi_provider=doi_provider,
    )
    gateway = GraphEvidenceGateway(
        graph,
        providers,
        seed_provider.name,
        reference_mapper=reference_mapper,
        fulltext_max_chars=cfg.llm.fulltext_max_chars,
        question=question,
    )
    policy = GreedyPolicy(seed=rk.seed)
    agent = LLMResearchAgent(llm, model=cfg.llm.explorer_model, max_tokens=rk.output_reserve)
    rubric = (
        LLMRubricEvaluator(llm, model=cfg.llm.judge_model)
        if rk.evaluator_enabled
        else None
    )
    evaluator = CompositeEvaluator(
        integrity=DeterministicIntegrity(paper_exists=gateway.paper_exists),
        rubric=rubric,
        weights=rk.weights,
        rubric_version=rk.rubric_version,
    )
    options = KernelOptions(
        max_transient_attempts=rk.transient_retry_attempts,
        plateau_turns=rk.plateau_turns,
        convergence_epsilon=rk.convergence_epsilon,
        snapshot_interval=rk.snapshot_interval,
        context_input_target=rk.context_input_target,
        output_reserve=rk.output_reserve,
        eval_interval=rk.eval_interval,
        evaluator_enabled=rk.evaluator_enabled,
    )
    kernel = ResearchKernel(
        store=store, gateway=gateway, policy=policy, agent=agent, evaluator=evaluator,
        options=options,
    )
    objective = ResearchObjective(
        run_id=uuid.uuid4().hex[:12],
        seed_paper_id=seed_ref.fetch_value,
        question=question,
        budget=BudgetState(
            max_fetches=rk.max_fetches,
            max_tokens=rk.max_tokens,
            max_time_seconds=rk.max_time_seconds,
            max_turns=rk.max_turns,
        ),
        random_seed=rk.seed,
    )

    async def _run() -> str:
        try:
            answer = await kernel.run(objective)
            return answer.render_markdown()
        finally:
            await llm.aclose()
            for provider in providers.values():
                await provider.aclose()
            graph.close()
            store.close()

    report = asyncio.run(_run())
    if output:
        Path(output).write_text(report, encoding="utf-8")
        typer.echo(f"Research answer written to {output}")
    else:
        typer.echo("\n" + "=" * 80)
        typer.echo("RESEARCH ANSWER")
        typer.echo("=" * 80)
        typer.echo(report)


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
    typer.echo(f"Pipeline: {cfg.pipeline}")
    typer.echo(f"ACO: colony={cfg.aco.colony_size}, K={cfg.aco.max_concurrent}, k={cfg.aco.k_per_turn}")
    typer.echo(f"Providers: {cfg.providers.active} (default: {cfg.providers.default})")
    typer.echo(f"Budget: {cfg.budget.type}={cfg.budget.max_fetches}")
    rk = cfg.research_kernel
    typer.echo(
        f"ResearchKernel: policy={rk.policy}, fetches={rk.max_fetches}, "
        f"turns={rk.max_turns}, evaluator={rk.evaluator_enabled} ({rk.rubric_version})"
    )
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
