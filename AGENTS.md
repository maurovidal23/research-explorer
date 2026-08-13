# AGENTS.md — Development guide for AI agents working on this repo

## Project

Research Explorer: ACO multi-agent heuristic exploration of scientific citation graphs
via MCP. Python 3.10+, uv-managed.

## Commands

```bash
# Install dependencies (including dev)
uv sync --extra dev

# Run tests
uv run pytest tests/ -q

# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/

# Run the CLI
uv run research-explorer --help
uv run research-explorer config
uv run research-explorer explore <DOI> "research line query"

# Run an MCP server standalone
uv run python -m research_explorer.mcp.semantic_scholar_server
```

## Architecture

See `docs/model.md` for the formal model. Key modules:

- `providers/` — Resilient HTTP adapters for academic APIs (Semantic Scholar, OpenAlex,
  PubMed, arXiv). Each has its own rate limiter, circuit breaker, semaphore, and cache.
- `mcp/` — Thin MCP server wrappers over the providers (for external use like Claude
  Desktop). The ACO engine imports providers directly, not via MCP.
- `graph/` — SQLite store for papers, edges, embeddings, and pheromone. Shared by all
  agents (collective memory).
- `agents/` — LLM client (NaN/OpenAI-compatible), agent state, explorer agent, prompts.
- `evaluation/` — Quality function Q = S + P + J + R (self-assess, peer vote, virgin
  judge, structural metrics).
- `aco/` — Colony, scheduler (K concurrent slots), convergence checker.
- `orchestrator/` — Main loop tying it all together.
- `config.py` — TOML config loader into typed dataclasses.

## Conventions

- No comments in code unless explicitly requested.
- Type hints required on all public functions.
- Pydantic for data models, dataclasses for config.
- structlog for logging (never print to stdout in MCP servers — use stderr).
- async/await for all I/O (providers, LLM, MCP tools).
- Tests use pytest + pytest-asyncio (asyncio_mode = "auto").
