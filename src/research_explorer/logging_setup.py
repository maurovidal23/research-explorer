"""Structured logging setup with ACO-aware terminal formatting.

Uses structlog with a custom renderer that recognizes ACO exploration events
and formats them into a visual narrative with headers, indentation, timing,
and quality breakdowns — making the exploration process easy to follow.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any, ClassVar

import structlog

from research_explorer.redaction import redact_obj

WIDTH = 72


class _Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GRAY = "\033[90m"
    MAGENTA = "\033[35m"


_COLORS = sys.stderr.isatty()


def _c(color: str, text: str) -> str:
    if not _COLORS:
        return text
    return f"{color}{text}{_Color.RESET}"


def _short_agent(agent_id: str) -> str:
    parts = agent_id.split("-")
    if len(parts) >= 2 and parts[0] == "agent":
        try:
            return f"A{int(parts[1]):02d}"
        except ValueError:
            pass
    return agent_id


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}m {s:.0f}s"


def _box(lines: list[str]) -> str:
    inner = WIDTH - 4
    top = "  +-" + "-" * inner + "-+"
    bottom = "  +-" + "-" * inner + "-+"
    content = []
    for line in lines:
        if len(line) > inner:
            line = line[: inner - 3] + "..."
        content.append(f"  | {line.ljust(inner)} |")
    return "\n".join([top, *content, bottom])


def _separator(char: str = "=") -> str:
    return char * WIDTH


class ACORenderer:
    """Custom structlog renderer that formats ACO exploration events for terminal."""

    def __call__(self, logger, method_name, event_dict):
        event = event_dict.get("event", "")
        level = event_dict.get("level", "info")

        handler = self._dispatch.get(event)
        if handler:
            return handler(self, event_dict)
        if event.startswith("Retrying "):
            return self._fmt_retry(event, event_dict)
        return self._fmt_default(event, level, event_dict)

    # ── Major lifecycle events ──────────────────────────────────────

    def _fmt_orchestrator_start(self, d):
        seed = d.get("seed", "?")
        query = d.get("query", "?")
        colony = d.get("colony_size", "?")
        K = d.get("K", "?")
        budget = d.get("max_fetches", "?")
        lines = [
            "RESEARCH EXPLORER -- ACO Multi-Agent Exploration",
            "",
            f"Seed:   {seed}",
            f'Query:  "{query}"',
            f"Colony: {colony} agents | K={K} concurrent | Budget: {budget} fetches",
        ]
        return "\n" + _c(_Color.CYAN + _Color.BOLD, _box(lines))

    def _fmt_orchestrator_complete(self, d):
        winner = _short_agent(d.get("winner", "?"))
        best_q = d.get("best_Q", 0)
        peak_q = d.get("peak_Q", best_q)
        snap_oleada = d.get("snapshot_oleada", "?")
        oleadas = d.get("oleadas", "?")
        fetches = d.get("total_fetches", "?")
        elapsed = d.get("elapsed", 0)
        lines = [
            "EXPLORATION COMPLETE",
            "",
            f"Winner:  {winner}  (peak at oleada {snap_oleada})",
            f"Peak Q:  {peak_q:.2f}",
            f"Oleadas: {oleadas}  |  Fetches: {fetches}  |  Time: {_fmt_time(elapsed)}",
        ]
        return _c(_Color.GREEN + _Color.BOLD, _box(lines))

    def _fmt_colony_initialized(self, d):
        size = d.get("size", "?")
        budget = d.get("budget_per_agent", "?")
        castes = d.get("castes", {})
        caste_str = " | ".join(f"{v} {k}" for k, v in castes.items()) if castes else ""
        line1 = f"  Colony ready -- {size} agents ({budget} fetches each)"
        result = _c(_Color.GREEN, line1)
        if caste_str:
            result += "\n" + _c(_Color.GRAY, f"    Castes: {caste_str}")
        return result

    # ── Oleada events ────────────────────────────────────────────────

    def _fmt_oleada_start(self, d):
        n = d.get("oleada", "?")
        active = d.get("active", [])
        short = ", ".join(_short_agent(a) for a in active)
        sep = _c(_Color.CYAN, _separator("="))
        header = _c(_Color.CYAN + _Color.BOLD, f"  OLEADA {n}  |  active: {short}")
        return f"\n{sep}\n{header}\n{sep}"

    def _fmt_oleada_complete(self, d):
        n = d.get("oleada", "?")
        best_q = d.get("best_Q", 0)
        total = d.get("total_fetches", 0)
        max_f = d.get("max_fetches", "?")
        elapsed = d.get("elapsed", 0)
        ranking = d.get("ranking", [])
        sep = _c(_Color.GRAY, "  " + "-" * (WIDTH - 2))
        line = (
            f"  Oleada {n} done  |  best_Q={best_q:.2f}  |  "
            f"fetches: {total}/{max_f}  |  {_fmt_time(elapsed)}"
        )
        result = f"{sep}\n{_c(_Color.BOLD, line)}"
        if ranking:
            rank_str = "  ".join(f"{_short_agent(a)}={q:.2f}" for a, q in ranking)
            result += "\n" + _c(_Color.GRAY, f"  Top: {rank_str}")
        result += f"\n{sep}"
        return result

    # ── Agent turn events ────────────────────────────────────────────

    def _fmt_agent_turn_start(self, d):
        agent = _short_agent(d.get("agent", "?"))
        caste = d.get("caste", "?")
        turn = d.get("turn", "?")
        return _c(_Color.BOLD, f"  > {agent} ({caste}) -- turn {turn}")

    def _fmt_agent_step(self, d):
        mode = d.get("mode", "?")
        title = d.get("title", "?")
        year = d.get("year", "?")
        mode_tag = _c(_Color.MAGENTA, f"[{mode:5s}]")
        return f"      {mode_tag} \"{title}\" ({year})"

    def _fmt_agent_turn_complete(self, d):
        q = d.get("Q", 0)
        s = d.get("S", 0)
        p = d.get("P", 0)
        j = d.get("J", 0)
        r = d.get("R", 0)
        delta = d.get("delta_q", 0)
        budget = d.get("budget", "?")
        frontier = d.get("frontier", "?")
        fetches = d.get("fetches", "?")
        sign = "+" if delta >= 0 else ""
        q_line = (
            f"      Q={q:.2f}  S={s:.2f}  P={p:.2f}  "
            f"J={j:.2f}  R={r:.2f}  dQ={sign}{delta:.2f}"
        )
        b_line = f"      budget: {budget}  frontier: {frontier}  fetches: {fetches}"
        return _c(_Color.GREEN, q_line) + "\n" + _c(_Color.GRAY, b_line)

    # ── Warnings and info ────────────────────────────────────────────

    def _fmt_new_best(self, d):
        agent = _short_agent(d.get("agent", "?"))
        q = d.get("Q", 0)
        oleada = d.get("oleada", "?")
        return _c(
            _Color.GREEN + _Color.BOLD,
            f"  *** NEW BEST  {agent}  Q={q:.2f}  (oleada {oleada})  -- narrative snapshotted",
        )

    def _fmt_all_agents_exhausted(self, d):
        return _c(_Color.YELLOW, "  ! All agents exhausted -- stopping")

    def _fmt_no_winner(self, d):
        return _c(_Color.RED, "  ! No winner -- exploration produced no narrative")

    def _fmt_no_active_agents(self, d):
        return _c(_Color.YELLOW, "  ! No active agents -- stopping")

    def _fmt_warning_generic(self, d):
        event = d.get("event", "warning").replace("_", " ")
        d.pop("event", None)
        d.pop("level", None)
        d.pop("timestamp", None)
        parts = [f"{k}={v}" for k, v in d.items()]
        extra = " ".join(parts)
        line = f"  ! {event}"
        if extra:
            line += f" -- {extra}"
        return _c(_Color.YELLOW, line)

    def _fmt_retry(self, event, d):
        if len(event) > 90:
            event = event[:87] + "..."
        return _c(_Color.YELLOW, f"  [retry] {event}")

    def _fmt_research_event(self, d):
        """Render kernel research events with explicit field separators."""
        d.pop("event", None)
        d.pop("level", None)
        d.pop("timestamp", None)
        event_type = str(d.pop("event_type", "event"))
        outcome = d.pop("outcome", None)
        preferred = (
            "run_id",
            "seq",
            "turn",
            "actor",
            "tokens",
            "fetches",
            "seconds",
        )
        keys = [k for k in preferred if k in d and d[k] is not None]
        keys.extend(
            k for k, v in d.items() if k not in preferred and v is not None
        )
        label = f"{event_type} ({outcome})" if outcome else event_type
        color = self._research_event_colors.get(event_type, _Color.GRAY)
        line = _c(color + _Color.BOLD, f"  [kernel] {label}")
        if keys:
            parts = [f"{key}={d[key]}" for key in keys]
            line += _c(_Color.GRAY, "  " + " | ".join(parts))
        return line

    def _fmt_default(self, event, level, d):
        d.pop("event", None)
        d.pop("level", None)
        d.pop("timestamp", None)
        exc = d.pop("exception", None)
        parts = [f"{k}={v}" for k, v in d.items()]
        extra = " ".join(parts)
        prefix = f"[{level}]" if level != "info" else ""
        line = f"  {prefix} {event}".strip()
        if extra:
            line += f"  {extra}"
        if exc:
            line += f"\n{exc}"
        return line

    _research_event_colors: ClassVar[dict[str, str]] = {
        "kernel_start": _Color.CYAN,
        "run_complete": _Color.GREEN,
        "final_answer": _Color.GREEN,
        "evidence_acquired": _Color.GREEN,
        "search_complete": _Color.GREEN,
        "evaluation_complete": _Color.GREEN,
        "verdict_transition": _Color.CYAN,
        "provider_failure": _Color.YELLOW,
        "agent_turn_failed": _Color.YELLOW,
    }

    _dispatch: ClassVar[dict[str, Callable]] = {
        "orchestrator_start": _fmt_orchestrator_start,
        "orchestrator_complete": _fmt_orchestrator_complete,
        "colony_initialized": _fmt_colony_initialized,
        "oleada_start": _fmt_oleada_start,
        "oleada_complete": _fmt_oleada_complete,
        "agent_turn_start": _fmt_agent_turn_start,
        "agent_step": _fmt_agent_step,
        "agent_turn_complete": _fmt_agent_turn_complete,
        "new_best": _fmt_new_best,
        "all_agents_exhausted": _fmt_all_agents_exhausted,
        "no_winner": _fmt_no_winner,
        "no_active_agents": _fmt_no_active_agents,
        "seed_embedding_failed": _fmt_warning_generic,
        "fetch_failed": _fmt_warning_generic,
        "integrate_failed": _fmt_warning_generic,
        "circuit_open": _fmt_warning_generic,
        "request_failed": _fmt_warning_generic,
        "self_assess_failed": _fmt_warning_generic,
        "virgin_judge_failed": _fmt_warning_generic,
        "research_event": _fmt_research_event,
    }


def _redact_processor(logger, method_name, event_dict):
    """Redact secrets from every value, recursing into nested structures."""
    return redact_obj(event_dict)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog for human-readable or JSON output.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
        json_logs: If True, emit JSON lines; else ACO-formatted console output.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # Suppress noisy HTTP logs from httpx/openai (they flood the terminal)
    for noisy in ("httpx", "openai", "openai._base_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_processor,
    ]

    processors: list[Any]
    if json_logs:
        processors = [*shared_processors, structlog.processors.JSONRenderer()]
    else:
        processors = [*shared_processors, ACORenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    """Get a structured logger."""
    return structlog.get_logger(name) if name else structlog.get_logger()
