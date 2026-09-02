"""Configuration loading from TOML files into typed dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _load_env(config_path: str | Path) -> None:
    """Load .env files: one next to the config, then the project root, then CWD."""
    config_path = Path(config_path)
    candidates = [
        config_path.parent / ".env",
        config_path.parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if p.is_file():
            load_dotenv(p, override=False)


@dataclass
class LLMConfig:
    base_url: str = "https://api.nan.builders/v1"
    api_key_env: str = "NAN_API_KEY"
    max_concurrent: int = 5
    rpm: int = 60
    explorer_model: str = "qwen3.6"
    judge_model: str = "deepseek-v4-flash"
    embedding_model: str = "qwen3-embedding"
    rerank_model: str = "rerank"
    temperature: float = 0.6
    max_tokens: int = 2000
    fulltext_max_chars: int = 100000


@dataclass
class ACOConfig:
    colony_size: int = 15
    max_concurrent: int = 3
    k_per_turn: int = 4
    alpha: float = 1.0
    beta: float = 3.0
    rho: float = 0.1
    epsilon: float = 0.1
    tau_min: float = 0.1
    tau_max: float = 10.0
    lambda_elite: float = 0.5
    tau_init: float = 1.0


@dataclass
class QualityConfig:
    w_self: float = 0.25
    w_peers: float = 0.25
    w_virgin: float = 0.25
    w_structural: float = 0.25
    peer_aggregation: str = "median"  # "median" | "reputation"


@dataclass
class DirectionConfig:
    ref_weight: float = 0.7
    cites_weight: float = 0.3
    phase_directional: bool = False  # si True, varía pesos por fase global


@dataclass
class HeuristicaConfig:
    w_sim: float = 0.6
    w_citas: float = 0.2
    w_recencia: float = 0.2
    use_rerank: bool = True
    eta_llm: bool = False  # si True, usa LLM virgen para η (costoso)


@dataclass
class ProviderEntry:
    rate: int = 1
    period: int = 1
    concurrency: int = 5
    timeout: float = 30.0
    api_key_env: str | None = None


@dataclass
class ProvidersConfig:
    default: str = "semantic_scholar"
    active: list[str] = field(default_factory=lambda: ["semantic_scholar", "openalex", "pubmed", "arxiv"])
    entries: dict[str, ProviderEntry] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    type: str = "fetches"  # "fetches" | "time" | "convergence"
    max_fetches: int = 500
    max_time_seconds: int = 3600


@dataclass
class ConvergenceConfig:
    plateau_T: int = 10
    epsilon: float = 0.01
    theta: float = 5.0


@dataclass
class SchedulerConfig:
    gamma: float = 0.5
    delta: float = 0.3


@dataclass
class StorageConfig:
    db_path: str = "data/explorer.db"
    cache_dir: str = "data/.cache"
    trace_db_path: str = "data/replay.db"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    heuristica: HeuristicaConfig = field(default_factory=HeuristicaConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    log_level: str = "INFO"
    seed_query: str = ""


def _resolve_env(value: str | None) -> str | None:
    """Resolve a value that may reference an env var by name."""
    if value is None:
        return None
    return os.environ.get(value, value) if value.isupper() and "_" in value else value


def _load_provider_entries(raw: dict[str, Any]) -> dict[str, ProviderEntry]:
    entries: dict[str, ProviderEntry] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        entries[name] = ProviderEntry(
            rate=cfg.get("rate", 1),
            period=cfg.get("period", 1),
            concurrency=cfg.get("concurrency", 5),
            timeout=cfg.get("timeout", 30.0),
            api_key_env=cfg.get("api_key_env"),
        )
    return entries


def load_config(path: str | Path) -> Config:
    """Load a TOML config file into a Config dataclass.

    Also loads .env files (next to config, project root, or CWD) so API keys
    can be provided without setting OS environment variables.

    Args:
        path: Path to the TOML config file.

    Returns:
        Parsed Config instance.
    """
    _load_env(path)
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)

    cfg = Config()
    cfg.log_level = data.get("log_level", "INFO")
    cfg.seed_query = data.get("seed_query", "")

    if "llm" in data:
        llm = data["llm"]
        cfg.llm = LLMConfig(
            base_url=llm.get("base_url", cfg.llm.base_url),
            api_key_env=llm.get("api_key_env", cfg.llm.api_key_env),
            max_concurrent=llm.get("max_concurrent", cfg.llm.max_concurrent),
            rpm=llm.get("rpm", cfg.llm.rpm),
            explorer_model=llm.get("explorer_model", cfg.llm.explorer_model),
            judge_model=llm.get("judge_model", cfg.llm.judge_model),
            embedding_model=llm.get("embedding_model", cfg.llm.embedding_model),
            rerank_model=llm.get("rerank_model", cfg.llm.rerank_model),
            temperature=llm.get("temperature", cfg.llm.temperature),
            max_tokens=llm.get("max_tokens", cfg.llm.max_tokens),
            fulltext_max_chars=llm.get("fulltext_max_chars", cfg.llm.fulltext_max_chars),
        )

    if "aco" in data:
        aco = data["aco"]
        cfg.aco = ACOConfig(
            colony_size=aco.get("colony_size", cfg.aco.colony_size),
            max_concurrent=aco.get("max_concurrent", cfg.aco.max_concurrent),
            k_per_turn=aco.get("k_per_turn", cfg.aco.k_per_turn),
            alpha=aco.get("alpha", cfg.aco.alpha),
            beta=aco.get("beta", cfg.aco.beta),
            rho=aco.get("rho", cfg.aco.rho),
            epsilon=aco.get("epsilon", cfg.aco.epsilon),
            tau_min=aco.get("tau_min", cfg.aco.tau_min),
            tau_max=aco.get("tau_max", cfg.aco.tau_max),
            lambda_elite=aco.get("lambda_elite", cfg.aco.lambda_elite),
            tau_init=aco.get("tau_init", cfg.aco.tau_init),
        )

    if "quality" in data:
        q = data["quality"]
        cfg.quality = QualityConfig(
            w_self=q.get("w_self", cfg.quality.w_self),
            w_peers=q.get("w_peers", cfg.quality.w_peers),
            w_virgin=q.get("w_virgin", cfg.quality.w_virgin),
            w_structural=q.get("w_structural", cfg.quality.w_structural),
            peer_aggregation=q.get("peer_aggregation", cfg.quality.peer_aggregation),
        )

    if "direction" in data:
        d = data["direction"]
        cfg.direction = DirectionConfig(
            ref_weight=d.get("ref_weight", cfg.direction.ref_weight),
            cites_weight=d.get("cites_weight", cfg.direction.cites_weight),
            phase_directional=d.get("phase_directional", cfg.direction.phase_directional),
        )

    if "heuristica" in data:
        h = data["heuristica"]
        cfg.heuristica = HeuristicaConfig(
            w_sim=h.get("w_sim", cfg.heuristica.w_sim),
            w_citas=h.get("w_citas", cfg.heuristica.w_citas),
            w_recencia=h.get("w_recencia", cfg.heuristica.w_recencia),
            use_rerank=h.get("use_rerank", cfg.heuristica.use_rerank),
            eta_llm=h.get("eta_llm", cfg.heuristica.eta_llm),
        )

    if "providers" in data:
        p = data["providers"]
        cfg.providers = ProvidersConfig(
            default=p.get("default", cfg.providers.default),
            active=p.get("active", cfg.providers.active),
            entries=_load_provider_entries(
                {k: v for k, v in p.items() if k not in ("default", "active")}
            ),
        )

    if "budget" in data:
        b = data["budget"]
        cfg.budget = BudgetConfig(
            type=b.get("type", cfg.budget.type),
            max_fetches=b.get("max_fetches", cfg.budget.max_fetches),
            max_time_seconds=b.get("max_time_seconds", cfg.budget.max_time_seconds),
        )

    if "convergence" in data:
        c = data["convergence"]
        cfg.convergence = ConvergenceConfig(
            plateau_T=c.get("plateau_T", cfg.convergence.plateau_T),
            epsilon=c.get("epsilon", cfg.convergence.epsilon),
            theta=c.get("theta", cfg.convergence.theta),
        )

    if "scheduler" in data:
        s = data["scheduler"]
        cfg.scheduler = SchedulerConfig(
            gamma=s.get("gamma", cfg.scheduler.gamma),
            delta=s.get("delta", cfg.scheduler.delta),
        )

    if "storage" in data:
        st = data["storage"]
        cfg.storage = StorageConfig(
            db_path=st.get("db_path", cfg.storage.db_path),
            cache_dir=st.get("cache_dir", cfg.storage.cache_dir),
            trace_db_path=st.get("trace_db_path", cfg.storage.trace_db_path),
        )

    return cfg


def get_api_key(api_key_env: str | None) -> str | None:
    """Resolve an API key from the environment variable named by api_key_env."""
    if api_key_env is None:
        return None
    return os.environ.get(api_key_env)
