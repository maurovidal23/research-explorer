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


@dataclass
class HeuristicaConfig:
    w_sim: float = 0.6
    w_citas: float = 0.2
    w_recencia: float = 0.2
    w_confidence: float = 0.1
    w_llm: float = 0.2
    eta_llm: bool = False  # si True, incluye la prioridad LLM en η (costoso)


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
    seed_routing: list[str] = field(default_factory=list)


@dataclass
class ResolutionConfig:
    enabled: bool = False
    primary: str = "openalex"
    fallbacks: list[str] = field(default_factory=lambda: ["semantic_scholar"])
    incoming_enabled: bool = True
    outgoing_enabled: bool = True
    incoming_limit: int = 50
    outgoing_limit: int = 50
    title_search_limit: int = 5
    min_title_similarity: float = 0.82
    min_author_overlap: float = 0.34
    min_confidence: float = 0.75
    year_tolerance: int = 2


@dataclass
class ReferenceMappingConfig:
    """Paper-level shared bibliography mapping (see FRG-1..FRG-6)."""

    enabled: bool = True
    batch_size: int = 10
    max_concurrent_batches: int = 2
    max_retries: int = 2
    lease_seconds: int = 300
    model: str = ""  # empty => use llm.explorer_model
    max_completion_tokens_per_batch: int = 1500
    min_parse_confidence: float = 0.3
    allow_provisional_nodes: bool = True
    max_entries: int = 0  # 0 = all entries; >0 = explicit safety ceiling
    mapper_version: str = "v1"
    prompt_version: str = "v1"


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
    research_db_path: str = "data/research.db"


def _default_eval_weights() -> dict[str, float]:
    return {
        "integrity": 0.35,
        "relevance": 0.12,
        "coverage": 0.10,
        "mechanistic_depth": 0.08,
        "methodological_understanding": 0.05,
        "evidence_traceability": 0.15,
        "counterevidence": 0.05,
        "uncertainty_calibration": 0.05,
        "novelty": 0.03,
        "redundancy": 0.02,
    }


@dataclass
class ResearchKernelConfig:
    """Additive configuration for the ``research_kernel`` pipeline mode."""

    policy: str = "greedy"
    seed: int = 0
    require_question: bool = True
    max_fetches: int = 20
    max_tokens: int = 60_000
    max_time_seconds: int = 900
    max_turns: int = 10
    context_input_target: int = 6_000
    output_reserve: int = 1_500
    evaluator_enabled: bool = True
    rubric_version: str = "v1"
    eval_interval: int = 1
    transient_retry_attempts: int = 2
    snapshot_interval: int = 1
    plateau_turns: int = 3
    convergence_epsilon: float = 0.01
    weights: dict[str, float] = field(default_factory=_default_eval_weights)


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: str = "aco"
    aco: ACOConfig = field(default_factory=ACOConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    heuristica: HeuristicaConfig = field(default_factory=HeuristicaConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    reference_mapping: ReferenceMappingConfig = field(default_factory=ReferenceMappingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    research_kernel: ResearchKernelConfig = field(default_factory=ResearchKernelConfig)
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
    cfg.pipeline = data.get("pipeline", "aco")

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
        )

    if "heuristica" in data:
        h = data["heuristica"]
        cfg.heuristica = HeuristicaConfig(
            w_sim=h.get("w_sim", cfg.heuristica.w_sim),
            w_citas=h.get("w_citas", cfg.heuristica.w_citas),
            w_recencia=h.get("w_recencia", cfg.heuristica.w_recencia),
            w_confidence=h.get("w_confidence", cfg.heuristica.w_confidence),
            w_llm=h.get("w_llm", cfg.heuristica.w_llm),
            eta_llm=h.get("eta_llm", cfg.heuristica.eta_llm),
        )

    if "providers" in data:
        p = data["providers"]
        cfg.providers = ProvidersConfig(
            default=p.get("default", cfg.providers.default),
            active=p.get("active", cfg.providers.active),
            seed_routing=p.get("seed_routing", cfg.providers.seed_routing),
            entries=_load_provider_entries(
                {k: v for k, v in p.items() if k not in ("default", "active", "seed_routing")}
            ),
        )

    if "resolution" in data:
        r = data["resolution"]
        cfg.resolution = ResolutionConfig(
            enabled=r.get("enabled", cfg.resolution.enabled),
            primary=r.get("primary", cfg.resolution.primary),
            fallbacks=r.get("fallbacks", cfg.resolution.fallbacks),
            incoming_enabled=r.get("incoming_enabled", cfg.resolution.incoming_enabled),
            outgoing_enabled=r.get("outgoing_enabled", cfg.resolution.outgoing_enabled),
            incoming_limit=r.get("incoming_limit", cfg.resolution.incoming_limit),
            outgoing_limit=r.get("outgoing_limit", cfg.resolution.outgoing_limit),
            title_search_limit=r.get("title_search_limit", cfg.resolution.title_search_limit),
            min_title_similarity=r.get("min_title_similarity", cfg.resolution.min_title_similarity),
            min_author_overlap=r.get("min_author_overlap", cfg.resolution.min_author_overlap),
            min_confidence=r.get("min_confidence", cfg.resolution.min_confidence),
            year_tolerance=r.get("year_tolerance", cfg.resolution.year_tolerance),
        )

    if "reference_mapping" in data:
        rm = data["reference_mapping"]
        cfg.reference_mapping = ReferenceMappingConfig(
            enabled=rm.get("enabled", cfg.reference_mapping.enabled),
            batch_size=rm.get("batch_size", cfg.reference_mapping.batch_size),
            max_concurrent_batches=rm.get(
                "max_concurrent_batches", cfg.reference_mapping.max_concurrent_batches
            ),
            max_retries=rm.get("max_retries", cfg.reference_mapping.max_retries),
            lease_seconds=rm.get("lease_seconds", cfg.reference_mapping.lease_seconds),
            model=rm.get("model", cfg.reference_mapping.model),
            max_completion_tokens_per_batch=rm.get(
                "max_completion_tokens_per_batch",
                cfg.reference_mapping.max_completion_tokens_per_batch,
            ),
            min_parse_confidence=rm.get(
                "min_parse_confidence", cfg.reference_mapping.min_parse_confidence
            ),
            allow_provisional_nodes=rm.get(
                "allow_provisional_nodes", cfg.reference_mapping.allow_provisional_nodes
            ),
            max_entries=rm.get("max_entries", cfg.reference_mapping.max_entries),
            mapper_version=rm.get(
                "mapper_version", cfg.reference_mapping.mapper_version
            ),
            prompt_version=rm.get(
                "prompt_version", cfg.reference_mapping.prompt_version
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
            research_db_path=st.get("research_db_path", cfg.storage.research_db_path),
        )

    if "research_kernel" in data:
        rk = data["research_kernel"]
        cfg.research_kernel = ResearchKernelConfig(
            policy=rk.get("policy", cfg.research_kernel.policy),
            seed=rk.get("seed", cfg.research_kernel.seed),
            require_question=rk.get("require_question", cfg.research_kernel.require_question),
            max_fetches=rk.get("max_fetches", cfg.research_kernel.max_fetches),
            max_tokens=rk.get("max_tokens", cfg.research_kernel.max_tokens),
            max_time_seconds=rk.get("max_time_seconds", cfg.research_kernel.max_time_seconds),
            max_turns=rk.get("max_turns", cfg.research_kernel.max_turns),
            context_input_target=rk.get(
                "context_input_target", cfg.research_kernel.context_input_target
            ),
            output_reserve=rk.get("output_reserve", cfg.research_kernel.output_reserve),
            evaluator_enabled=rk.get(
                "evaluator_enabled", cfg.research_kernel.evaluator_enabled
            ),
            rubric_version=rk.get("rubric_version", cfg.research_kernel.rubric_version),
            eval_interval=rk.get("eval_interval", cfg.research_kernel.eval_interval),
            transient_retry_attempts=rk.get(
                "transient_retry_attempts", cfg.research_kernel.transient_retry_attempts
            ),
            snapshot_interval=rk.get(
                "snapshot_interval", cfg.research_kernel.snapshot_interval
            ),
            plateau_turns=rk.get("plateau_turns", cfg.research_kernel.plateau_turns),
            convergence_epsilon=rk.get(
                "convergence_epsilon", cfg.research_kernel.convergence_epsilon
            ),
            weights=rk.get("weights", cfg.research_kernel.weights),
        )

    return cfg


def get_api_key(api_key_env: str | None) -> str | None:
    """Resolve an API key from the environment variable named by api_key_env."""
    if api_key_env is None:
        return None
    return os.environ.get(api_key_env)
