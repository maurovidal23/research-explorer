"""Configuration backward compatibility and additive research-kernel defaults."""

from __future__ import annotations

from pathlib import Path

from research_explorer.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_default_config_keeps_aco_and_gains_kernel_defaults() -> None:
    cfg = load_config(CONFIG_DIR / "default.toml")
    assert cfg.pipeline == "aco"
    assert cfg.providers.seed_routing == []
    assert cfg.storage.research_db_path == "data/research.db"
    rk = cfg.research_kernel
    assert rk.policy == "greedy"
    assert rk.require_question is True
    assert rk.max_turns == 10
    assert rk.transient_retry_attempts == 2
    assert rk.evaluator_enabled is True
    assert rk.weights["integrity"] == 0.35


def test_kernel_quick_profile_selects_pipeline_and_budgets() -> None:
    cfg = load_config(CONFIG_DIR / "profiles" / "kernel_quick.toml")
    assert cfg.pipeline == "research-kernel"
    assert cfg.providers.seed_routing == ["openalex", "semantic_scholar"]
    rk = cfg.research_kernel
    assert rk.max_fetches == 8
    assert rk.max_turns == 5
    assert rk.transient_retry_attempts == 2


def test_kernel_live_profile_loads() -> None:
    cfg = load_config(CONFIG_DIR / "profiles" / "kernel_live.toml")
    assert cfg.pipeline == "research-kernel"
    assert cfg.research_kernel.max_turns == 20
    assert cfg.providers.seed_routing == ["openalex", "semantic_scholar"]


def test_missing_kernel_section_falls_back_to_defaults(tmp_path) -> None:
    path = tmp_path / "minimal.toml"
    path.write_text('log_level = "DEBUG"\n[aco]\ncolony_size = 2\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.pipeline == "aco"
    assert cfg.research_kernel.max_fetches == 20
    assert cfg.research_kernel.weights["integrity"] == 0.35


def test_unknown_top_level_keys_are_ignored(tmp_path) -> None:
    path = tmp_path / "unknown.toml"
    path.write_text(
        'pipeline = "research-kernel"\nfuture_setting = {a = 1}\n[research_kernel]\nmax_turns = 3\n',
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.pipeline == "research-kernel"
    assert cfg.research_kernel.max_turns == 3


def test_reference_mapping_defaults_on_shipped_config() -> None:
    cfg = load_config(CONFIG_DIR / "default.toml")
    rm = cfg.reference_mapping
    assert rm.enabled is True
    assert rm.batch_size == 10
    assert rm.max_concurrent_batches == 2
    assert rm.max_retries == 2
    assert rm.max_entries == 0
    assert rm.allow_provisional_nodes is True
    assert rm.min_parse_confidence == 0.3
    assert rm.mapper_version == "v1"
    assert rm.prompt_version == "v1"


def test_reference_mapping_section_overrides_and_keeps_defaults(tmp_path) -> None:
    path = tmp_path / "mapping.toml"
    path.write_text(
        "[reference_mapping]\nenabled = false\nbatch_size = 3\nmax_entries = 7\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.reference_mapping.enabled is False
    assert cfg.reference_mapping.batch_size == 3
    assert cfg.reference_mapping.max_entries == 7
    # Unspecified keys retain the shipped defaults.
    assert cfg.reference_mapping.max_retries == 2
    assert cfg.reference_mapping.lease_seconds == 300

