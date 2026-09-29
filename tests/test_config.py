"""Configuration backward compatibility and additive research-kernel defaults."""

from __future__ import annotations

from pathlib import Path

from research_explorer.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_default_config_keeps_aco_and_gains_kernel_defaults() -> None:
    cfg = load_config(CONFIG_DIR / "default.toml")
    assert cfg.pipeline == "aco"
    assert cfg.providers.seed_routing == []
    assert cfg.llm.max_tokens == 2000
    assert cfg.llm.evaluation_max_tokens == 4000
    assert cfg.llm.structured_output_attempts == 2
    assert cfg.storage.research_db_path == "data/research.db"
    rk = cfg.research_kernel
    assert rk.policy == "greedy"
    assert rk.require_question is True
    assert rk.max_turns == 10
    assert rk.transient_retry_attempts == 2
    assert rk.evaluator_enabled is True
    assert rk.weights["integrity"] == 0.35


def test_nan_experiment_profiles_use_large_completion_budgets() -> None:
    quick = load_config(CONFIG_DIR / "profiles" / "quick.toml")
    full = load_config(CONFIG_DIR / "profiles" / "nan_full.toml")
    experiment = load_config(CONFIG_DIR / "profiles" / "experiment.toml")
    assert quick.llm.max_tokens == 16000
    assert quick.llm.evaluation_max_tokens == 16000
    assert full.llm.max_tokens == 32000
    assert full.llm.evaluation_max_tokens == 32000
    assert experiment.llm.structured_output_attempts == 3


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


def test_memory_examination_and_selection_defaults_on_shipped_config() -> None:
    cfg = load_config(CONFIG_DIR / "default.toml")
    assert cfg.memory.enabled is True
    assert cfg.memory.synthesis_words == 2000
    exam = cfg.examination
    assert exam.enabled is False
    assert exam.examiner_model == "gpt-5.6-sol"
    assert exam.examiner_provider == "openai"
    assert exam.selection_count == 30
    assert exam.holdout_count == 20
    assert exam.options_per_item == 4
    assert exam.allow_examiner_fallback is False
    assert cfg.terminal_selection.w_selection == 0.70
    assert cfg.terminal_selection.w_process == 0.20
    assert cfg.terminal_selection.w_grounding == 0.10
    assert cfg.baseline.enabled is True


def test_benchmark_test_profile_is_network_free() -> None:
    cfg = load_config(CONFIG_DIR / "profiles" / "benchmark_test.toml")
    assert cfg.examination.enabled is True
    assert cfg.examination.examiner_provider == "fake"
    assert cfg.examination.answer_model == "fake-answer-v1"
    assert cfg.examination.selection_count == 6
    assert cfg.examination.holdout_count == 4


def test_old_config_without_examination_sections_gets_defaults(tmp_path) -> None:
    path = tmp_path / "old.toml"
    path.write_text('log_level = "INFO"\n[aco]\ncolony_size = 3\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.examination.enabled is False
    assert cfg.memory.synthesis_words == 2000
    assert cfg.baseline.context_max_chars == 40000


def test_invalid_examination_settings_are_rejected(tmp_path) -> None:
    import pytest

    path = tmp_path / "bad.toml"
    path.write_text("[examination]\nselection_count = 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)

    path.write_text(
        "[terminal_selection]\nw_selection = 0.9\nw_process = 0.5\nw_grounding = 0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(path)


def test_old_agent_state_loads_without_structured_memory() -> None:
    from research_explorer.agents.state import AgentState

    legacy = {"id": "agent-0", "pos": "arxiv:1", "visited": ["arxiv:1"], "budget": 3}
    state = AgentState.model_validate(legacy)
    assert state.dossiers == {}
    assert state.claims == {}
    assert state.research_scope == ""
    assert state.scope_origin == "derived"
    assert state.synthesis == ""
    assert state.extraction_failures == []
    assert state.regenerate_synthesis()  # tolerant default synthesis

