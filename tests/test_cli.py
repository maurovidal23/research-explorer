"""CLI acceptance path for the research-kernel pipeline selection."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from research_explorer.cli import app

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
runner = CliRunner()


def test_config_command_reports_kernel_profile() -> None:
    result = runner.invoke(
        app, ["config", "--config", str(CONFIG_DIR / "profiles" / "kernel_quick.toml")]
    )
    assert result.exit_code == 0
    assert "Pipeline: research-kernel" in result.output
    assert "ResearchKernel: policy=greedy" in result.output


def test_explore_rejects_kernel_run_without_enabled_providers(tmp_path) -> None:
    config = tmp_path / "no_providers.toml"
    config.write_text(
        'pipeline = "research-kernel"\n[providers]\ndefault = "openalex"\nactive = []\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "explore",
            "10.1038/nrn3241",
            "How do extracellular fields originate?",
            "--pipeline",
            "research-kernel",
            "--config",
            str(config),
        ],
    )
    assert result.exit_code != 0
    assert "No providers enabled" in result.output
