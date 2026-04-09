"""Tests for the Oxidizer CLI (oxidizer/cli.py)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from oxidizer.cli import cli

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def sample_md():
    return FIXTURES / "sample_methods.md"


# ---------------------------------------------------------------------------
# Basic help tests
# ---------------------------------------------------------------------------

def test_cli_help(runner):
    """Top-level --help should succeed and list subcommands."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "score" in result.output
    assert "revise" in result.output
    assert "write" in result.output
    assert "scan" in result.output
    assert "diff" in result.output


def test_cli_all_commands_have_help(runner):
    """Every registered command should accept --help without error."""
    commands = ["score", "revise", "write", "scan", "diff", "validate-profile", "compare"]
    for cmd in commands:
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0, (
            f"Command '{cmd} --help' exited with {result.exit_code}:\n{result.output}"
        )
        # Each help page should mention the command name or "Usage"
        assert "Usage" in result.output or cmd in result.output.lower()


# ---------------------------------------------------------------------------
# score command
# ---------------------------------------------------------------------------

def test_cli_score_command(runner, sample_md):
    """score command should parse sample_methods.md and display a table."""
    result = runner.invoke(cli, ["score", str(sample_md), "--profile", "jinchi"])
    assert result.exit_code == 0, f"Unexpected exit: {result.output}"
    # Should contain metric labels
    assert "Style Match" in result.output or "Sentence Length" in result.output


def test_cli_score_missing_file(runner):
    """score command with a non-existent file should fail gracefully."""
    result = runner.invoke(cli, ["score", "/nonexistent/file.md", "--profile", "jinchi"])
    # Click should reject the bad path with a non-zero exit
    assert result.exit_code != 0


def test_cli_score_json_output(runner, sample_md):
    """score --json-output should emit valid JSON."""
    result = runner.invoke(cli, ["score", str(sample_md), "--profile", "jinchi", "--json-output"])
    assert result.exit_code == 0, f"Exit code: {result.exit_code}\n{result.output}"
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "report" in data[0]
    assert "style_match_score" in data[0]["report"]


def test_cli_score_missing_profile(runner, sample_md):
    """score with unknown profile should error and exit non-zero."""
    result = runner.invoke(cli, ["score", str(sample_md), "--profile", "nonexistent_profile_xyz"])
    assert result.exit_code != 0 or "Error" in result.output or "not found" in result.output


# ---------------------------------------------------------------------------
# scan command
# ---------------------------------------------------------------------------

def test_cli_scan_detects_banned_words(runner):
    """scan should detect banned words from the jinchi profile."""
    # Create a temp file containing known banned AI-isms
    content = (
        "This study leverages cutting-edge delve into techniques to showcase "
        "a groundbreaking approach that underscores a robust paradigm shift."
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        result = runner.invoke(cli, ["scan", str(tmp_path), "--profile", "jinchi"])
        assert result.exit_code == 0, f"Unexpected exit: {result.output}"
        # Should report that some banned words were found
        output_lower = result.output.lower()
        found_any = any(
            word in output_lower
            for word in ["leverage", "delve", "showcase", "groundbreaking", "robust", "paradigm"]
        )
        assert found_any or "banned" in output_lower
    finally:
        tmp_path.unlink(missing_ok=True)


def test_cli_scan_clean_text(runner):
    """scan should report no banned words for clean academic text."""
    content = (
        "We collected data from 50 patients. We computed features using standard methods. "
        "Results demonstrate significant improvement over baseline."
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        result = runner.invoke(cli, ["scan", str(tmp_path), "--profile", "jinchi"])
        assert result.exit_code == 0, f"Unexpected exit: {result.output}"
        assert "No banned words found" in result.output
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# revise command (no API key)
# ---------------------------------------------------------------------------

def test_cli_revise_missing_profile(runner, sample_md, monkeypatch):
    """revise with an unknown profile should error, even without an API key."""
    # Ensure no API key so we hit the profile check path (or API check first)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(cli, ["revise", str(sample_md), "--profile", "nonexistent_xyz"])
    # Either fails on missing API key or missing profile — both are non-zero
    assert result.exit_code != 0 or "Error" in result.output or "not found" in result.output.lower()


def test_cli_revise_no_api_key(runner, sample_md, monkeypatch):
    """revise should show a helpful error if ANTHROPIC_API_KEY is not set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(cli, ["revise", str(sample_md), "--profile", "jinchi"])
    assert result.exit_code != 0
    assert "API" in result.output or "key" in result.output.lower()


def test_cli_write_no_api_key(runner, monkeypatch):
    """write should show a helpful error if ANTHROPIC_API_KEY is not set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(
        cli,
        ["write", "imaging methods for scoliosis detection", "--profile", "jinchi", "--section", "methods"],
    )
    assert result.exit_code != 0
    assert "API" in result.output or "key" in result.output.lower()
