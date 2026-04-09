"""Integration tests for the full Oxidizer pipeline."""
from unittest.mock import MagicMock
from pathlib import Path

from oxidizer.parsers.markdown_parser import parse_markdown
from oxidizer.engine.revise import revise_section
from oxidizer.profiles.loader import load_profile
from oxidizer.scoring.reporter import compute_style_report
from oxidizer.scoring.metrics import count_banned_words
from oxidizer.parsers.latex_parser import parse_latex
from oxidizer.llm import is_api_available
from click.testing import CliRunner
from oxidizer.cli import cli

PROFILES_DIR = Path(__file__).parent.parent / "profiles"

SAMPLE_INPUT = """We collected data from 76 patients (42 female, 34 male; mean age 62.3 +/- 8.1 years) who underwent pretreatment MRI at UCSF between 2019 and 2023 [1]. All patients provided informed consent as approved by the institutional review board (IRB #19-28456).

Radiomics features were extracted using PyRadiomics (v3.0.1) from manually segmented regions of interest (ROIs) [2, 3]. A total of 107 features were computed across five categories."""

MOCK_RESTYLED = """We collected data from 76 patients (42 female, 34 male; mean age 62.3 +/- 8.1 years) who underwent pretreatment MRI at UCSF between 2019 and 2023 [1]. All patients provided informed consent as approved by the institutional review board (IRB #19-28456).

We extracted radiomics features using PyRadiomics (v3.0.1) from manually segmented regions of interest (ROIs) [2, 3]; a total of 107 features were computed across five categories."""


def test_full_revise_pipeline_with_mock_api():
    profile = load_profile("jinchi", search_paths=[PROFILES_DIR])
    sections = parse_markdown(f"## Methods\n\n{SAMPLE_INPUT}")
    assert len(sections) == 1
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=MOCK_RESTYLED)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    result = revise_section(sections[0], profile, client=mock_client)
    assert result.preservation is not None
    assert result.preservation.passed is True, f"Missing: {result.preservation.missing}"
    assert result.style_report is not None
    assert result.style_report.style_match_score > 0
    assert result.style_report.banned_words_found == []


def test_score_only_no_api():
    profile = load_profile("jinchi", search_paths=[PROFILES_DIR])
    report = compute_style_report(SAMPLE_INPUT, profile)
    assert report.style_match_score > 0
    assert report.banned_words_found == []
    assert report.contraction_count == 0


def test_scan_pipeline_local():
    profile = load_profile("jinchi", search_paths=[PROFILES_DIR])
    text = "We delve into this multifaceted problem."
    found = count_banned_words(text, profile.vocabulary.banned_aiisms)
    assert "delve" in found
    assert "multifaceted" in found


def test_latex_parsing():
    text = (
        r"\section{Methods}" + "\nWe collected data.\n"
        + r"\section{Results}" + "\nAccuracy was 0.95."
    )
    sections = parse_latex(text)
    assert len(sections) == 2


def test_api_mode_detection():
    result = is_api_available()
    assert isinstance(result, bool)


def test_all_cli_commands_have_help():
    runner = CliRunner()
    for cmd in ["score", "scan", "revise", "write", "diff", "compare", "validate-profile"]:
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"
