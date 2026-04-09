"""Tests for oxidizer.engine.revise."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oxidizer.engine.revise import build_restyle_prompt, revise_section
from oxidizer.parsers.markdown_parser import Section
from oxidizer.profiles.loader import load_profile_from_path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "jinchi.yaml"


@pytest.fixture(scope="module")
def jinchi_profile():
    return load_profile_from_path(_PROFILE_PATH)


@pytest.fixture
def sample_section():
    return Section(
        heading="Results",
        body=(
            "The system achieved a mean registration error of 1.43 ± 0.30 mm "
            "across all 24 patients. We used a CNN-based approach (Smith et al., 2023) "
            "to segment vertebrae, as shown in Figure 1."
        ),
        context="results",
        level=2,
    )


def _make_mock_client(response_text: str) -> MagicMock:
    """Build a mock Anthropic client that returns the given text."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# build_restyle_prompt
# ---------------------------------------------------------------------------

class TestBuildRestylePrompt:
    def test_includes_style_instructions_from_profile(self, jinchi_profile, sample_section):
        prompt = build_restyle_prompt(sample_section, jinchi_profile, [])
        # Style rules section must appear
        assert "Style Rules" in prompt
        # Target sentence length from profile
        assert "24" in prompt  # mean ~24.1

    def test_includes_banned_words_in_instructions(self, jinchi_profile, sample_section):
        prompt = build_restyle_prompt(sample_section, jinchi_profile, [])
        # At least one banned word should appear in the instructions
        assert "delve" in prompt or "tapestry" in prompt

    def test_includes_locked_entities(self, jinchi_profile, sample_section):
        locked = ["1.43 ± 0.30 mm", "[1]", "Figure 1"]
        prompt = build_restyle_prompt(sample_section, jinchi_profile, locked)
        assert "1.43 ± 0.30 mm" in prompt
        assert "[1]" in prompt
        assert "Figure 1" in prompt

    def test_locked_entities_section_absent_when_empty(self, jinchi_profile, sample_section):
        prompt = build_restyle_prompt(sample_section, jinchi_profile, [])
        assert "Locked Entities" not in prompt

    def test_includes_section_context(self, jinchi_profile, sample_section):
        prompt = build_restyle_prompt(sample_section, jinchi_profile, [])
        assert "results" in prompt.lower() or "Results" in prompt

    def test_includes_section_heading(self, jinchi_profile, sample_section):
        prompt = build_restyle_prompt(sample_section, jinchi_profile, [])
        assert "Results" in prompt

    def test_includes_few_shot_examples(self, jinchi_profile, sample_section):
        prompt = build_restyle_prompt(sample_section, jinchi_profile, [])
        # Profile has few_shot_examples; at least one category name should appear
        assert "problem_framing" in prompt or "methods_with_reasoning" in prompt

    def test_includes_original_text_in_task(self, jinchi_profile, sample_section):
        prompt = build_restyle_prompt(sample_section, jinchi_profile, [])
        assert sample_section.body in prompt

    def test_includes_no_em_dash_rule(self, jinchi_profile, sample_section):
        prompt = build_restyle_prompt(sample_section, jinchi_profile, [])
        assert "em dash" in prompt.lower() or "em-dash" in prompt.lower()

    def test_includes_we_rule(self, jinchi_profile, sample_section):
        prompt = build_restyle_prompt(sample_section, jinchi_profile, [])
        assert '"we"' in prompt or "'we'" in prompt


# ---------------------------------------------------------------------------
# revise_section
# ---------------------------------------------------------------------------

class TestReviseSection:
    def test_calls_api_and_returns_pipeline_result(self, jinchi_profile, sample_section):
        response_text = (
            "We attained a mean registration error of 1.43 ± 0.30 mm across all 24 patients. "
            "We employed a CNN-based approach (Smith et al., 2023) to segment vertebrae, "
            "as shown in Figure 1."
        )
        mock_client = _make_mock_client(response_text)

        result = revise_section(sample_section, jinchi_profile, client=mock_client)

        # API was called at least once
        mock_client.messages.create.assert_called()
        # Result has correct structure
        assert result.text == response_text
        assert result.original_text == sample_section.body
        assert result.heading == "Results"
        assert result.style_report is not None
        assert result.preservation is not None

    def test_result_text_matches_api_response(self, jinchi_profile, sample_section):
        response_text = "We observed robust performance across 24 patients."
        mock_client = _make_mock_client(response_text)

        result = revise_section(sample_section, jinchi_profile, client=mock_client)

        assert result.text == response_text

    def test_retries_on_entity_failure(self, jinchi_profile):
        """Client returns text missing an entity first, then correct text."""
        # Section with a specific entity
        section = Section(
            heading="Methods",
            body="Registration error was 1.43 ± 0.30 mm (Smith et al., 2023).",
            context="methods",
            level=2,
        )

        # First response is missing "(Smith et al., 2023)"
        bad_response = "Registration error was 1.43 ± 0.30 mm."
        # Second response includes the entity
        good_response = "Registration error was 1.43 ± 0.30 mm (Smith et al., 2023)."

        mock_client = MagicMock()
        mock_bad = MagicMock()
        mock_bad.content = [MagicMock(text=bad_response)]
        mock_good = MagicMock()
        mock_good.content = [MagicMock(text=good_response)]

        mock_client.messages.create.side_effect = [mock_bad, mock_good]

        result = revise_section(section, jinchi_profile, client=mock_client, max_retries=2)

        # Should have retried
        assert mock_client.messages.create.call_count == 2
        assert result.retries == 1
        assert result.text == good_response

    def test_returns_retries_zero_on_first_success(self, jinchi_profile, sample_section):
        response_text = (
            "We attained 1.43 ± 0.30 mm error across 24 patients (Smith et al., 2023), "
            "as shown in Figure 1."
        )
        mock_client = _make_mock_client(response_text)

        result = revise_section(sample_section, jinchi_profile, client=mock_client)

        assert result.retries == 0

    def test_gives_up_after_max_retries(self, jinchi_profile):
        """After max_retries exhausted, returns best effort with warnings."""
        section = Section(
            heading="Results",
            body="Error was 99.9% (Jones, 2022).",
            context="results",
            level=2,
        )
        # Response always missing "(Jones, 2022)"
        always_bad = "Error was 99.9%."
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=always_bad)]
        mock_client.messages.create.return_value = mock_resp

        result = revise_section(section, jinchi_profile, client=mock_client, max_retries=1)

        # Should have tried 2 times total (1 initial + 1 retry)
        assert mock_client.messages.create.call_count == 2
        # Preservation failed — warnings should mention missing entities
        assert any("missing" in w.lower() or "entities" in w.lower() for w in result.warnings)

    def test_style_report_attached(self, jinchi_profile, sample_section):
        response_text = "We validated the approach across 24 patients with robust results."
        mock_client = _make_mock_client(response_text)

        result = revise_section(sample_section, jinchi_profile, client=mock_client)

        assert result.style_report is not None
        assert hasattr(result.style_report, "style_match_score")

    def test_entities_extracted_from_original(self, jinchi_profile, sample_section):
        response_text = (
            "We attained 1.43 ± 0.30 mm error across 24 patients (Smith et al., 2023), "
            "as shown in Figure 1."
        )
        mock_client = _make_mock_client(response_text)

        result = revise_section(sample_section, jinchi_profile, client=mock_client)

        # Entities should come from original text
        assert result.entities is not None
        all_e = result.entities.all_entities()
        assert any("1.43" in e for e in all_e)
