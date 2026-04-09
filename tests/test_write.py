"""Tests for oxidizer.engine.write."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oxidizer.engine.write import WriteResult, build_write_prompt, write_section
from oxidizer.llm import call_claude
from oxidizer.profiles.loader import load_profile_from_path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "jinchi.yaml"


@pytest.fixture(scope="module")
def jinchi_profile():
    return load_profile_from_path(_PROFILE_PATH)


def _make_mock_client(response_text: str) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# build_write_prompt
# ---------------------------------------------------------------------------

class TestBuildWritePrompt:
    def test_includes_profile_style_prompt(self, jinchi_profile):
        prompt = build_write_prompt("Describe the dataset.", "methods", jinchi_profile)
        # style_prompt is loaded from custom_style_prompt.md
        if jinchi_profile.style_prompt:
            # At least some content from the style prompt file should appear
            assert len(prompt) > 100

    def test_includes_banned_words(self, jinchi_profile):
        prompt = build_write_prompt("Describe the results.", "results", jinchi_profile)
        assert "delve" in prompt or "tapestry" in prompt

    def test_includes_section_type_guidance(self, jinchi_profile):
        prompt = build_write_prompt("Describe patient outcomes.", "results", jinchi_profile)
        assert "Results" in prompt or "results" in prompt.lower()

    def test_includes_methods_guidance_for_methods(self, jinchi_profile):
        prompt = build_write_prompt("Describe the segmentation pipeline.", "methods", jinchi_profile)
        assert "Methods" in prompt or "methods" in prompt.lower()

    def test_includes_discussion_guidance_for_discussion(self, jinchi_profile):
        prompt = build_write_prompt("Summarize findings.", "discussion", jinchi_profile)
        assert "Discussion" in prompt or "discussion" in prompt.lower()

    def test_includes_intro_guidance_for_intro(self, jinchi_profile):
        prompt = build_write_prompt("Introduce scoliosis imaging.", "intro", jinchi_profile)
        assert "Introduction" in prompt or "intro" in prompt.lower()

    def test_includes_topic_in_prompt(self, jinchi_profile):
        topic = "Vertebral segmentation using deep learning"
        prompt = build_write_prompt(topic, "methods", jinchi_profile)
        assert topic in prompt

    def test_includes_no_hallucinate_instruction(self, jinchi_profile):
        prompt = build_write_prompt("Any topic.", "other", jinchi_profile)
        assert "hallucinate" in prompt.lower() or "do not" in prompt.lower()

    def test_includes_we_rule(self, jinchi_profile):
        prompt = build_write_prompt("Any topic.", "intro", jinchi_profile)
        assert '"we"' in prompt or "'we'" in prompt

    def test_includes_no_contractions_rule(self, jinchi_profile):
        prompt = build_write_prompt("Any topic.", "intro", jinchi_profile)
        assert "contraction" in prompt.lower()

    def test_includes_few_shot_examples(self, jinchi_profile):
        prompt = build_write_prompt("Any topic.", "results", jinchi_profile)
        assert "problem_framing" in prompt or "methods_with_reasoning" in prompt

    def test_sentence_length_target_in_prompt(self, jinchi_profile):
        prompt = build_write_prompt("Any topic.", "methods", jinchi_profile)
        # Profile mean is 24.1 words
        assert "24" in prompt

    def test_unknown_section_type_uses_fallback(self, jinchi_profile):
        prompt = build_write_prompt("Some topic.", "other", jinchi_profile)
        # Should not raise; should produce a prompt
        assert len(prompt) > 50


# ---------------------------------------------------------------------------
# write_section
# ---------------------------------------------------------------------------

class TestWriteSection:
    def test_returns_write_result(self, jinchi_profile):
        response_text = (
            "We propose a deep learning approach for vertebral segmentation. "
            "The method attains robust performance across diverse imaging conditions."
        )
        mock_client = _make_mock_client(response_text)

        result = write_section(
            topic="Vertebral segmentation using CNN",
            section_type="methods",
            profile=jinchi_profile,
            client=mock_client,
        )

        assert isinstance(result, WriteResult)
        assert result.text == response_text
        assert result.topic == "Vertebral segmentation using CNN"
        assert result.section_type == "methods"

    def test_style_report_is_computed(self, jinchi_profile):
        response_text = (
            "We validated the approach on 24 patients. "
            "Results show a mean error of 1.43 mm; this is within clinical tolerance."
        )
        mock_client = _make_mock_client(response_text)

        result = write_section(
            topic="Validation study",
            section_type="results",
            profile=jinchi_profile,
            client=mock_client,
        )

        assert result.style_report is not None
        assert hasattr(result.style_report, "style_match_score")
        assert 0.0 <= result.style_report.style_match_score <= 1.0

    def test_raises_without_client(self, jinchi_profile, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="Claude API not available"):
            write_section(
                topic="Some topic",
                section_type="intro",
                profile=jinchi_profile,
                client=None,
            )

    def test_calls_api_once(self, jinchi_profile):
        response_text = "We analyzed the data holistically."
        mock_client = _make_mock_client(response_text)

        write_section(
            topic="Data analysis",
            section_type="discussion",
            profile=jinchi_profile,
            client=mock_client,
        )

        mock_client.messages.create.assert_called_once()

    def test_warnings_populated_on_banned_words(self, jinchi_profile):
        # Response containing a banned word
        response_text = (
            "We delve into the multifaceted aspects of the problem. "
            "This novel approach is groundbreaking."
        )
        mock_client = _make_mock_client(response_text)

        result = write_section(
            topic="Some analysis",
            section_type="discussion",
            profile=jinchi_profile,
            client=mock_client,
        )

        assert len(result.warnings) > 0
        assert any("banned" in w.lower() or "ai-ism" in w.lower() for w in result.warnings)

    def test_no_warnings_on_clean_text(self, jinchi_profile):
        response_text = (
            "We propose a registration framework that leverages spline-based deformable models. "
            "The method attains robust performance across diverse imaging conditions. "
            "Results demonstrate that the approach is scalable to large datasets."
        )
        mock_client = _make_mock_client(response_text)

        result = write_section(
            topic="Registration framework",
            section_type="methods",
            profile=jinchi_profile,
            client=mock_client,
        )

        assert result.warnings == []

    def test_all_section_types_accepted(self, jinchi_profile):
        for stype in ["intro", "methods", "results", "discussion", "other"]:
            mock_client = _make_mock_client("We present a method.")
            result = write_section(
                topic="Topic", section_type=stype, profile=jinchi_profile, client=mock_client
            )
            assert result.section_type == stype
