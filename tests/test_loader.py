"""Tests for oxidizer.profiles.loader."""
import pytest
from pathlib import Path
from unittest.mock import patch
import yaml

from oxidizer.profiles.loader import (
    resolve_profile_path,
    load_profile,
    load_profile_from_path,
)
from oxidizer.profiles.schema import StyleProfile


# Resolve the actual profiles directory relative to this test file
PROFILES_DIR = Path(__file__).parent.parent / "profiles"


# ---------------------------------------------------------------------------
# test_resolve_profile_path_local
# ---------------------------------------------------------------------------
def test_resolve_profile_path_local():
    result = resolve_profile_path("jinchi", search_paths=[PROFILES_DIR])
    assert result is not None
    assert result.exists()
    assert result.name == "jinchi.yaml"


# ---------------------------------------------------------------------------
# test_resolve_profile_path_not_found
# ---------------------------------------------------------------------------
def test_resolve_profile_path_not_found():
    result = resolve_profile_path("nonexistent_profile_xyz", search_paths=[PROFILES_DIR])
    assert result is None


# ---------------------------------------------------------------------------
# test_load_profile_from_yaml
# ---------------------------------------------------------------------------
def test_load_profile_from_yaml():
    profile = load_profile_from_path(PROFILES_DIR / "jinchi.yaml")

    # Top-level identity fields
    assert isinstance(profile, StyleProfile)
    assert profile.name == "Jinchi Wei"
    assert profile.version == 1
    assert len(profile.source_documents) == 3
    assert "MSE Thesis (2021)" in profile.source_documents

    # Sentence length
    assert profile.sentence_length.mean == pytest.approx(24.1)
    assert profile.sentence_length.median == 23
    assert profile.sentence_length.std == pytest.approx(10.1)
    assert profile.sentence_length.range_min == 6
    assert profile.sentence_length.range_max == 77

    # Paragraph
    assert profile.paragraph.mean_words == 71
    assert profile.paragraph.sentences_per_paragraph == [3, 4]

    # Voice
    assert profile.voice.active_ratio == pytest.approx(0.90)
    assert "methods" in profile.voice.passive_contexts

    # Contractions / TTR
    assert profile.contractions is False
    assert profile.type_token_ratio == pytest.approx(0.346)

    # Transitions
    assert "however" in profile.transitions.preferred
    assert "furthermore" in profile.transitions.acceptable

    # Vocabulary
    assert "robust" in profile.vocabulary.preferred
    assert "delve" in profile.vocabulary.banned_aiisms

    # Punctuation — YAML key mapping
    assert profile.punctuation.semicolons_per_100 == 12
    assert profile.punctuation.parentheticals_per_100 == 25
    assert profile.punctuation.em_dashes == 0
    assert profile.punctuation.inline_enumerations is True

    # Voice rules
    assert profile.voice_rules.person == "we"
    assert "would likely" in profile.voice_rules.hedging
    assert profile.voice_rules.reasoning is True
    assert profile.voice_rules.problem_before_solution is True
    assert profile.voice_rules.quantitative_precision is True

    # Few-shot examples
    assert len(profile.few_shot_examples) == 5
    categories = [e.category for e in profile.few_shot_examples]
    assert "problem_framing" in categories
    assert "results_with_precision" in categories


# ---------------------------------------------------------------------------
# test_load_profile_includes_style_prompt
# ---------------------------------------------------------------------------
def test_load_profile_includes_style_prompt():
    profile = load_profile_from_path(PROFILES_DIR / "jinchi.yaml")
    assert profile.style_prompt is not None
    assert len(profile.style_prompt) > 0
    # Check it loaded real content (not just a file path string)
    assert "style" in profile.style_prompt.lower() or "voice" in profile.style_prompt.lower()


# ---------------------------------------------------------------------------
# test_banned_words_normalized_to_lowercase
# ---------------------------------------------------------------------------
def test_banned_words_normalized_to_lowercase():
    profile = load_profile_from_path(PROFILES_DIR / "jinchi.yaml")
    for word in profile.vocabulary.banned_aiisms:
        assert word == word.lower(), f"Expected lowercase but got: {word!r}"


# ---------------------------------------------------------------------------
# Additional tests from eng review
# ---------------------------------------------------------------------------

def test_load_profile_invalid_yaml(tmp_path):
    """Loading a file with invalid YAML should raise a descriptive error."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("name: [\nunclosed bracket\n")
    with pytest.raises(yaml.YAMLError):
        load_profile_from_path(bad_yaml)


def test_load_profile_missing_style_prompt_file(tmp_path):
    """When style_prompt_file references a nonexistent file, raise FileNotFoundError."""
    # Write a minimal valid profile that points to a nonexistent prompt
    minimal = {
        "name": "Ghost",
        "version": 1,
        "source_documents": ["Doc A"],
        "metrics": {
            "sentence_length": {
                "mean": 20.0,
                "median": 19.0,
                "std": 5.0,
                "range": [5, 50],
            },
            "paragraph_length": {
                "mean": 60,
                "sentences_per_paragraph": [3, 4],
            },
            "voice": {
                "active_ratio": 0.85,
                "passive_contexts": ["methods"],
            },
            "contractions": False,
            "type_token_ratio": 0.35,
        },
        "transitions": {"preferred": ["however"], "acceptable": ["furthermore"]},
        "vocabulary": {"preferred": ["robust"], "banned_aiisms": ["delve"]},
        "punctuation": {
            "semicolons_per_100_sentences": 10,
            "parenthetical_pairs_per_100_sentences": 20,
            "em_dashes": 0,
            "inline_enumerations": True,
        },
        "voice_rules": {
            "person": "we",
            "hedging": ["would likely"],
            "reasoning": True,
            "problem_before_solution": True,
            "quantitative_precision": True,
        },
        "style_prompt_file": "does_not_exist.md",
        "few_shot_examples": [],
    }
    profile_path = tmp_path / "ghost.yaml"
    profile_path.write_text(yaml.dump(minimal))
    with pytest.raises(FileNotFoundError):
        load_profile_from_path(profile_path)
