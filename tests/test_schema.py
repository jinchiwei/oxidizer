"""Tests for oxidizer.profiles.schema dataclasses."""
import pytest
from oxidizer.profiles.schema import (
    SentenceLengthMetrics,
    ParagraphMetrics,
    VoiceMetrics,
    PunctuationMetrics,
    TransitionConfig,
    VocabularyConfig,
    VoiceRules,
    FewShotExample,
    StyleProfile,
)


def test_sentence_length_metrics():
    metrics = SentenceLengthMetrics(
        mean=24.1,
        median=23.0,
        std=10.1,
        range_min=6,
        range_max=77,
    )
    assert metrics.mean == 24.1
    assert metrics.median == 23.0
    assert metrics.std == 10.1
    assert metrics.range_min == 6
    assert metrics.range_max == 77


def test_style_profile_has_all_fields():
    profile = StyleProfile(
        name="Test Author",
        version=1,
        source_documents=["Doc A"],
        sentence_length=SentenceLengthMetrics(
            mean=20.0, median=19.0, std=5.0, range_min=5, range_max=50
        ),
        paragraph=ParagraphMetrics(mean_words=60, sentences_per_paragraph=[3, 4]),
        voice=VoiceMetrics(active_ratio=0.85, passive_contexts=["methods"]),
        contractions=False,
        type_token_ratio=0.35,
        transitions=TransitionConfig(preferred=["however"], acceptable=["furthermore"]),
        vocabulary=VocabularyConfig(preferred=["robust"], banned_aiisms=["Delve"]),
        punctuation=PunctuationMetrics(
            semicolons_per_100=10,
            parentheticals_per_100=20,
            em_dashes=0,
            inline_enumerations=True,
        ),
        voice_rules=VoiceRules(
            person="we",
            hedging=["would likely"],
            reasoning=True,
            problem_before_solution=True,
            quantitative_precision=True,
        ),
        style_prompt=None,
        few_shot_examples=[
            FewShotExample(category="intro", text="Sample text.")
        ],
    )
    assert profile.name == "Test Author"
    assert profile.version == 1
    assert profile.source_documents == ["Doc A"]
    assert profile.contractions is False
    assert profile.type_token_ratio == 0.35
    assert profile.style_prompt is None
    assert len(profile.few_shot_examples) == 1
    # Spot-check nested fields
    assert profile.sentence_length.mean == 20.0
    assert profile.voice.active_ratio == 0.85
    assert profile.transitions.preferred == ["however"]
    assert profile.punctuation.inline_enumerations is True
    assert profile.voice_rules.person == "we"


def test_banned_words_are_lowercase():
    vocab = VocabularyConfig(
        preferred=["robust"],
        banned_aiisms=["Delve", "TAPESTRY", "Multifaceted"],
    )
    assert vocab.banned_aiisms == ["delve", "tapestry", "multifaceted"]
