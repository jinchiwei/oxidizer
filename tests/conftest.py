"""Shared pytest fixtures for the Oxidizer test suite."""
import pytest
from pathlib import Path
from oxidizer.profiles.schema import (
    StyleProfile,
    SentenceLengthMetrics,
    ParagraphMetrics,
    VoiceMetrics,
    PunctuationMetrics,
    TransitionConfig,
    VocabularyConfig,
    VoiceRules,
    FewShotExample,
)


@pytest.fixture
def test_profile():
    return StyleProfile(
        name="Test",
        version=1,
        source_documents=[],
        sentence_length=SentenceLengthMetrics(
            mean=24.1, median=23, std=10.1, range_min=6, range_max=77
        ),
        paragraph=ParagraphMetrics(mean_words=71, sentences_per_paragraph=(3, 4)),
        voice=VoiceMetrics(active_ratio=0.90, passive_contexts=[]),
        contractions=False,
        type_token_ratio=0.346,
        transitions=TransitionConfig(preferred=["while", "however"], acceptable=[]),
        vocabulary=VocabularyConfig(
            preferred=["robust"],
            banned_aiisms=["delve", "tapestry", "multifaceted"],
        ),
        punctuation=PunctuationMetrics(
            semicolons_per_100=12,
            parentheticals_per_100=25,
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
        style_prompt="Write in Jinchi Wei's voice. Use 'we'. No em dashes. No contractions.",
        few_shot_examples=[
            FewShotExample(
                category="results",
                text="Registration error was 1.43 +/- 0.30 mm.",
            )
        ],
    )


@pytest.fixture
def profiles_dir():
    return Path(__file__).parent.parent / "profiles"


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"
