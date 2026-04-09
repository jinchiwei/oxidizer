"""Style profile schema dataclasses for Oxidizer."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SentenceLengthMetrics:
    mean: float
    median: float
    std: float
    range_min: int
    range_max: int


@dataclass
class ParagraphMetrics:
    mean_words: int
    sentences_per_paragraph: list[int]


@dataclass
class VoiceMetrics:
    active_ratio: float
    passive_contexts: list[str]


@dataclass
class PunctuationMetrics:
    semicolons_per_100: float
    parentheticals_per_100: float
    em_dashes: int
    inline_enumerations: bool


@dataclass
class TransitionConfig:
    preferred: list[str]
    acceptable: list[str]


@dataclass
class VocabularyConfig:
    preferred: list[str]
    banned_aiisms: list[str]

    def __post_init__(self):
        self.banned_aiisms = [w.lower() for w in self.banned_aiisms]


@dataclass
class VoiceRules:
    person: str
    hedging: list[str]
    reasoning: bool
    problem_before_solution: bool
    quantitative_precision: bool


@dataclass
class FewShotExample:
    category: str
    text: str


@dataclass
class AIDetectionConfig:
    """Configuration for AI detection strictness."""
    p2_density_threshold: int = 3
    p2_density_threshold_long: int = 5
    structural_enabled: bool = True
    statistical_enabled: bool = True
    context_exemptions: dict = field(default_factory=dict)


@dataclass
class StyleProfile:
    name: str
    version: int
    source_documents: list[str]
    sentence_length: SentenceLengthMetrics
    paragraph: ParagraphMetrics
    voice: VoiceMetrics
    contractions: bool
    type_token_ratio: float
    transitions: TransitionConfig
    vocabulary: VocabularyConfig
    punctuation: PunctuationMetrics
    voice_rules: VoiceRules
    style_prompt: Optional[str]
    few_shot_examples: list[FewShotExample]
    ai_detection: AIDetectionConfig = field(default_factory=AIDetectionConfig)
