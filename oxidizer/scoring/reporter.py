"""Style-match reporter for Oxidizer.

Computes a weighted composite style score given a text and a StyleProfile.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from oxidizer.profiles.schema import StyleProfile
from oxidizer.scoring.metrics import (
    compute_active_voice_ratio,
    compute_sentence_lengths,
    compute_transition_score,
    count_banned_words,
    count_contractions,
    count_parentheticals_per_100,
    count_semicolons_per_100,
)

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

_WEIGHTS: dict[str, float] = {
    "sentence_length_mean": 0.20,
    "sentence_length_variance": 0.10,
    "active_voice": 0.15,
    "banned_words": 0.20,
    "semicolons": 0.10,
    "parentheticals": 0.10,
    "transitions": 0.10,
    "contractions": 0.05,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [*lo*, *hi*]."""
    return max(lo, min(hi, value))


def _proximity_score(actual: float, target: float) -> float:
    """Return clamp(1 - |actual - target| / target, 0, 1).

    If *target* is 0, returns 1.0 when *actual* is also 0, else 0.0.
    """
    if target == 0.0:
        return 1.0 if actual == 0.0 else 0.0
    return _clamp(1.0 - abs(actual - target) / target)


# ---------------------------------------------------------------------------
# StyleReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class StyleReport:
    """Comprehensive style-match report for a single text sample."""

    # Composite
    style_match_score: float

    # Sentence length
    sentence_length_mean: float
    sentence_length_std: float
    sentence_length_target_mean: float
    sentence_length_target_std: float

    # Voice
    active_voice_ratio: float
    active_voice_target: float

    # Vocabulary
    banned_words_found: list[str]

    # Punctuation
    semicolons_per_100: float
    semicolons_target: float
    parentheticals_per_100: float
    parentheticals_target: float

    # Contractions
    contraction_count: int

    # Transitions
    transition_score: float

    # Per-dimension sub-scores
    sub_scores: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_style_report(text: str, profile: StyleProfile) -> StyleReport:
    """Compute all style metrics and a weighted composite score.

    Args:
        text: The text to evaluate.
        profile: A loaded :class:`StyleProfile` providing targets.

    Returns:
        A populated :class:`StyleReport`.
    """
    # ------------------------------------------------------------------ #
    # Raw metric computation
    # ------------------------------------------------------------------ #
    sentence_lengths = compute_sentence_lengths(text)
    if sentence_lengths:
        sl_mean = statistics.mean(sentence_lengths)
        sl_std = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    else:
        sl_mean = 0.0
        sl_std = 0.0

    active_ratio = compute_active_voice_ratio(text)

    banned = count_banned_words(text, profile.vocabulary.banned_aiisms)

    semicolons = count_semicolons_per_100(text)
    parentheticals = count_parentheticals_per_100(text)

    contractions = count_contractions(text)

    transition_sc = compute_transition_score(text, profile.transitions.preferred)

    # ------------------------------------------------------------------ #
    # Targets from profile
    # ------------------------------------------------------------------ #
    target_sl_mean = profile.sentence_length.mean
    target_sl_std = profile.sentence_length.std
    target_active = profile.voice.active_ratio
    target_semi = profile.punctuation.semicolons_per_100
    target_paren = profile.punctuation.parentheticals_per_100

    # ------------------------------------------------------------------ #
    # Sub-scores
    # ------------------------------------------------------------------ #
    sub_scores: dict[str, float] = {}

    # Sentence length mean
    sub_scores["sentence_length_mean"] = _proximity_score(sl_mean, target_sl_mean)

    # Sentence length variance (compare stdev)
    sub_scores["sentence_length_variance"] = _proximity_score(sl_std, target_sl_std)

    # Active voice
    sub_scores["active_voice"] = _proximity_score(active_ratio, target_active)

    # Banned words: 1.0 if zero, else max(0, 1 - 0.1 * count)
    banned_count = len(banned)
    if banned_count == 0:
        sub_scores["banned_words"] = 1.0
    else:
        sub_scores["banned_words"] = max(0.0, 1.0 - 0.1 * banned_count)

    # Semicolons
    sub_scores["semicolons"] = _proximity_score(semicolons, target_semi)

    # Parentheticals
    sub_scores["parentheticals"] = _proximity_score(parentheticals, target_paren)

    # Transitions
    sub_scores["transitions"] = transition_sc  # already in [0, 1]

    # Contractions: 1.0 if zero, else 0.0
    sub_scores["contractions"] = 1.0 if contractions == 0 else 0.0

    # ------------------------------------------------------------------ #
    # Weighted composite
    # ------------------------------------------------------------------ #
    composite = sum(
        _WEIGHTS[dim] * score for dim, score in sub_scores.items()
    )

    return StyleReport(
        style_match_score=composite,
        sentence_length_mean=sl_mean,
        sentence_length_std=sl_std,
        sentence_length_target_mean=target_sl_mean,
        sentence_length_target_std=target_sl_std,
        active_voice_ratio=active_ratio,
        active_voice_target=target_active,
        banned_words_found=banned,
        semicolons_per_100=semicolons,
        semicolons_target=target_semi,
        parentheticals_per_100=parentheticals,
        parentheticals_target=target_paren,
        contraction_count=contractions,
        transition_score=transition_sc,
        sub_scores=sub_scores,
    )
