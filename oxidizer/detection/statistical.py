"""Statistical AI signal detection for Oxidizer.

Three complementary metrics probe writing rhythm and lexical diversity:

- Burstiness: variance in sentence-length pace.  Human writing is bursty
  (B ≈ 0.5-1.0); AI output is metronomic (B ≈ 0.1-0.3).
- Trigram repetition: fraction of unique trigrams that recur.  Human < 0.05;
  AI > 0.10.
- Sentence CoV: coefficient of variation of sentence word-counts.  Human
  CoV > 0.40; AI CoV < 0.25.
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StatisticalReport:
    """Results of all three statistical AI-signal analyses."""

    burstiness: float
    trigram_repetition: float
    sentence_cov: float
    ai_risk_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sentence tokenisation helper
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using nltk, with a download fallback."""
    try:
        from nltk.tokenize import sent_tokenize
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = sent_tokenize(text)
    except Exception:
        # Last-resort fallback: split on period/exclamation/question mark
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [s for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_burstiness(text: str) -> float:
    """Measure variance in writing pace via sentence-length burstiness.

    Formula: raw B = (std - mean) / (std + mean) for word counts per
    sentence, then normalised from [-1, 1] to [0, 1] via (raw + 1) / 2.
    Result is clamped to [0, 1].

    Human writing is bursty (≈ 0.5-1.0); AI output is metronomic (≈ 0.1-0.3).

    Returns 0.0 for empty text or text with fewer than 3 sentences.
    """
    if not text or not text.strip():
        return 0.0

    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return 0.0

    lengths = [len(s.split()) for s in sentences]
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0

    std = statistics.stdev(lengths)
    denom = std + mean
    if denom == 0:
        return 0.0

    raw = (std - mean) / denom
    normalised = (raw + 1) / 2
    return max(0.0, min(1.0, normalised))


def compute_trigram_repetition(text: str) -> float:
    """Compute fraction of unique trigrams that appear more than once.

    Human text typically scores < 0.05; AI-generated text > 0.10.

    Returns 0.0 for empty text or text with fewer than 4 words.
    """
    if not text or not text.strip():
        return 0.0

    words = text.lower().split()
    if len(words) < 4:
        return 0.0

    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    unique = len(counts)
    if unique == 0:
        return 0.0

    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / unique


def compute_sentence_cov(text: str) -> float:
    """Coefficient of variation of sentence word-counts (stdev / mean).

    Human writing has high CoV (> 0.40); AI prose has low CoV (< 0.25).

    Returns 0.0 for empty text or text with fewer than 3 sentences.
    """
    if not text or not text.strip():
        return 0.0

    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return 0.0

    lengths = [len(s.split()) for s in sentences]
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0

    std = statistics.stdev(lengths)
    return std / mean


# ---------------------------------------------------------------------------
# Composite analysis
# ---------------------------------------------------------------------------

def analyze_statistical_signals(text: str) -> StatisticalReport:
    """Run all three statistical analyses and flag AI-like signals.

    Flag conditions:
    - burstiness < 0.3 AND text > 50 words → metronomic sentence rhythm
    - trigram_repetition > 0.10 → repetitive trigram patterns
    - sentence_cov < 0.25 AND >= 4 sentences → uniform sentence lengths

    Parameters
    ----------
    text:
        The academic text to analyse.

    Returns
    -------
    StatisticalReport
        Populated dataclass with metric values and any AI-risk flags.
    """
    burstiness = compute_burstiness(text)
    trigram_repetition = compute_trigram_repetition(text)
    sentence_cov = compute_sentence_cov(text)

    flags: list[str] = []

    word_count = len(text.split()) if text else 0
    if burstiness < 0.3 and word_count > 50:
        flags.append("metronomic sentence rhythm (low burstiness)")

    if trigram_repetition > 0.10:
        flags.append("repetitive trigram patterns")

    sentences = _split_sentences(text) if text and text.strip() else []
    if sentence_cov < 0.25 and len(sentences) >= 4:
        flags.append("uniform sentence lengths (low CoV)")

    return StatisticalReport(
        burstiness=burstiness,
        trigram_repetition=trigram_repetition,
        sentence_cov=sentence_cov,
        ai_risk_flags=flags,
    )
