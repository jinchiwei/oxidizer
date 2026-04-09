"""Sentence-level structural AI pattern detectors for Oxidizer.

Detects patterns characteristic of AI-generated academic text at the
structural / sentence level rather than at the vocabulary level.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class PatternType(Enum):
    REPETITIVE_STARTERS   = "repetitive_starters"
    LENGTH_UNIFORMITY     = "length_uniformity"
    RULE_OF_THREE         = "rule_of_three"
    COPULA_AVOIDANCE      = "copula_avoidance"
    NEGATIVE_PARALLELISM  = "negative_parallelism"
    EM_DASH_OVERUSE       = "em_dash_overuse"
    CONFIDENCE_STACKING   = "confidence_stacking"


@dataclass
class StructuralFinding:
    """A single structural pattern finding."""

    pattern: PatternType
    description: str
    evidence: str
    severity: str  # "high", "medium", "low"


# ---------------------------------------------------------------------------
# NLTK sentence tokenizer with download fallback
# ---------------------------------------------------------------------------

def _sent_tokenize(text: str) -> list[str]:
    """Tokenize *text* into sentences using NLTK, downloading if needed."""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except LookupError:
        import nltk
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _first_word(sentence: str) -> str:
    """Return the first alphabetic word of *sentence*, lowercased."""
    match = re.match(r"\s*([A-Za-z]+)", sentence)
    return match.group(1).lower() if match else ""


def _detect_repetitive_starters(sentences: list[str]) -> StructuralFinding | None:
    """Flag if 40%+ of sentences (min 3) begin with the same word."""
    if len(sentences) < 3:
        return None

    from collections import Counter
    starters = [_first_word(s) for s in sentences if _first_word(s)]
    if not starters:
        return None

    counts = Counter(starters)
    top_word, top_count = counts.most_common(1)[0]
    ratio = top_count / len(starters)

    if top_count >= 3 and ratio >= 0.40:
        pct = int(ratio * 100)
        return StructuralFinding(
            pattern=PatternType.REPETITIVE_STARTERS,
            description=(
                f"{pct}% of sentences start with \"{top_word}\" "
                f"({top_count}/{len(starters)})"
            ),
            evidence=f'Repeated starter: "{top_word}"',
            severity="high",
        )
    return None


def _detect_length_uniformity(sentences: list[str]) -> StructuralFinding | None:
    """Flag if sentence-length CoV < 0.25 (AI-like uniformity).

    Requires at least 4 sentences.
    """
    if len(sentences) < 4:
        return None

    lengths = [len(s.split()) for s in sentences]
    mean = statistics.mean(lengths)
    if mean == 0:
        return None

    stdev = statistics.pstdev(lengths)
    cov = stdev / mean

    if cov < 0.25:
        return StructuralFinding(
            pattern=PatternType.LENGTH_UNIFORMITY,
            description=(
                f"Sentence lengths are unusually uniform (CoV={cov:.2f}; "
                f"human text typically CoV>0.40)"
            ),
            evidence=f"Lengths: {lengths}",
            severity="high",
        )
    return None


# Triad pattern: "X, Y, and Z" or "X, Y, or Z"
_TRIAD_RE = re.compile(
    r"\b[\w\-]+(?:\s+[\w\-]+){0,3}"     # first item (1-4 words)
    r",\s+"
    r"[\w\-]+(?:\s+[\w\-]+){0,3}"        # second item
    r",\s+(?:and|or)\s+"
    r"[\w\-]+(?:\s+[\w\-]+){0,3}",       # third item
    re.IGNORECASE,
)


def _detect_rule_of_three(text: str) -> StructuralFinding | None:
    """Flag 2+ triadic 'X, Y, and Z' patterns in the same text."""
    matches = _TRIAD_RE.findall(text)
    if len(matches) >= 2:
        preview = "; ".join(f'"{m}"' for m in matches[:3])
        return StructuralFinding(
            pattern=PatternType.RULE_OF_THREE,
            description=(
                f"Found {len(matches)} triadic list(s) — a common AI rhythm "
                f"(one is acceptable; multiple is a signal)"
            ),
            evidence=preview,
            severity="medium",
        )
    return None


# Copula avoidance: AI substitutes "serves as", "boasts", "features [a/the]" for is/has
_COPULA_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bserves\s+as\b", re.IGNORECASE),
    re.compile(r"\bboasts\b", re.IGNORECASE),
    re.compile(r"\bfeatures\s+(?:a|an|the)\b", re.IGNORECASE),
]


def _detect_copula_avoidance(text: str) -> StructuralFinding | None:
    """Flag if 2+ copula-avoidance substitutions are found."""
    hits: list[str] = []
    for pattern in _COPULA_PATTERNS:
        for match in pattern.finditer(text):
            hits.append(match.group(0))

    if len(hits) >= 2:
        preview = "; ".join(f'"{h}"' for h in hits[:5])
        return StructuralFinding(
            pattern=PatternType.COPULA_AVOIDANCE,
            description=(
                f"Found {len(hits)} copula-avoidance substitution(s) "
                f"(\"serves as\", \"boasts\", \"features a/the\" used instead of "
                f"\"is\"/\"has\")"
            ),
            evidence=preview,
            severity="medium",
        )
    return None


# Negative parallelism: "not just X, it's Y" / "not just X, it is Y"
_NEG_PARALLEL_RE = re.compile(
    r"\bnot\s+just\b.{1,80}?\bit(?:'s|\s+is)\b",
    re.IGNORECASE | re.DOTALL,
)


def _detect_negative_parallelism(text: str) -> StructuralFinding | None:
    """Flag any 'It's not just X, it's Y' pattern."""
    matches = _NEG_PARALLEL_RE.findall(text)
    if matches:
        preview = f'"{matches[0][:80].strip()}"'
        return StructuralFinding(
            pattern=PatternType.NEGATIVE_PARALLELISM,
            description='Found "not just X, it\'s Y" negative parallelism construct',
            evidence=preview,
            severity="medium",
        )
    return None


def _detect_em_dash_overuse(text: str) -> StructuralFinding | None:
    """Flag 2+ em dashes (—) when rate > 2 per 1000 words."""
    em_dashes = text.count("\u2014")  # —
    if em_dashes < 2:
        return None

    word_count = len(text.split())
    if word_count == 0:
        return None

    rate = em_dashes / word_count * 1000
    if rate > 2.0:
        return StructuralFinding(
            pattern=PatternType.EM_DASH_OVERUSE,
            description=(
                f"Found {em_dashes} em dash(es) "
                f"({rate:.1f} per 1000 words; threshold: 2.0)"
            ),
            evidence=f"{em_dashes} em dash(es) in {word_count} words",
            severity="medium",
        )
    return None


_CONFIDENCE_ADVERBS: tuple[str, ...] = (
    "Interestingly",
    "Notably",
    "Surprisingly",
    "Importantly",
    "Remarkably",
    "Significantly",
    "Crucially",
    "Strikingly",
)

# Compiled pattern: any of those adverbs at the start of a sentence (after
# optional whitespace), followed immediately by a comma.
_CONFIDENCE_RE = re.compile(
    r"(?:^|\.\s+|\?\s+|!\s+)(" + "|".join(_CONFIDENCE_ADVERBS) + r"),",
    re.IGNORECASE,
)


def _detect_confidence_stacking(text: str) -> StructuralFinding | None:
    """Flag 2+ sentences opening with confidence adverbs."""
    matches = _CONFIDENCE_RE.findall(text)
    if len(matches) >= 2:
        adverbs = [m.capitalize() for m in matches]
        preview = ", ".join(f'"{a},"' for a in adverbs[:5])
        return StructuralFinding(
            pattern=PatternType.CONFIDENCE_STACKING,
            description=(
                f"Found {len(matches)} confidence-stacking adverb opener(s) — "
                f"a hallmark of AI-inflated prose"
            ),
            evidence=preview,
            severity="medium",
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_structural_patterns(text: str) -> list[StructuralFinding]:
    """Detect sentence-level AI structural patterns in *text*.

    Parameters
    ----------
    text:
        The academic text to analyse.

    Returns
    -------
    list[StructuralFinding]
        One entry per pattern detected; empty list when text is clean.
    """
    if not text or not text.strip():
        return []

    sentences = _sent_tokenize(text)
    findings: list[StructuralFinding] = []

    # Sentence-level checks
    finding = _detect_repetitive_starters(sentences)
    if finding:
        findings.append(finding)

    finding = _detect_length_uniformity(sentences)
    if finding:
        findings.append(finding)

    # Full-text checks
    finding = _detect_rule_of_three(text)
    if finding:
        findings.append(finding)

    finding = _detect_copula_avoidance(text)
    if finding:
        findings.append(finding)

    finding = _detect_negative_parallelism(text)
    if finding:
        findings.append(finding)

    finding = _detect_em_dash_overuse(text)
    if finding:
        findings.append(finding)

    finding = _detect_confidence_stacking(text)
    if finding:
        findings.append(finding)

    return findings
