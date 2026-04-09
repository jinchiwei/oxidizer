"""Tiered AI vocabulary scanner for Oxidizer.

Three severity tiers:
- P0: Always flag, even one occurrence. Credibility killers.
- P1: Flag when 2+ distinct P1 terms appear in the same text.
- P2: Flag only at high density (3+ distinct terms in text < 500 words,
      or 5+ distinct terms in longer text).

Context exemptions allow certain terms to be suppressed in specific
academic sections (e.g., "robust" in methods, "significant" in results).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Tier(Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


@dataclass
class VocabFinding:
    """A single vocabulary finding from the scanner."""

    term: str
    tier: Tier
    replacement: str | None
    position: int  # char offset of the match in the original text
    context_exempt: bool = False


# ---------------------------------------------------------------------------
# Vocabulary lists
# ---------------------------------------------------------------------------

# P0 entries: (pattern, replacement)
# Patterns may be plain words or short phrases.
_P0_ENTRIES: list[tuple[str, str]] = [
    # phrase-first so multi-word entries match before single-word fallback
    ("delve into", "explore, examine"),
    ("delve", "explore, examine"),
    ("in order to", "to"),
    ("due to the fact that", "because"),
    ("it's worth noting", "cut"),
    ("it is worth noting", "cut"),
    ("it's important to note", "cut"),
    ("it is important to note", "cut"),
    ("in today's world", "cut"),
    ("game-changer", "describe what changed"),
    ("game changer", "describe what changed"),
    ("watershed moment", "turning point"),
    ("deep dive", "examine"),
    ("ever-evolving", "changing"),
    ("ever evolving", "changing"),
    ("thought leader", "expert"),
    ("best practices", "what works"),
    ("plays a crucial role", "is important for"),
    ("plays a pivotal role", "is important for"),
    ("plays a vital role", "is important for"),
    ("harness the power", "use"),
    ("navigate the complexities", "address"),
    ("pave the way", "enable"),
    ("push the boundaries", "advance"),
    ("tapestry", "describe complexity"),
    ("utilize", "use"),
    ("leverage", "use"),
    ("leveraging", "using"),
    ("showcase", "show"),
    ("showcasing", "showing"),
    ("underscores", "highlights"),
    ("embark", "start"),
    ("serves as", "is"),
    ("testament to", "shows"),
    ("actionable", "practical"),
    ("impactful", "effective"),
    ("learnings", "lessons"),
    ("boasts", "has"),
    ("meticulous", "careful"),
    ("meticulously", "carefully"),
    ("seamless", "smooth"),
    ("seamlessly", "smoothly"),
    ("unravel", "explain"),
    ("plethora", "many"),
    ("myriad", "many"),
    ("paradigm", "model, framework"),
    ("nestled", "located"),
    ("bustling", "busy"),
    ("vibrant", "active"),
    ("beacon", "rewrite"),
    ("commence", "start"),
    ("ascertain", "determine"),
    ("endeavor", "effort"),
]

# P1 entries: plain terms (no replacements defined at tier level)
_P1_TERMS: list[str] = [
    "harness",
    "harnessing",
    "foster",
    "elevate",
    "streamline",
    "empower",
    "bolster",
    "spearhead",
    "resonate",
    "revolutionize",
    "facilitate",
    "cultivate",
    "illuminate",
    "elucidate",
    "galvanize",
    "augment",
    "catalyze",
    "reimagine",
    "encompass",
    "unleash",
    "navigate",
    "landscape",
    "cornerstone",
    "pivotal",
    "groundbreaking",
    "cutting-edge",
    "transformative",
    "synergy",
    "innovative",
    "novel",
    "realm",
    "intricacies",
    "paramount",
    "poised",
    "burgeoning",
    "nascent",
    "quintessential",
    "overarching",
    "robust",
    "comprehensive",
    "holistic",
    "multifaceted",
    "nuanced",
]

# P2 entries: plain terms
_P2_TERMS: list[str] = [
    "significant",
    "significantly",
    "innovative",
    "innovation",
    "effective",
    "effectively",
    "dynamic",
    "compelling",
    "unprecedented",
    "exceptional",
    "exceptionally",
    "remarkable",
    "remarkably",
    "sophisticated",
    "instrumental",
    "notable",
    "substantial",
    "considerably",
]

# Context exemptions: term -> set of contexts where it is exempt
_CONTEXT_EXEMPTIONS: dict[str, set[str]] = {
    "robust": {"methods", "results"},
    "significant": {"results"},
    "significantly": {"results"},
    "comprehensive": {"methods"},
    "facilitate": {"methods"},
}


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------

def _build_pattern(phrase: str) -> re.Pattern[str]:
    """Build a word-boundary-aware pattern for a phrase or word."""
    escaped = re.escape(phrase)
    # Use \b at word boundary positions; hyphens mean the boundary is implicit
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


# P0: list of (canonical_term, replacement, compiled_pattern)
_P0_COMPILED: list[tuple[str, str, re.Pattern[str]]] = [
    (phrase, replacement, _build_pattern(phrase))
    for phrase, replacement in _P0_ENTRIES
]

# P1: list of (term, compiled_pattern)
_P1_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (term, _build_pattern(term)) for term in _P1_TERMS
]

# P2: list of (term, compiled_pattern)
_P2_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (term, _build_pattern(term)) for term in _P2_TERMS
]


# ---------------------------------------------------------------------------
# Public scanner
# ---------------------------------------------------------------------------

def scan_vocabulary(text: str, context: str | None = None) -> list[VocabFinding]:
    """Scan *text* for AI-vocabulary anti-patterns and return findings.

    Parameters
    ----------
    text:
        The academic text to scan.
    context:
        Optional section label (e.g. ``"methods"``, ``"results"``,
        ``"discussion"``).  Used to apply context-based exemptions.

    Returns
    -------
    list[VocabFinding]
        Findings sorted by position.  Context-exempt findings are included
        with ``context_exempt=True`` but are present so callers can decide
        whether to surface them.
    """
    ctx = context.lower().strip() if context else None
    findings: list[VocabFinding] = []

    # --- P0: flag every occurrence ---
    # Track which character ranges are already claimed by a longer phrase match
    # so we don't double-flag overlapping single-word sub-patterns.
    claimed_spans: list[tuple[int, int]] = []

    for phrase, replacement, pattern in _P0_COMPILED:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            # Skip if this span is already covered by a longer phrase match
            if any(cs <= start and end <= ce for cs, ce in claimed_spans):
                continue
            claimed_spans.append((start, end))
            exempt = _is_exempt(phrase, ctx)
            findings.append(
                VocabFinding(
                    term=phrase,
                    tier=Tier.P0,
                    replacement=replacement,
                    position=start,
                    context_exempt=exempt,
                )
            )

    # --- P1: flag when 2+ distinct terms appear ---
    p1_hits: list[tuple[str, int]] = []  # (canonical_term, position)
    seen_p1_terms: set[str] = set()

    for term, pattern in _P1_COMPILED:
        for match in pattern.finditer(text):
            canonical = term.lower()
            if canonical not in seen_p1_terms:
                seen_p1_terms.add(canonical)
                p1_hits.append((term, match.start()))
            break  # one occurrence per term is enough to count it

    if len(seen_p1_terms) >= 2:
        for term, position in p1_hits:
            exempt = _is_exempt(term, ctx)
            findings.append(
                VocabFinding(
                    term=term,
                    tier=Tier.P1,
                    replacement=None,
                    position=position,
                    context_exempt=exempt,
                )
            )

    # --- P2: flag at high density ---
    word_count = len(text.split())
    threshold = 3 if word_count < 500 else 5

    p2_hits: list[tuple[str, int]] = []
    seen_p2_terms: set[str] = set()

    for term, pattern in _P2_COMPILED:
        for match in pattern.finditer(text):
            canonical = term.lower()
            if canonical not in seen_p2_terms:
                seen_p2_terms.add(canonical)
                p2_hits.append((term, match.start()))
            break  # one occurrence per term is enough to count it

    if len(seen_p2_terms) >= threshold:
        for term, position in p2_hits:
            exempt = _is_exempt(term, ctx)
            findings.append(
                VocabFinding(
                    term=term,
                    tier=Tier.P2,
                    replacement=None,
                    position=position,
                    context_exempt=exempt,
                )
            )

    findings.sort(key=lambda f: f.position)
    return findings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_exempt(term: str, context: str | None) -> bool:
    """Return True if *term* is exempt given the section *context*."""
    if context is None:
        return False
    exempt_contexts = _CONTEXT_EXEMPTIONS.get(term.lower(), set())
    return context in exempt_contexts
