"""Entity extractor for Oxidizer preservation system.

Extracts "lockable" entities from academic text that must be preserved
verbatim when text is restyled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Numbered citations: [1], [2, 3], [4-6]
_RE_NUMBERED_CITATION = re.compile(
    r"\[\d+(?:\s*[-\u2013,]\s*\d+)*\]"
)

# Author-year citations in parentheses.
# Handles:
#   (Smith et al., 2024)
#   (Jones, 2023)
#   (Smith and Jones, 2024)
#   (Smith, Jones, & Wei, 2024)
# Uses ordered alternation so the most-specific forms are tried first.
_RE_AUTHOR_YEAR_CITATION = re.compile(
    r"\("
    r"(?:"
        r"[A-Z][A-Za-z\u2019'\-]+\s+et\s+al\."                     # et al. form
        r"|[A-Z][A-Za-z\u2019'\-]+\s+and\s+[A-Z][A-Za-z\u2019'\-]+"  # two-author "and" form
        r"|[A-Z][A-Za-z\u2019'\-]+(?:,\s+[A-Z][A-Za-z\u2019'\-]+)*(?:,?\s*&\s*[A-Z][A-Za-z\u2019'\-]+)?"  # comma/& list
        r"|[A-Z][A-Za-z\u2019'\-]+"                                 # single author
    r")"
    r",\s*\d{4}"
    r"\)"
)

# Numbers with error margins: 1.43 ± 0.30 mm  /  16.62 +/- 7.04 mm
# The unit suffix is optional.
_RE_NUMBER_WITH_ERROR = re.compile(
    r"\d+(?:\.\d+)?\s*(?:\u00b1|\+/-)\s*\d+(?:\.\d+)?(?:\s*[A-Za-z%]+)?"
)

# Numbers with units: 62.3 years, 95.2%, 2300 ms, 1.0 mm
# Two sub-cases:
#   - percent: \b<number>%  (% is not a word char so trailing \b would fail)
#   - alpha unit: \b<number> <unit-word>\b
_RE_NUMBER_WITH_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*%|\s+[A-Za-z]{1,10}\b)"
)

# Abbreviation definitions: Full Name (ABBR)
# Capture just the ABBR part (all-caps, 2-10 chars).
_RE_ABBREVIATION = re.compile(
    r"\b[A-Z][A-Za-z\-]+(?:\s+[A-Za-z\-]+)*\s+\(([A-Z][A-Z0-9\-]{1,9})\)"
)

# LaTeX inline equations: $...$
# Non-greedy, does not cross newlines.
_RE_LATEX_EQUATION = re.compile(r"\$[^\$\n]+?\$")

# Figure/table references
_RE_FIGURE_TABLE_REF = re.compile(
    r"\b(?:Supplementary\s+)?(?:Figure|Table|Fig\.)\s+\d+\b"
)


# ---------------------------------------------------------------------------
# LockedEntities dataclass
# ---------------------------------------------------------------------------

@dataclass
class LockedEntities:
    """Container for all lockable entities extracted from a text."""

    citations: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    abbreviations: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    figure_table_refs: list[str] = field(default_factory=list)

    def all_entities(self) -> list[str]:
        """Return a flat, deduplicated list of all locked entities."""
        seen: set[str] = set()
        result: list[str] = []
        for item in (
            self.citations
            + self.numbers
            + self.abbreviations
            + self.equations
            + self.figure_table_refs
        ):
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _dedupe(items: list[str]) -> list[str]:
    """Return items with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _extract_citations(text: str) -> list[str]:
    numbered = _RE_NUMBERED_CITATION.findall(text)
    author_year = _RE_AUTHOR_YEAR_CITATION.findall(text)
    return _dedupe(numbered + author_year)


def _extract_numbers(text: str) -> list[str]:
    """Extract numbers with error margins and numbers with units.

    Numbers with error margins are extracted first (they are more specific).
    Then numbers with units are extracted from the remaining text to avoid
    double-counting substrings.
    """
    results: list[str] = []
    seen: set[str] = set()

    # Error-margin numbers (higher priority / more specific)
    for m in _RE_NUMBER_WITH_ERROR.finditer(text):
        val = m.group(0).strip()
        if val not in seen:
            seen.add(val)
            results.append(val)

    # Numbers with units — skip if already captured by error-margin pattern.
    # Build a mask of already-covered spans.
    covered_spans: list[tuple[int, int]] = [
        m.span() for m in _RE_NUMBER_WITH_ERROR.finditer(text)
    ]

    for m in _RE_NUMBER_WITH_UNIT.finditer(text):
        # Skip if this match overlaps a covered span.
        start, end = m.span()
        overlaps = any(cs <= start < ce or cs < end <= ce for cs, ce in covered_spans)
        if overlaps:
            continue
        val = m.group(0).strip()
        if val not in seen:
            seen.add(val)
            results.append(val)

    return results


def _extract_abbreviations(text: str) -> list[str]:
    """Extract abbreviations from 'Full Name (ABBR)' patterns."""
    abbrs = [m.group(1) for m in _RE_ABBREVIATION.finditer(text)]
    return _dedupe(abbrs)


def _extract_equations(text: str) -> list[str]:
    return _dedupe(_RE_LATEX_EQUATION.findall(text))


def _extract_figure_table_refs(text: str) -> list[str]:
    return _dedupe(_RE_FIGURE_TABLE_REF.findall(text))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> LockedEntities:
    """Extract all lockable entities from *text*.

    Returns a :class:`LockedEntities` instance with each category
    deduplicated.
    """
    return LockedEntities(
        citations=_extract_citations(text),
        numbers=_extract_numbers(text),
        abbreviations=_extract_abbreviations(text),
        equations=_extract_equations(text),
        figure_table_refs=_extract_figure_table_refs(text),
    )
