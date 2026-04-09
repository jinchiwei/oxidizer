"""Style diff computation for Oxidizer.

Compares two texts using difflib.SequenceMatcher and annotates changes
with detected style rule violations (passive voice, banned words, etc.).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from oxidizer.scoring.metrics import count_banned_words, count_contractions

# Common passive voice indicators (auxiliary + past participle patterns)
_PASSIVE_INDICATORS = re.compile(
    r"\b(?:is|are|was|were|been|being|be)\s+\w+ed\b",
    re.IGNORECASE,
)

_CONTRACTION_RE = re.compile(
    r"\b(?:can't|cannot|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't"
    r"|haven't|hasn't|hadn't|wouldn't|couldn't|shouldn't|it's|they're|we're"
    r"|you're|I'm|he's|she's|that's|there's|let's)\b",
    re.IGNORECASE,
)


@dataclass
class DiffChange:
    """Represents a single changed line between original and revised texts."""

    line_num: int
    original: str
    revised: str
    annotation: str  # e.g., "passive → active", "banned word removed"


@dataclass
class DiffResult:
    """Full diff result between two texts, with optional style reports."""

    changes: list[DiffChange] = field(default_factory=list)
    original_report: dict | None = None
    revised_report: dict | None = None


def _detect_passive(text: str) -> bool:
    """Return True if text contains passive voice indicators."""
    return bool(_PASSIVE_INDICATORS.search(text))


def _detect_contractions(text: str) -> bool:
    """Return True if text contains contractions."""
    return bool(_CONTRACTION_RE.search(text))


def _annotate_change(original: str, revised: str, banned_list: list[str] | None = None) -> str:
    """Infer the style rule that drove a change between original and revised lines.

    Args:
        original: The original line text.
        revised: The revised line text.
        banned_list: Optional list of banned words/phrases to check removal of.

    Returns:
        A human-readable annotation string.
    """
    annotations: list[str] = []

    # Check passive → active
    had_passive = _detect_passive(original)
    has_passive = _detect_passive(revised)
    if had_passive and not has_passive:
        annotations.append("passive → active")
    elif not had_passive and has_passive:
        annotations.append("active → passive")

    # Check contraction removal
    had_contractions = _detect_contractions(original)
    has_contractions = _detect_contractions(revised)
    if had_contractions and not has_contractions:
        annotations.append("contraction removed")
    elif not had_contractions and has_contractions:
        annotations.append("contraction added")

    # Check banned word removal
    if banned_list:
        orig_banned = count_banned_words(original, banned_list)
        rev_banned = count_banned_words(revised, banned_list)
        if orig_banned and not rev_banned:
            words = ", ".join(f'"{w}"' for w in orig_banned)
            annotations.append(f"banned word removed ({words})")
        elif not orig_banned and rev_banned:
            words = ", ".join(f'"{w}"' for w in rev_banned)
            annotations.append(f"banned word added ({words})")

    # Generic style change fallback
    if not annotations:
        orig_len = len(original.split())
        rev_len = len(revised.split())
        if orig_len > rev_len + 5:
            annotations.append("sentence shortened")
        elif rev_len > orig_len + 5:
            annotations.append("sentence expanded")
        else:
            annotations.append("style change")

    return "; ".join(annotations)


def compute_diff(
    original_text: str,
    revised_text: str,
    banned_list: list[str] | None = None,
    original_report: dict | None = None,
    revised_report: dict | None = None,
) -> DiffResult:
    """Compute a style-annotated diff between two texts.

    Uses difflib.SequenceMatcher to find changed regions. Each change is
    annotated with an inferred style rule (passive→active, banned word removed,
    contraction removed, etc.).

    Args:
        original_text: The original document text.
        revised_text: The revised document text.
        banned_list: Optional list of banned words for annotation.
        original_report: Optional pre-computed style report dict for original.
        revised_report: Optional pre-computed style report dict for revised.

    Returns:
        A DiffResult with annotated DiffChange entries and optional reports.
    """
    original_lines = original_text.splitlines()
    revised_lines = revised_text.splitlines()

    matcher = difflib.SequenceMatcher(
        None, original_lines, revised_lines, autojunk=False
    )

    changes: list[DiffChange] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        if tag == "replace":
            # Pair up changed lines
            orig_chunk = original_lines[i1:i2]
            rev_chunk = revised_lines[j1:j2]
            max_len = max(len(orig_chunk), len(rev_chunk))
            for k in range(max_len):
                orig_line = orig_chunk[k] if k < len(orig_chunk) else ""
                rev_line = rev_chunk[k] if k < len(rev_chunk) else ""
                annotation = _annotate_change(orig_line, rev_line, banned_list)
                changes.append(
                    DiffChange(
                        line_num=i1 + k + 1,
                        original=orig_line,
                        revised=rev_line,
                        annotation=annotation,
                    )
                )

        elif tag == "delete":
            for k, orig_line in enumerate(original_lines[i1:i2]):
                changes.append(
                    DiffChange(
                        line_num=i1 + k + 1,
                        original=orig_line,
                        revised="",
                        annotation="line removed",
                    )
                )

        elif tag == "insert":
            for k, rev_line in enumerate(revised_lines[j1:j2]):
                changes.append(
                    DiffChange(
                        line_num=i1 + 1,
                        original="",
                        revised=rev_line,
                        annotation="line added",
                    )
                )

    return DiffResult(
        changes=changes,
        original_report=original_report,
        revised_report=revised_report,
    )
