"""Preservation checker for Oxidizer.

Verifies that all locked entities extracted from the original text are present
as verbatim substrings in the restyled output text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from oxidizer.preservation.extractor import LockedEntities


def _entity_present(entity: str, text: str) -> bool:
    """Check if an entity is present in text, using word boundaries for short entities."""
    if len(entity) < 20:
        # Use word boundaries to avoid partial matches like "Fig. 1" in "Fig. 10"
        pattern = re.escape(entity)
        # Add word boundary after the entity if it ends with a word character or digit
        if entity[-1].isalnum():
            pattern += r'\b'
        return bool(re.search(pattern, text))
    return entity in text


@dataclass
class PreservationResult:
    """Result of a preservation check."""

    passed: bool
    preserved_count: int
    total_count: int
    missing: list[str] = field(default_factory=list)


def check_entity_preservation(
    entities: LockedEntities,
    output_text: str,
) -> PreservationResult:
    """Check that every locked entity appears verbatim in *output_text*.

    Parameters
    ----------
    entities:
        The :class:`~oxidizer.preservation.extractor.LockedEntities` extracted
        from the original text.
    output_text:
        The restyled text to check against.

    Returns
    -------
    PreservationResult
        ``passed`` is ``True`` iff every entity in ``entities.all_entities()``
        appears as a substring of ``output_text``.
    """
    all_e = entities.all_entities()
    total = len(all_e)

    if total == 0:
        return PreservationResult(
            passed=True,
            preserved_count=0,
            total_count=0,
            missing=[],
        )

    missing: list[str] = [e for e in all_e if not _entity_present(e, output_text)]
    preserved_count = total - len(missing)

    return PreservationResult(
        passed=len(missing) == 0,
        preserved_count=preserved_count,
        total_count=total,
        missing=missing,
    )
