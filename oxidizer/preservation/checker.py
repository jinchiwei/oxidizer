"""Preservation checker for Oxidizer.

Verifies that all locked entities extracted from the original text are present
as verbatim substrings in the restyled output text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from oxidizer.preservation.extractor import LockedEntities


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

    missing: list[str] = [e for e in all_e if e not in output_text]
    preserved_count = total - len(missing)

    return PreservationResult(
        passed=len(missing) == 0,
        preserved_count=preserved_count,
        total_count=total,
        missing=missing,
    )
