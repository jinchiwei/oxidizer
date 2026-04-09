"""Shared pipeline orchestration for revise and write engines."""
from dataclasses import dataclass, field

from oxidizer.parsers.markdown_parser import Section
from oxidizer.preservation.extractor import extract_entities, LockedEntities
from oxidizer.preservation.checker import check_entity_preservation, PreservationResult
from oxidizer.profiles.schema import StyleProfile
from oxidizer.scoring.reporter import compute_style_report, StyleReport


@dataclass
class PipelineResult:
    text: str
    original_text: str
    heading: str
    entities: LockedEntities
    preservation: PreservationResult | None = None
    style_report: StyleReport | None = None
    retries: int = 0
    warnings: list[str] = field(default_factory=list)


def extract_and_lock(text: str) -> tuple[LockedEntities, list[str]]:
    """Extract entities from text and return them as a lockable list."""
    entities = extract_entities(text)
    return entities, entities.all_entities()


def verify_and_score(
    restyled_text: str,
    entities: LockedEntities,
    profile: StyleProfile,
) -> tuple[PreservationResult, StyleReport, list[str]]:
    """Verify entity preservation and compute style score."""
    preservation = check_entity_preservation(entities, restyled_text)
    style_report = compute_style_report(restyled_text, profile)
    warnings = []
    if not preservation.passed:
        warnings.append(f"Entities missing: {preservation.missing}")
    if style_report.banned_words_found:
        warnings.append(f"Banned AI-isms found: {style_report.banned_words_found}")
    return preservation, style_report, warnings
