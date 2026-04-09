"""Tests for oxidizer.preservation.checker."""
import pytest

from oxidizer.preservation.extractor import LockedEntities, extract_entities
from oxidizer.preservation.checker import (
    PreservationResult,
    check_entity_preservation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entities(**kwargs) -> LockedEntities:
    """Construct a LockedEntities with only the given fields populated."""
    return LockedEntities(
        citations=kwargs.get("citations", []),
        numbers=kwargs.get("numbers", []),
        abbreviations=kwargs.get("abbreviations", []),
        equations=kwargs.get("equations", []),
        figure_table_refs=kwargs.get("figure_table_refs", []),
    )


# ---------------------------------------------------------------------------
# All entities preserved → passed=True
# ---------------------------------------------------------------------------

class TestAllPreserved:
    def test_single_citation_preserved(self):
        entities = _entities(citations=["[1]"])
        output = "As shown [1], the result holds."
        result = check_entity_preservation(entities, output)
        assert result.passed is True
        assert result.missing == []
        assert result.preserved_count == 1
        assert result.total_count == 1

    def test_multiple_citations_all_preserved(self):
        entities = _entities(citations=["[1]", "[2, 3]", "(Smith et al., 2024)"])
        output = "See [1], [2, 3], and (Smith et al., 2024)."
        result = check_entity_preservation(entities, output)
        assert result.passed is True
        assert result.preserved_count == 3
        assert result.total_count == 3

    def test_numbers_preserved(self):
        entities = _entities(numbers=["1.43 \u00b1 0.30 mm", "95.2%"])
        output = "Mean: 1.43 \u00b1 0.30 mm. Accuracy: 95.2%."
        result = check_entity_preservation(entities, output)
        assert result.passed is True
        assert result.missing == []

    def test_figure_table_refs_preserved(self):
        entities = _entities(figure_table_refs=["Figure 1", "Table 2"])
        output = "Results are in Figure 1 and Table 2."
        result = check_entity_preservation(entities, output)
        assert result.passed is True

    def test_equations_preserved(self):
        entities = _entities(equations=[r"$E = mc^2$"])
        output = r"As per $E = mc^2$, energy is conserved."
        result = check_entity_preservation(entities, output)
        assert result.passed is True

    def test_abbreviations_preserved(self):
        entities = _entities(abbreviations=["MRI", "CT"])
        output = "Both MRI and CT were used."
        result = check_entity_preservation(entities, output)
        assert result.passed is True

    def test_all_categories_all_preserved(self):
        entities = _entities(
            citations=["[1]", "(Jones, 2023)"],
            numbers=["95.2%"],
            abbreviations=["MRI"],
            equations=[r"$\alpha$"],
            figure_table_refs=["Figure 1"],
        )
        output = (
            "See [1] and (Jones, 2023). "
            "Accuracy 95.2%. "
            r"Formula $\alpha$. "
            "Figure 1 shows MRI results."
        )
        result = check_entity_preservation(entities, output)
        assert result.passed is True
        assert result.total_count == 6
        assert result.preserved_count == 6
        assert result.missing == []


# ---------------------------------------------------------------------------
# Missing citation detected → passed=False
# ---------------------------------------------------------------------------

class TestMissingCitation:
    def test_missing_numbered_citation(self):
        entities = _entities(citations=["[1]", "[2]"])
        output = "As shown [1], the claim holds."  # [2] is missing
        result = check_entity_preservation(entities, output)
        assert result.passed is False
        assert "[2]" in result.missing
        assert "[1]" not in result.missing
        assert result.preserved_count == 1
        assert result.total_count == 2

    def test_missing_author_year_citation(self):
        entities = _entities(citations=["(Smith et al., 2024)", "(Jones, 2023)"])
        output = "As noted (Smith et al., 2024)."
        result = check_entity_preservation(entities, output)
        assert result.passed is False
        assert "(Jones, 2023)" in result.missing

    def test_all_citations_missing(self):
        entities = _entities(citations=["[1]", "[2]", "[3]"])
        output = "No citations retained in this output."
        result = check_entity_preservation(entities, output)
        assert result.passed is False
        assert result.preserved_count == 0
        assert result.total_count == 3
        assert len(result.missing) == 3


# ---------------------------------------------------------------------------
# Missing number detected
# ---------------------------------------------------------------------------

class TestMissingNumber:
    def test_missing_error_margin_number(self):
        entities = _entities(numbers=["1.43 \u00b1 0.30 mm", "95.2%"])
        output = "Accuracy was 95.2%."  # error-margin number missing
        result = check_entity_preservation(entities, output)
        assert result.passed is False
        assert "1.43 \u00b1 0.30 mm" in result.missing
        assert "95.2%" not in result.missing

    def test_missing_percentage(self):
        entities = _entities(numbers=["95.2%"])
        output = "The method performed well."
        result = check_entity_preservation(entities, output)
        assert result.passed is False
        assert "95.2%" in result.missing

    def test_partial_number_not_sufficient(self):
        # "95" appears but not "95.2%" exactly
        entities = _entities(numbers=["95.2%"])
        output = "About 95 percent was retained."
        result = check_entity_preservation(entities, output)
        assert result.passed is False
        assert "95.2%" in result.missing


# ---------------------------------------------------------------------------
# Empty entities → always passes
# ---------------------------------------------------------------------------

class TestEmptyEntities:
    def test_all_empty_lists(self):
        entities = LockedEntities()
        result = check_entity_preservation(entities, "Any output text.")
        assert result.passed is True
        assert result.total_count == 0
        assert result.preserved_count == 0
        assert result.missing == []

    def test_empty_with_empty_output(self):
        entities = LockedEntities()
        result = check_entity_preservation(entities, "")
        assert result.passed is True

    def test_empty_text_extraction_always_passes(self):
        entities = extract_entities("")
        result = check_entity_preservation(entities, "Some output.")
        assert result.passed is True


# ---------------------------------------------------------------------------
# PreservationResult dataclass
# ---------------------------------------------------------------------------

class TestPreservationResultDataclass:
    def test_fields_exist(self):
        r = PreservationResult(
            passed=True,
            preserved_count=3,
            total_count=3,
            missing=[],
        )
        assert r.passed is True
        assert r.preserved_count == 3
        assert r.total_count == 3
        assert r.missing == []

    def test_failed_result(self):
        r = PreservationResult(
            passed=False,
            preserved_count=1,
            total_count=3,
            missing=["[2]", "[3]"],
        )
        assert r.passed is False
        assert r.preserved_count == 1
        assert r.total_count == 3
        assert len(r.missing) == 2

    def test_missing_defaults_to_empty_list(self):
        r = PreservationResult(passed=True, preserved_count=0, total_count=0)
        assert r.missing == []


# ---------------------------------------------------------------------------
# Round-trip: extract from source, check in output
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_extract_and_check_full_preservation(self):
        source = (
            "As shown in Figure 1 [1], the mean was 1.43 \u00b1 0.30 mm. "
            "Magnetic Resonance Imaging (MRI) was used (Smith et al., 2024)."
        )
        output = (
            "Figure 1 [1] illustrates the mean of 1.43 \u00b1 0.30 mm. "
            "MRI was the modality employed (Smith et al., 2024)."
        )
        entities = extract_entities(source)
        result = check_entity_preservation(entities, output)
        assert result.passed is True

    def test_extract_and_check_with_missing(self):
        source = "See Figure 1 and Table 2. Values [1] and [2]."
        output = "Figure 1 shows the data. Values [1] are listed."
        entities = extract_entities(source)
        result = check_entity_preservation(entities, output)
        assert result.passed is False
        assert "Table 2" in result.missing
        assert "[2]" in result.missing
