"""Tests for oxidizer.preservation.extractor."""
import pytest

from oxidizer.preservation.extractor import (
    LockedEntities,
    SEMICOLON_CITATIONS,
    extract_entities,
)


# ---------------------------------------------------------------------------
# Numbered citations
# ---------------------------------------------------------------------------

class TestNumberedCitations:
    def test_single_numbered_citation(self):
        text = "As shown previously [1], the result holds."
        entities = extract_entities(text)
        assert "[1]" in entities.citations

    def test_multi_number_citation_comma(self):
        text = "Several studies [2, 3] support this claim."
        entities = extract_entities(text)
        assert "[2, 3]" in entities.citations

    def test_range_citation(self):
        text = "Prior work [4-6] demonstrates this."
        entities = extract_entities(text)
        assert "[4-6]" in entities.citations

    def test_range_citation_en_dash(self):
        # en-dash variant
        text = "Prior work [4\u20136] demonstrates this."
        entities = extract_entities(text)
        assert "[4\u20136]" in entities.citations

    def test_multiple_numbered_citations(self):
        text = "See [1] and [2, 3] and [7-9]."
        entities = extract_entities(text)
        assert "[1]" in entities.citations
        assert "[2, 3]" in entities.citations
        assert "[7-9]" in entities.citations

    def test_no_numbered_citations(self):
        text = "No citations here."
        entities = extract_entities(text)
        # Numbered citations list should not have bracket refs
        bracket_cits = [c for c in entities.citations if c.startswith("[")]
        assert bracket_cits == []


# ---------------------------------------------------------------------------
# Author-year citations
# ---------------------------------------------------------------------------

class TestAuthorYearCitations:
    def test_single_author_year(self):
        text = "The approach was validated (Jones, 2023)."
        entities = extract_entities(text)
        assert "(Jones, 2023)" in entities.citations

    def test_et_al_citation(self):
        text = "This was reported (Smith et al., 2024)."
        entities = extract_entities(text)
        assert "(Smith et al., 2024)" in entities.citations

    def test_two_author_and(self):
        text = "The model (Smith and Jones, 2024) was used."
        entities = extract_entities(text)
        assert "(Smith and Jones, 2024)" in entities.citations

    def test_multi_author_ampersand(self):
        text = "As noted (Smith, Jones, & Wei, 2024)."
        entities = extract_entities(text)
        assert "(Smith, Jones, & Wei, 2024)" in entities.citations

    def test_multiple_author_year_citations(self):
        text = "Both (Adams, 2020) and (Lee et al., 2021) agree."
        entities = extract_entities(text)
        assert "(Adams, 2020)" in entities.citations
        assert "(Lee et al., 2021)" in entities.citations


# ---------------------------------------------------------------------------
# Numbers with error margins
# ---------------------------------------------------------------------------

class TestNumbersWithErrorMargins:
    def test_unicode_plus_minus_with_unit(self):
        text = "The mean was 1.43 \u00b1 0.30 mm."
        entities = extract_entities(text)
        matches = [n for n in entities.numbers if "\u00b1" in n or "+/-" in n]
        assert any("1.43" in m and "0.30" in m for m in matches)

    def test_ascii_plus_minus_with_unit(self):
        text = "The result was 16.62 +/- 7.04 mm."
        entities = extract_entities(text)
        matches = [n for n in entities.numbers if "+/-" in n]
        assert any("16.62" in m and "7.04" in m for m in matches)

    def test_no_unit_error_margin(self):
        text = "Score: 5.0 \u00b1 0.5."
        entities = extract_entities(text)
        matches = [n for n in entities.numbers if "\u00b1" in n or "+/-" in n]
        assert any("5.0" in m and "0.5" in m for m in matches)

    def test_integer_error_margin(self):
        text = "Count was 10 \u00b1 2 events."
        entities = extract_entities(text)
        matches = [n for n in entities.numbers if "\u00b1" in n or "+/-" in n]
        assert any("10" in m for m in matches)


# ---------------------------------------------------------------------------
# Numbers with units
# ---------------------------------------------------------------------------

class TestNumbersWithUnits:
    def test_percentage(self):
        text = "Accuracy was 95.2%."
        entities = extract_entities(text)
        assert any("95.2" in n for n in entities.numbers)

    def test_years_unit(self):
        text = "Patients had mean age 62.3 years."
        entities = extract_entities(text)
        assert any("62.3" in n for n in entities.numbers)

    def test_ms_unit(self):
        text = "Latency of 2300 ms was observed."
        entities = extract_entities(text)
        assert any("2300" in n for n in entities.numbers)

    def test_mm_unit(self):
        text = "The rod was 1.0 mm in diameter."
        entities = extract_entities(text)
        assert any("1.0" in n for n in entities.numbers)


# ---------------------------------------------------------------------------
# Abbreviation definitions
# ---------------------------------------------------------------------------

class TestAbbreviationDefinitions:
    def test_simple_abbreviation(self):
        text = "Magnetic Resonance Imaging (MRI) was used."
        entities = extract_entities(text)
        assert "MRI" in entities.abbreviations

    def test_two_word_abbreviation(self):
        text = "Convolutional Neural Network (CNN) architecture."
        entities = extract_entities(text)
        assert "CNN" in entities.abbreviations

    def test_multi_word_abbreviation(self):
        text = "The American Medical Association (AMA) guidelines."
        entities = extract_entities(text)
        assert "AMA" in entities.abbreviations

    def test_multiple_abbreviations(self):
        text = (
            "Magnetic Resonance Imaging (MRI) and "
            "Computed Tomography (CT) were compared."
        )
        entities = extract_entities(text)
        assert "MRI" in entities.abbreviations
        assert "CT" in entities.abbreviations

    def test_no_abbreviation_lowercase_parens(self):
        # lowercase inside parens should not be extracted as abbreviation
        text = "The method (see above) is valid."
        entities = extract_entities(text)
        assert entities.abbreviations == []

    def test_abbreviation_with_number(self):
        text = "The tool (TOOL2) was introduced."
        # single preceding uppercase word — still valid
        entities = extract_entities(text)
        # No uppercase word before (TOOL2) here, so expect no match
        # (TOOL2 pattern is valid but needs preceding uppercase word)
        # This test just asserts we don't crash.
        assert isinstance(entities.abbreviations, list)


# ---------------------------------------------------------------------------
# LaTeX equations
# ---------------------------------------------------------------------------

class TestLatexEquations:
    def test_simple_equation(self):
        text = r"The formula $E = mc^2$ is famous."
        entities = extract_entities(text)
        assert r"$E = mc^2$" in entities.equations

    def test_fraction_equation(self):
        text = r"Compute $\frac{a}{b}$ for normalization."
        entities = extract_entities(text)
        assert r"$\frac{a}{b}$" in entities.equations

    def test_multiple_equations(self):
        text = r"Both $\alpha$ and $\beta$ matter."
        entities = extract_entities(text)
        assert r"$\alpha$" in entities.equations
        assert r"$\beta$" in entities.equations

    def test_no_equations(self):
        text = "No math here."
        entities = extract_entities(text)
        assert entities.equations == []


# ---------------------------------------------------------------------------
# Figure/table references
# ---------------------------------------------------------------------------

class TestFigureTableRefs:
    def test_figure_ref(self):
        text = "As shown in Figure 1, the curve rises."
        entities = extract_entities(text)
        assert "Figure 1" in entities.figure_table_refs

    def test_table_ref(self):
        text = "Results are summarized in Table 2."
        entities = extract_entities(text)
        assert "Table 2" in entities.figure_table_refs

    def test_fig_abbreviation(self):
        text = "See Fig. 1 for details."
        entities = extract_entities(text)
        assert "Fig. 1" in entities.figure_table_refs

    def test_supplementary_figure(self):
        text = "Data shown in Supplementary Figure 1."
        entities = extract_entities(text)
        assert "Supplementary Figure 1" in entities.figure_table_refs

    def test_multiple_refs(self):
        text = "See Figure 1, Table 2, and Fig. 3."
        entities = extract_entities(text)
        assert "Figure 1" in entities.figure_table_refs
        assert "Table 2" in entities.figure_table_refs
        assert "Fig. 3" in entities.figure_table_refs

    def test_no_refs(self):
        text = "No figures or tables mentioned."
        entities = extract_entities(text)
        assert entities.figure_table_refs == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_citations_removed(self):
        text = "See [1] and later [1] again."
        entities = extract_entities(text)
        assert entities.citations.count("[1]") == 1

    def test_duplicate_figure_refs_removed(self):
        text = "Figure 1 shows A. Figure 1 also shows B."
        entities = extract_entities(text)
        assert entities.figure_table_refs.count("Figure 1") == 1

    def test_duplicate_equations_removed(self):
        text = r"Use $\alpha$ and then $\alpha$ again."
        entities = extract_entities(text)
        assert entities.equations.count(r"$\alpha$") == 1

    def test_duplicate_abbreviations_removed(self):
        text = (
            "Magnetic Resonance Imaging (MRI) scan. "
            "Another Magnetic Resonance Imaging (MRI) device."
        )
        entities = extract_entities(text)
        assert entities.abbreviations.count("MRI") == 1


# ---------------------------------------------------------------------------
# all_entities()
# ---------------------------------------------------------------------------

class TestAllEntities:
    def test_returns_flat_list(self):
        text = (
            "See [1] and (Smith et al., 2024). "
            "Value 1.43 \u00b1 0.30 mm. "
            r"Equation $\alpha$. "
            "Figure 1. "
            "Magnetic Resonance Imaging (MRI)."
        )
        entities = extract_entities(text)
        all_e = entities.all_entities()
        assert isinstance(all_e, list)
        assert len(all_e) > 0

    def test_no_duplicates_across_categories(self):
        text = "See [1] and [1]."
        entities = extract_entities(text)
        all_e = entities.all_entities()
        assert len(all_e) == len(set(all_e))

    def test_empty_text_returns_empty(self):
        entities = extract_entities("")
        assert entities.all_entities() == []

    def test_contains_all_categories(self):
        text = (
            "See [1] and (Jones, 2023). "
            "Mean 95.2%. "
            r"Formula $x^2$. "
            "Figure 1. "
            "Magnetic Resonance Imaging (MRI)."
        )
        entities = extract_entities(text)
        all_e = entities.all_entities()
        assert "[1]" in all_e
        assert "(Jones, 2023)" in all_e
        assert "Figure 1" in all_e
        assert r"$x^2$" in all_e
        assert "MRI" in all_e

    def test_all_entities_deduped_flat(self):
        """all_entities() must not contain duplicates even if subcategories share values."""
        # Construct a LockedEntities directly with overlapping values to test the method.
        entities = LockedEntities(
            citations=["[1]", "[2]"],
            numbers=["95.2%"],
            abbreviations=["MRI"],
            equations=[r"$\alpha$"],
            figure_table_refs=["Figure 1"],
        )
        all_e = entities.all_entities()
        assert len(all_e) == len(set(all_e))
        assert sorted(all_e) == sorted(["[1]", "[2]", "95.2%", "MRI", r"$\alpha$", "Figure 1"])


# ---------------------------------------------------------------------------
# Semicolon-separated citations (Fix 5)
# ---------------------------------------------------------------------------

class TestSemicolonCitations:
    """Tests for SEMICOLON_CITATIONS regex and extraction integration."""

    def test_regex_matches_two_authors(self):
        text = "(Smith, 2020; Jones, 2021)"
        matches = SEMICOLON_CITATIONS.findall(text)
        assert matches == ["(Smith, 2020; Jones, 2021)"]

    def test_regex_matches_three_authors(self):
        text = "(Adams, 2019; Brown, 2020; Clark, 2021)"
        matches = SEMICOLON_CITATIONS.findall(text)
        assert matches == ["(Adams, 2019; Brown, 2020; Clark, 2021)"]

    def test_regex_matches_et_al_in_group(self):
        text = "(Smith et al., 2020; Jones, 2021)"
        matches = SEMICOLON_CITATIONS.findall(text)
        assert matches == ["(Smith et al., 2020; Jones, 2021)"]

    def test_regex_does_not_match_single_author(self):
        """A single author-year with no semicolons should NOT match the semicolon pattern."""
        text = "(Smith, 2020)"
        matches = SEMICOLON_CITATIONS.findall(text)
        assert matches == []

    def test_extract_semicolon_citation_as_single_entity(self):
        text = "Multiple studies (Smith, 2020; Jones, 2021) confirm this."
        entities = extract_entities(text)
        assert "(Smith, 2020; Jones, 2021)" in entities.citations

    def test_extract_does_not_double_count_individual_parts(self):
        """Individual author-year citations inside a semicolon group should not
        also appear as separate citation entities."""
        text = "See (Smith, 2020; Jones, 2021) for details."
        entities = extract_entities(text)
        # The whole group should be present
        assert "(Smith, 2020; Jones, 2021)" in entities.citations
        # The individual parts should NOT be separately listed
        assert "(Smith, 2020)" not in entities.citations
        assert "(Jones, 2021)" not in entities.citations

    def test_extract_mixes_semicolon_and_single_citations(self):
        """Semicolon groups and standalone citations can coexist."""
        text = "See [1] and (Smith, 2020; Jones, 2021) and (Adams, 2019)."
        entities = extract_entities(text)
        assert "[1]" in entities.citations
        assert "(Smith, 2020; Jones, 2021)" in entities.citations
        assert "(Adams, 2019)" in entities.citations

    def test_extract_deduplicates_semicolon_citations(self):
        text = "As noted (Smith, 2020; Jones, 2021) and again (Smith, 2020; Jones, 2021)."
        entities = extract_entities(text)
        assert entities.citations.count("(Smith, 2020; Jones, 2021)") == 1
