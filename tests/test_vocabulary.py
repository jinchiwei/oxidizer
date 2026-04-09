"""Tests for oxidizer.detection.vocabulary — tiered AI vocabulary scanner."""
import pytest

from oxidizer.detection.vocabulary import (
    Tier,
    VocabFinding,
    scan_vocabulary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _terms(findings: list[VocabFinding]) -> set[str]:
    return {f.term for f in findings}


def _non_exempt(findings: list[VocabFinding]) -> list[VocabFinding]:
    return [f for f in findings if not f.context_exempt]


# ---------------------------------------------------------------------------
# P0: always flag on a single occurrence
# ---------------------------------------------------------------------------

class TestP0AlwaysFlag:
    def test_delve(self):
        findings = scan_vocabulary("We delve into the data.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert len(p0) >= 1
        assert any("delve" in f.term for f in p0)

    def test_tapestry(self):
        findings = scan_vocabulary("A rich tapestry of methods.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("tapestry" in f.term for f in p0)

    def test_leverage(self):
        findings = scan_vocabulary("We leverage machine learning.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("leverage" in f.term for f in p0)

    def test_utilize(self):
        findings = scan_vocabulary("We utilize a novel approach.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("utilize" in f.term for f in p0)

    def test_serves_as(self):
        findings = scan_vocabulary("This serves as a proxy.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("serves as" in f.term for f in p0)

    def test_testament_to(self):
        findings = scan_vocabulary("This is a testament to the effort.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("testament to" in f.term for f in p0)

    def test_in_order_to(self):
        findings = scan_vocabulary("In order to proceed, we ran the model.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("in order to" in f.term for f in p0)

    def test_due_to_the_fact_that(self):
        findings = scan_vocabulary("Due to the fact that results varied, we repeated.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("due to the fact that" in f.term for f in p0)

    def test_meticulous(self):
        findings = scan_vocabulary("The meticulous process took weeks.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("meticulous" in f.term for f in p0)

    def test_paradigm(self):
        findings = scan_vocabulary("This shifts the paradigm.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("paradigm" in f.term for f in p0)

    def test_plethora(self):
        findings = scan_vocabulary("A plethora of studies confirm this.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("plethora" in f.term for f in p0)


# ---------------------------------------------------------------------------
# P0: provides replacement suggestions
# ---------------------------------------------------------------------------

class TestP0Replacements:
    def test_delve_has_replacement(self):
        findings = scan_vocabulary("We delve into the data.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any(f.replacement is not None for f in p0)

    def test_utilize_replacement(self):
        findings = scan_vocabulary("We utilize this.")
        p0 = [f for f in findings if f.tier == Tier.P0 and "utilize" in f.term]
        assert p0 and p0[0].replacement == "use"

    def test_in_order_to_replacement(self):
        findings = scan_vocabulary("In order to test this we ran it.")
        p0 = [f for f in findings if f.tier == Tier.P0 and "in order to" in f.term]
        assert p0 and p0[0].replacement == "to"

    def test_leverage_replacement(self):
        findings = scan_vocabulary("We leverage the model.")
        p0 = [f for f in findings if f.tier == Tier.P0 and "leverage" == f.term]
        assert p0 and p0[0].replacement == "use"

    def test_tapestry_replacement(self):
        findings = scan_vocabulary("A tapestry of data.")
        p0 = [f for f in findings if f.tier == Tier.P0 and "tapestry" in f.term]
        assert p0 and p0[0].replacement == "describe complexity"

    def test_due_to_the_fact_replacement(self):
        findings = scan_vocabulary("Due to the fact that this works.")
        p0 = [f for f in findings if f.tier == Tier.P0 and "due to the fact that" in f.term]
        assert p0 and p0[0].replacement == "because"


# ---------------------------------------------------------------------------
# P1: single occurrence NOT flagged
# ---------------------------------------------------------------------------

class TestP1SingleNotFlagged:
    def test_single_robust(self):
        findings = scan_vocabulary("The model is robust.")
        p1 = [f for f in findings if f.tier == Tier.P1]
        assert len(p1) == 0

    def test_single_novel(self):
        findings = scan_vocabulary("We propose a novel approach.")
        p1 = [f for f in findings if f.tier == Tier.P1]
        assert len(p1) == 0

    def test_single_comprehensive(self):
        findings = scan_vocabulary("A comprehensive review was performed.")
        p1 = [f for f in findings if f.tier == Tier.P1]
        assert len(p1) == 0

    def test_single_pivotal(self):
        findings = scan_vocabulary("This was pivotal to the outcome.")
        p1 = [f for f in findings if f.tier == Tier.P1]
        assert len(p1) == 0


# ---------------------------------------------------------------------------
# P1: two distinct terms ARE flagged
# ---------------------------------------------------------------------------

class TestP1TwoDistinct:
    def test_two_terms_flagged(self):
        findings = scan_vocabulary("The robust and comprehensive analysis.")
        p1 = [f for f in findings if f.tier == Tier.P1]
        assert len(p1) == 2

    def test_two_different_terms(self):
        findings = scan_vocabulary("The novel landscape requires a paradigm.")
        # "paradigm" is P0; only P1 terms matter here
        p1 = [f for f in findings if f.tier == Tier.P1]
        found_terms = {f.term for f in p1}
        # "novel" and "landscape" are both P1
        assert "novel" in found_terms
        assert "landscape" in found_terms

    def test_harness_and_foster(self):
        findings = scan_vocabulary("We harness data to foster collaboration.")
        p1 = [f for f in findings if f.tier == Tier.P1]
        found_terms = {f.term for f in p1}
        assert "harness" in found_terms
        assert "foster" in found_terms


# ---------------------------------------------------------------------------
# P1: three distinct terms, all flagged
# ---------------------------------------------------------------------------

class TestP1ThreeDistinct:
    def test_three_terms_all_flagged(self):
        findings = scan_vocabulary(
            "The robust, comprehensive, and holistic approach fosters progress."
        )
        p1 = [f for f in findings if f.tier == Tier.P1]
        found_terms = {f.term for f in p1}
        assert "robust" in found_terms
        assert "comprehensive" in found_terms
        assert "holistic" in found_terms

    def test_three_terms_count(self):
        findings = scan_vocabulary(
            "We cultivate transformative and nuanced solutions."
        )
        p1 = [f for f in findings if f.tier == Tier.P1]
        assert len(p1) == 3


# ---------------------------------------------------------------------------
# P2: single occurrence NOT flagged
# ---------------------------------------------------------------------------

class TestP2SingleNotFlagged:
    def test_single_significant(self):
        findings = scan_vocabulary("The result was significant.")
        p2 = [f for f in findings if f.tier == Tier.P2]
        assert len(p2) == 0

    def test_single_effective(self):
        findings = scan_vocabulary("The treatment was effective.")
        p2 = [f for f in findings if f.tier == Tier.P2]
        assert len(p2) == 0

    def test_two_p2_terms_not_flagged_short_text(self):
        findings = scan_vocabulary("A significant and effective result.")
        p2 = [f for f in findings if f.tier == Tier.P2]
        assert len(p2) == 0


# ---------------------------------------------------------------------------
# P2: high density triggers flag
# ---------------------------------------------------------------------------

class TestP2HighDensity:
    def test_three_distinct_short_text(self):
        # Short text (< 500 words): threshold is 3
        text = "A significant, effective, and dynamic result was noted."
        findings = scan_vocabulary(text)
        p2 = [f for f in findings if f.tier == Tier.P2]
        assert len(p2) == 3

    def test_four_distinct_short_text(self):
        text = (
            "A significant, effective, dynamic, and compelling demonstration."
        )
        findings = scan_vocabulary(text)
        p2 = [f for f in findings if f.tier == Tier.P2]
        assert len(p2) >= 3

    def test_five_distinct_long_text(self):
        # Build text > 500 words: threshold is 5
        # Filler sentence is 7 words; 75 repetitions = 525 words
        filler = ("The study examined participants across multiple sites. " * 75)
        text = (
            filler
            + " A significant, effective, dynamic, compelling, and unprecedented outcome."
        )
        assert len(text.split()) > 500
        findings = scan_vocabulary(text)
        p2 = [f for f in findings if f.tier == Tier.P2]
        assert len(p2) == 5

    def test_four_distinct_long_text_not_flagged(self):
        filler = ("The study examined participants across multiple sites. " * 75)
        text = filler + " A significant, effective, dynamic, and compelling outcome."
        assert len(text.split()) > 500
        findings = scan_vocabulary(text)
        p2 = [f for f in findings if f.tier == Tier.P2]
        assert len(p2) == 0


# ---------------------------------------------------------------------------
# Context exemptions
# ---------------------------------------------------------------------------

class TestContextExemptions:
    def test_robust_exempt_in_methods(self):
        findings = scan_vocabulary("The robust pipeline was validated.", context="methods")
        p1 = [f for f in findings if f.tier == Tier.P1 and f.term == "robust"]
        # When only one P1 term is present, P1 is not triggered at all.
        # Test that even if triggered, it would be exempt.
        # Use two terms so P1 fires.
        findings2 = scan_vocabulary(
            "The robust and comprehensive pipeline.", context="methods"
        )
        p1_robust = [f for f in findings2 if f.tier == Tier.P1 and f.term == "robust"]
        assert p1_robust and p1_robust[0].context_exempt is True

    def test_robust_not_exempt_in_discussion(self):
        findings = scan_vocabulary(
            "The robust and comprehensive analysis.", context="discussion"
        )
        p1_robust = [f for f in findings if f.tier == Tier.P1 and f.term == "robust"]
        assert p1_robust and p1_robust[0].context_exempt is False

    def test_significant_exempt_in_results(self):
        # Need enough P2 terms to trigger P2 threshold
        text = "A significant, effective, and dynamic result."
        findings = scan_vocabulary(text, context="results")
        p2_sig = [f for f in findings if f.tier == Tier.P2 and f.term == "significant"]
        assert p2_sig and p2_sig[0].context_exempt is True

    def test_significant_not_exempt_in_introduction(self):
        text = "A significant, effective, and dynamic finding."
        findings = scan_vocabulary(text, context="introduction")
        p2_sig = [f for f in findings if f.tier == Tier.P2 and f.term == "significant"]
        assert p2_sig and p2_sig[0].context_exempt is False

    def test_comprehensive_exempt_in_methods(self):
        findings = scan_vocabulary(
            "The robust and comprehensive approach.", context="methods"
        )
        p1_comp = [
            f for f in findings if f.tier == Tier.P1 and f.term == "comprehensive"
        ]
        assert p1_comp and p1_comp[0].context_exempt is True

    def test_facilitate_exempt_in_methods(self):
        findings = scan_vocabulary(
            "These tools facilitate and bolster the workflow.", context="methods"
        )
        p1_fac = [f for f in findings if f.tier == Tier.P1 and f.term == "facilitate"]
        assert p1_fac and p1_fac[0].context_exempt is True

    def test_no_context_no_exemption(self):
        findings = scan_vocabulary("The robust and comprehensive pipeline.")
        p1 = [f for f in findings if f.tier == Tier.P1]
        assert all(not f.context_exempt for f in p1)


# ---------------------------------------------------------------------------
# Case-insensitive matching
# ---------------------------------------------------------------------------

class TestCaseInsensitive:
    def test_uppercase_p0(self):
        findings = scan_vocabulary("We UTILIZE a new method.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("utilize" in f.term for f in p0)

    def test_mixed_case_p0(self):
        findings = scan_vocabulary("We Leverage the approach.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert any("leverage" in f.term for f in p0)

    def test_uppercase_p1(self):
        findings = scan_vocabulary("A ROBUST and HOLISTIC design.")
        p1 = [f for f in findings if f.tier == Tier.P1]
        assert len(p1) == 2

    def test_titlecase_p2(self):
        text = "A Significant, Effective, and Dynamic experiment."
        findings = scan_vocabulary(text)
        p2 = [f for f in findings if f.tier == Tier.P2]
        assert len(p2) == 3


# ---------------------------------------------------------------------------
# Clean text returns no findings
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_plain_sentence(self):
        findings = scan_vocabulary("We measured bone displacement using ultrasound.")
        assert findings == []

    def test_academic_sentence(self):
        findings = scan_vocabulary(
            "Registration error was 1.43 ± 0.30 mm across all trials."
        )
        assert findings == []

    def test_empty_string(self):
        findings = scan_vocabulary("")
        assert findings == []

    def test_clinical_text(self):
        findings = scan_vocabulary(
            "Patients underwent intraoperative imaging to confirm alignment."
        )
        assert findings == []


# ---------------------------------------------------------------------------
# Finding has position field (char offset)
# ---------------------------------------------------------------------------

class TestPositionField:
    def test_p0_position(self):
        text = "We delve into the data."
        findings = scan_vocabulary(text)
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert p0
        # "delve into" starts at index 3
        assert p0[0].position == text.index("delve")

    def test_position_is_integer(self):
        findings = scan_vocabulary("We utilize this method.")
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert p0
        assert isinstance(p0[0].position, int)

    def test_position_within_text_bounds(self):
        text = "The tapestry of methods is rich."
        findings = scan_vocabulary(text)
        p0 = [f for f in findings if f.tier == Tier.P0]
        assert p0
        for f in p0:
            assert 0 <= f.position < len(text)

    def test_multiple_findings_sorted_by_position(self):
        text = "We utilize leverage in order to progress."
        findings = scan_vocabulary(text)
        p0 = [f for f in findings if f.tier == Tier.P0]
        positions = [f.position for f in p0]
        assert positions == sorted(positions)
