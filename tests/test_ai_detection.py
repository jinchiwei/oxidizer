"""Tests for oxidizer.detection.registry — full AI detection pipeline."""
import pytest

from oxidizer.detection.registry import (
    DetectionReport,
    Severity,
    run_full_detection,
)
from oxidizer.detection.vocabulary import Tier


# ---------------------------------------------------------------------------
# Fixtures / sample texts
# ---------------------------------------------------------------------------

CLEAN_TEXT = (
    "Intraoperative ultrasound imaging was used to track bone surface deformation "
    "during spine surgery. "
    "The probe was attached to a calibrated mechanical arm and registered to the "
    "preoperative CT volume. "
    "Surface points were collected. "
    "Registration error, computed as the root-mean-square distance between "
    "corresponding landmarks on phantom and image, remained below 1.5 mm in all "
    "five cadaveric specimens tested. "
    "No fiducial markers were required, and the procedure added approximately four "
    "minutes to the operative workflow. "
    "Deformations as small as 2 mm were detectable."
)

# Contains P0 terms: delve, tapestry, serves as, in order to, leverage, utilize,
# seamless, cutting-edge (P1).
AI_HEAVY_TEXT = (
    "This paper aims to delve into the intricate tapestry of machine learning "
    "methods that serves as the foundation for our analysis. In order to leverage "
    "the full potential of the dataset, we utilize a cutting-edge pipeline that "
    "enables seamless integration of heterogeneous data sources. "
    "The framework is robust, comprehensive, and holistic, offering a nuanced "
    "understanding of the underlying structure. Notably, this approach is "
    "groundbreaking. Interestingly, results confirm the paradigm shift."
)

MODERATE_TEXT = (
    "We leverage the existing dataset to build a classifier. "
    "The model is trained on labeled examples. "
    "Results show improved performance over the baseline."
)


# ---------------------------------------------------------------------------
# Core correctness tests
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_clean_text_severity(self):
        report = run_full_detection(CLEAN_TEXT)
        assert report.overall_severity == Severity.CLEAN

    def test_clean_text_summary(self):
        report = run_full_detection(CLEAN_TEXT)
        assert report.summary == "No AI patterns detected."


class TestAIHeavyText:
    def test_ai_heavy_severity_high_or_critical(self):
        report = run_full_detection(AI_HEAVY_TEXT)
        assert report.overall_severity in (Severity.HIGH, Severity.CRITICAL)

    def test_ai_heavy_has_p0_findings(self):
        report = run_full_detection(AI_HEAVY_TEXT)
        p0 = [f for f in report.vocab_findings if f.tier == Tier.P0]
        assert len(p0) >= 1

    def test_ai_heavy_summary_contains_severity(self):
        report = run_full_detection(AI_HEAVY_TEXT)
        assert "AI Detection" in report.summary
        assert report.overall_severity.name in report.summary


# ---------------------------------------------------------------------------
# Report structure tests
# ---------------------------------------------------------------------------

class TestReportStructure:
    def test_all_fields_present(self):
        report = run_full_detection(AI_HEAVY_TEXT)
        assert hasattr(report, "vocab_findings")
        assert hasattr(report, "structural_findings")
        assert hasattr(report, "statistical_report")
        assert hasattr(report, "overall_severity")
        assert hasattr(report, "summary")

    def test_vocab_findings_is_list(self):
        report = run_full_detection(AI_HEAVY_TEXT)
        assert isinstance(report.vocab_findings, list)

    def test_structural_findings_is_list(self):
        report = run_full_detection(AI_HEAVY_TEXT)
        assert isinstance(report.structural_findings, list)

    def test_statistical_report_has_burstiness(self):
        report = run_full_detection(AI_HEAVY_TEXT)
        assert hasattr(report.statistical_report, "burstiness")
        assert isinstance(report.statistical_report.burstiness, float)

    def test_overall_severity_is_severity_enum(self):
        report = run_full_detection(AI_HEAVY_TEXT)
        assert isinstance(report.overall_severity, Severity)

    def test_summary_is_string(self):
        report = run_full_detection(AI_HEAVY_TEXT)
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_vocab_findings_count_property(self):
        report = run_full_detection(AI_HEAVY_TEXT)
        assert report.vocab_findings_count == len(report.vocab_findings)

    def test_vocab_findings_count_property_clean(self):
        report = run_full_detection(CLEAN_TEXT)
        assert report.vocab_findings_count == 0


# ---------------------------------------------------------------------------
# Context exemption tests
# ---------------------------------------------------------------------------

class TestContextExemptions:
    def test_robust_exempt_in_methods(self):
        text = "The robust algorithm was validated in the methods section."
        # Without context: robust (P1) may be flagged
        report_no_ctx = run_full_detection(text)
        # With methods context: robust should be exempt
        report_methods = run_full_detection(text, context="methods")

        robust_findings_no_ctx = [
            f for f in report_no_ctx.vocab_findings
            if f.term == "robust" and not f.context_exempt
        ]
        robust_findings_methods = [
            f for f in report_methods.vocab_findings
            if f.term == "robust" and not f.context_exempt
        ]
        # In methods context, robust findings should be marked exempt (not non-exempt)
        assert len(robust_findings_methods) == 0 or all(
            f.context_exempt for f in report_methods.vocab_findings if f.term == "robust"
        )

    def test_context_affects_severity(self):
        """A text with only exempt terms in context should be cleaner."""
        # Text with robust and facilitate — both exempt in methods
        text = (
            "The robust statistical framework was used to facilitate "
            "the comparison between groups."
        )
        report_no_ctx = run_full_detection(text)
        report_methods = run_full_detection(text, context="methods")
        # Methods context should produce equal or lower severity
        assert report_methods.overall_severity <= report_no_ctx.overall_severity


# ---------------------------------------------------------------------------
# Severity ordering tests
# ---------------------------------------------------------------------------

class TestSeverityOrdering:
    def test_severity_ordering_clean_lt_moderate(self):
        report_clean = run_full_detection(CLEAN_TEXT)
        report_moderate = run_full_detection(MODERATE_TEXT)
        # Clean should be less than or equal to moderate
        assert report_clean.overall_severity <= report_moderate.overall_severity

    def test_severity_ordering_moderate_lt_heavy(self):
        report_moderate = run_full_detection(MODERATE_TEXT)
        report_heavy = run_full_detection(AI_HEAVY_TEXT)
        assert report_moderate.overall_severity <= report_heavy.overall_severity

    def test_severity_ordering_clean_lt_heavy(self):
        report_clean = run_full_detection(CLEAN_TEXT)
        report_heavy = run_full_detection(AI_HEAVY_TEXT)
        assert report_clean.overall_severity < report_heavy.overall_severity

    def test_severity_enum_ordering(self):
        assert Severity.CLEAN < Severity.LOW
        assert Severity.LOW < Severity.MEDIUM
        assert Severity.MEDIUM < Severity.HIGH
        assert Severity.HIGH < Severity.CRITICAL


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_text_returns_clean(self):
        report = run_full_detection("")
        assert report.overall_severity == Severity.CLEAN
        assert report.vocab_findings_count == 0

    def test_single_sentence_returns_report(self):
        report = run_full_detection("We analyzed the data.")
        assert isinstance(report, DetectionReport)

    def test_none_context_is_equivalent_to_no_context(self):
        report_none = run_full_detection(CLEAN_TEXT, context=None)
        report_default = run_full_detection(CLEAN_TEXT)
        assert report_none.overall_severity == report_default.overall_severity

    def test_p0_terms_in_summary(self):
        text = "We delve into this tapestry of methods."
        report = run_full_detection(text)
        if report.overall_severity != Severity.CLEAN:
            # P0 terms should appear somewhere in the summary
            assert "delve" in report.summary or "tapestry" in report.summary

    def test_statistical_report_flags_list(self):
        report = run_full_detection(CLEAN_TEXT)
        assert isinstance(report.statistical_report.ai_risk_flags, list)
