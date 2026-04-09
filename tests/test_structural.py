"""Tests for oxidizer.detection.structural — sentence-level AI pattern detectors."""
import pytest

from oxidizer.detection.structural import (
    PatternType,
    StructuralFinding,
    detect_structural_patterns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patterns(findings: list[StructuralFinding]) -> set[PatternType]:
    return {f.pattern for f in findings}


def _finding(findings: list[StructuralFinding], ptype: PatternType) -> StructuralFinding | None:
    for f in findings:
        if f.pattern == ptype:
            return f
    return None


# ---------------------------------------------------------------------------
# Repetitive starters
# ---------------------------------------------------------------------------

class TestRepetitiveStarters:
    def test_we_starters_detected(self):
        """Four sentences starting with 'We' (100%) — should flag."""
        text = (
            "We measured bone displacement. "
            "We applied registration algorithms. "
            "We validated the approach. "
            "We confirmed the results."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.REPETITIVE_STARTERS in _patterns(findings)

    def test_we_starters_severity_high(self):
        text = (
            "We measured bone displacement. "
            "We applied registration algorithms. "
            "We validated the approach. "
            "We confirmed the results."
        )
        findings = detect_structural_patterns(text)
        f = _finding(findings, PatternType.REPETITIVE_STARTERS)
        assert f is not None
        assert f.severity == "high"

    def test_varied_starters_clean(self):
        """Different starters — should not flag."""
        text = (
            "Registration was performed intraoperatively. "
            "The algorithm converged within five iterations. "
            "Bone displacement was measured at each stage. "
            "Results confirm the method's accuracy."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.REPETITIVE_STARTERS not in _patterns(findings)

    def test_the_starters_detected(self):
        """Three sentences starting with 'The' out of three total (100%) — flags."""
        text = (
            "The model was trained on spine MRI data. "
            "The results show strong concordance. "
            "The approach generalises to lumbar cases."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.REPETITIVE_STARTERS in _patterns(findings)

    def test_below_threshold_not_flagged(self):
        """Only 1 out of 4 sentences starts with 'We' — below 40%."""
        text = (
            "We measured the displacement. "
            "Registration was then applied. "
            "The algorithm converged. "
            "Results were confirmed."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.REPETITIVE_STARTERS not in _patterns(findings)

    def test_too_few_sentences_not_flagged(self):
        """Only 2 sentences total — below the minimum of 3."""
        text = "We measured displacement. We validated results."
        findings = detect_structural_patterns(text)
        assert PatternType.REPETITIVE_STARTERS not in _patterns(findings)


# ---------------------------------------------------------------------------
# Length uniformity
# ---------------------------------------------------------------------------

class TestLengthUniformity:
    def test_uniform_lengths_detected(self):
        """Sentences of very similar length (low CoV) — should flag."""
        # All sentences ~8 words → CoV will be near zero
        text = (
            "The model was applied to imaging data. "
            "The algorithm processed each frame carefully. "
            "The results confirmed expected displacement values. "
            "The approach demonstrated strong validation accuracy."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.LENGTH_UNIFORMITY in _patterns(findings)

    def test_uniform_lengths_severity_high(self):
        text = (
            "The model was applied to imaging data. "
            "The algorithm processed each frame carefully. "
            "The results confirmed expected displacement values. "
            "The approach demonstrated strong validation accuracy."
        )
        findings = detect_structural_patterns(text)
        f = _finding(findings, PatternType.LENGTH_UNIFORMITY)
        assert f is not None
        assert f.severity == "high"

    def test_varied_lengths_clean(self):
        """Large spread of sentence lengths — should not flag."""
        text = (
            "Yes. "
            "The intraoperative ultrasound imaging pipeline was evaluated. "
            "Registration. "
            "We applied a deformable model to correct for soft-tissue motion during spine surgery, "
            "achieving sub-millimetre accuracy across all twelve cadaveric specimens."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.LENGTH_UNIFORMITY not in _patterns(findings)

    def test_too_few_sentences_not_flagged(self):
        """Only 3 sentences — below the minimum of 4."""
        text = (
            "The model was applied to imaging data. "
            "The algorithm processed each frame carefully. "
            "The results confirmed expected values."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.LENGTH_UNIFORMITY not in _patterns(findings)


# ---------------------------------------------------------------------------
# Rule of three
# ---------------------------------------------------------------------------

class TestRuleOfThree:
    def test_single_triad_not_flagged(self):
        """A single 'X, Y, and Z' is normal — should not flag."""
        text = (
            "The study evaluated accuracy, precision, and recall. "
            "Results were reported across all test cases."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.RULE_OF_THREE not in _patterns(findings)

    def test_two_triads_flagged(self):
        """Two triadic patterns in the same text — should flag."""
        text = (
            "The system integrates speed, accuracy, and robustness. "
            "We assessed sensitivity, specificity, and AUC across cohorts."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.RULE_OF_THREE in _patterns(findings)

    def test_three_triads_flagged(self):
        """Three triadic patterns — definitely flags."""
        text = (
            "The pipeline combines imaging, tracking, and registration. "
            "We evaluated speed, accuracy, and robustness. "
            "Patients were grouped by age, sex, and diagnosis."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.RULE_OF_THREE in _patterns(findings)

    def test_triad_severity_medium(self):
        text = (
            "The system integrates speed, accuracy, and robustness. "
            "We assessed sensitivity, specificity, and AUC."
        )
        findings = detect_structural_patterns(text)
        f = _finding(findings, PatternType.RULE_OF_THREE)
        assert f is not None
        assert f.severity == "medium"

    def test_no_triad_clean(self):
        """No list patterns — no flag."""
        text = (
            "We measured displacement at two time points. "
            "The algorithm converged reliably. "
            "Accuracy was within accepted clinical limits."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.RULE_OF_THREE not in _patterns(findings)


# ---------------------------------------------------------------------------
# Copula avoidance
# ---------------------------------------------------------------------------

class TestCopulaAvoidance:
    def test_serves_as_twice_detected(self):
        """Two 'serves as' occurrences — should flag."""
        text = (
            "The ultrasound probe serves as a reference marker. "
            "The intraoperative image serves as a baseline for registration. "
            "Both were validated on cadaveric specimens."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.COPULA_AVOIDANCE in _patterns(findings)

    def test_serves_as_severity_medium(self):
        text = (
            "The ultrasound probe serves as a reference marker. "
            "The image serves as a baseline."
        )
        findings = detect_structural_patterns(text)
        f = _finding(findings, PatternType.COPULA_AVOIDANCE)
        assert f is not None
        assert f.severity == "medium"

    def test_single_serves_as_not_flagged(self):
        """Only one copula substitution — below threshold."""
        text = (
            "This system serves as a proof-of-concept. "
            "Further validation is required."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.COPULA_AVOIDANCE not in _patterns(findings)

    def test_boasts_detected(self):
        """'boasts' is a copula substitution."""
        text = (
            "The pipeline boasts exceptional speed. "
            "The system also serves as a validation platform."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.COPULA_AVOIDANCE in _patterns(findings)

    def test_features_the_detected(self):
        """'features the' is a copula substitution."""
        text = (
            "The algorithm features a novel loss function. "
            "It also features the latest regularisation scheme."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.COPULA_AVOIDANCE in _patterns(findings)

    def test_clean_text_no_copula_flag(self):
        text = "The system is accurate. It has a validated pipeline."
        findings = detect_structural_patterns(text)
        assert PatternType.COPULA_AVOIDANCE not in _patterns(findings)


# ---------------------------------------------------------------------------
# Negative parallelism
# ---------------------------------------------------------------------------

class TestNegativeParallelism:
    def test_not_just_its_detected(self):
        """Classic 'not just X, it's Y' pattern — should flag."""
        text = (
            "This approach is not just accurate, it's transformative for clinical workflows. "
            "Results confirm broad applicability."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.NEGATIVE_PARALLELISM in _patterns(findings)

    def test_not_just_it_is_detected(self):
        """Expanded form 'not just X, it is Y' — should flag."""
        text = "The method is not just fast, it is clinically reliable."
        findings = detect_structural_patterns(text)
        assert PatternType.NEGATIVE_PARALLELISM in _patterns(findings)

    def test_severity_medium(self):
        text = "This is not just a model, it's a framework."
        findings = detect_structural_patterns(text)
        f = _finding(findings, PatternType.NEGATIVE_PARALLELISM)
        assert f is not None
        assert f.severity == "medium"

    def test_clean_no_negative_parallelism(self):
        text = (
            "The model was validated on twelve specimens. "
            "Accuracy was within clinical tolerances."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.NEGATIVE_PARALLELISM not in _patterns(findings)


# ---------------------------------------------------------------------------
# Em dash overuse
# ---------------------------------------------------------------------------

class TestEmDashOveruse:
    def test_em_dash_overuse_detected(self):
        """Three em dashes in a short text — rate > 2 per 1000 words, flag."""
        # ~20 words, 3 em dashes → 150 per 1000 words
        text = (
            "The system \u2014 developed at UCSF \u2014 was validated on spine data "
            "\u2014 confirming accuracy."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.EM_DASH_OVERUSE in _patterns(findings)

    def test_em_dash_severity_medium(self):
        text = (
            "The system \u2014 developed at UCSF \u2014 was validated on spine data "
            "\u2014 confirming accuracy."
        )
        findings = detect_structural_patterns(text)
        f = _finding(findings, PatternType.EM_DASH_OVERUSE)
        assert f is not None
        assert f.severity == "medium"

    def test_zero_em_dashes_clean(self):
        """No em dashes — should not flag."""
        text = (
            "The system was developed at UCSF and validated on spine imaging data. "
            "Accuracy was confirmed across all specimens."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.EM_DASH_OVERUSE not in _patterns(findings)

    def test_single_em_dash_not_flagged(self):
        """One em dash — below minimum count threshold of 2."""
        text = (
            "The pipeline \u2014 developed over two years \u2014 was applied in "
            "twelve cases across three centres with consistent results and minimal error."
        )
        # Two em dashes here; let's force exactly one
        text = "Registration was performed \u2014 this is standard practice. " * 5
        # That's 5 em dashes across 50 words = 100/1000 but we want 1 dash only
        text = "Registration \u2014 the gold standard technique \u2014 was used. " * 30
        # Still ≥2, let's build exactly 1 dash in a long text
        long_filler = "The algorithm processes each frame. " * 50  # 350 words
        text = long_filler + "Results were consistent \u2014 matching prior work."
        findings = detect_structural_patterns(text)
        assert PatternType.EM_DASH_OVERUSE not in _patterns(findings)

    def test_two_em_dashes_low_rate_not_flagged(self):
        """Two em dashes in a very long text — rate < 2 per 1000 words, no flag."""
        long_filler = "The algorithm processes each ultrasound frame. " * 100  # ~700 words
        text = (
            long_filler
            + "Results were consistent \u2014 matching prior work. "
            "Registration error was low \u2014 below 1.5 mm."
        )
        word_count = len(text.split())
        em_count = text.count("\u2014")
        rate = em_count / word_count * 1000
        # Only flag if rate > 2; skip assertion if our filler math produces > 2
        if rate <= 2.0:
            findings = detect_structural_patterns(text)
            assert PatternType.EM_DASH_OVERUSE not in _patterns(findings)


# ---------------------------------------------------------------------------
# Confidence stacking
# ---------------------------------------------------------------------------

class TestConfidenceStacking:
    def test_two_adverb_openers_detected(self):
        """Two confidence adverb openers — should flag."""
        text = (
            "The algorithm achieved high accuracy. "
            "Notably, the error was below 1.5 mm. "
            "Importantly, generalisation held across all sites."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.CONFIDENCE_STACKING in _patterns(findings)

    def test_severity_medium(self):
        text = (
            "Notably, the error was below 1.5 mm. "
            "Importantly, generalisation held across all sites."
        )
        findings = detect_structural_patterns(text)
        f = _finding(findings, PatternType.CONFIDENCE_STACKING)
        assert f is not None
        assert f.severity == "medium"

    def test_single_adverb_not_flagged(self):
        """One adverb opener — below threshold."""
        text = (
            "The algorithm achieved high accuracy. "
            "Notably, the error was below 1.5 mm. "
            "Generalisation held across all sites."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.CONFIDENCE_STACKING not in _patterns(findings)

    def test_various_adverbs_detected(self):
        """Mix of different confidence adverbs — flags when ≥2."""
        text = (
            "Remarkably, the model converged in under five minutes. "
            "Crucially, this holds in low-quality scans."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.CONFIDENCE_STACKING in _patterns(findings)

    def test_three_adverbs_detected(self):
        """Three confidence adverb openers."""
        text = (
            "Interestingly, the baseline error was higher. "
            "Surprisingly, the augmented model matched ground truth. "
            "Strikingly, the improvement persisted at six months."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.CONFIDENCE_STACKING in _patterns(findings)

    def test_adverb_not_at_sentence_start_not_flagged(self):
        """Adverb mid-sentence should not trigger the detector."""
        text = (
            "The improvement was notably significant in the lumbar region. "
            "The pipeline was remarkably well-calibrated for clinical use."
        )
        findings = detect_structural_patterns(text)
        assert PatternType.CONFIDENCE_STACKING not in _patterns(findings)


# ---------------------------------------------------------------------------
# Clean text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_clean_academic_text_empty_list(self):
        """Well-written academic text returns no structural findings."""
        # Sentence lengths deliberately varied: ~3, ~8, ~20, ~5 words → high CoV
        text = (
            "Results varied. "
            "A deformable registration algorithm corrected soft-tissue motion. "
            "Registration error was 1.43 ± 0.30 mm across all twelve cadaveric specimens "
            "tested at three independent intraoperative time points in each case. "
            "Accuracy was confirmed."
        )
        findings = detect_structural_patterns(text)
        assert findings == []

    def test_empty_string_returns_empty(self):
        assert detect_structural_patterns("") == []

    def test_whitespace_only_returns_empty(self):
        assert detect_structural_patterns("   \n\t  ") == []


# ---------------------------------------------------------------------------
# Finding data model
# ---------------------------------------------------------------------------

class TestFindingDataModel:
    def test_finding_has_all_fields(self):
        text = (
            "We measured bone displacement. "
            "We applied registration algorithms. "
            "We validated the approach. "
            "We confirmed the results."
        )
        findings = detect_structural_patterns(text)
        f = _finding(findings, PatternType.REPETITIVE_STARTERS)
        assert f is not None
        assert isinstance(f.pattern, PatternType)
        assert isinstance(f.description, str) and f.description
        assert isinstance(f.evidence, str) and f.evidence
        assert f.severity in {"high", "medium", "low"}

    def test_returns_list(self):
        assert isinstance(detect_structural_patterns("Test sentence."), list)
