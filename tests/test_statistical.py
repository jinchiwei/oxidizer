"""Tests for oxidizer.detection.statistical — statistical AI signal detection."""
import pytest

from oxidizer.detection.statistical import (
    StatisticalReport,
    analyze_statistical_signals,
    compute_burstiness,
    compute_sentence_cov,
    compute_trigram_repetition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Uniform text: every sentence is ~10 words (metronomic / AI-like rhythm).
_UNIFORM_TEXT = (
    "The model achieved high accuracy on all test sets. "
    "The pipeline processed inputs within the allotted time. "
    "Each module was validated against ground truth data. "
    "Results were consistent across all experimental conditions. "
    "The method outperformed baselines on every benchmark metric. "
    "Performance remained stable throughout the evaluation period. "
)

# Varied text: sentence lengths differ substantially (bursty / human-like).
_VARIED_TEXT = (
    "Short. "
    "This sentence is a bit longer than the previous one. "
    "Ok. "
    "This particular sentence is quite a bit longer and contains many more words "
    "than the shorter ones above, which creates genuine variance. "
    "Brief. "
    "Another moderately long sentence that adds further variation to the rhythm. "
)

# Highly repetitive trigrams (AI-like).
_REPETITIVE_TEXT = (
    "the model achieved high accuracy. "
    "the model achieved high accuracy. "
    "the model achieved high accuracy. "
    "the model achieved high accuracy. "
    "the model achieved high accuracy on the test set. "
)

# Minimal trigram overlap (human-like).
_VARIED_TRIGRAM_TEXT = (
    "Bone displacement was measured intraoperatively using tracked ultrasound. "
    "Registration error averaged 1.4 mm across fifteen cadaveric specimens. "
    "A Kalman filter fused optical tracking with inertial measurements. "
    "Surgeons reported improved confidence in implant positioning. "
    "Further validation is planned in a prospective clinical trial. "
)


# ---------------------------------------------------------------------------
# compute_burstiness
# ---------------------------------------------------------------------------

class TestComputeBurstiness:
    def test_uniform_text_low_burstiness(self):
        score = compute_burstiness(_UNIFORM_TEXT)
        assert score < 0.3, f"Expected < 0.3, got {score}"

    def test_varied_text_higher_burstiness(self):
        score = compute_burstiness(_VARIED_TEXT)
        assert score > 0.3, f"Expected > 0.3, got {score}"

    def test_empty_string_returns_zero(self):
        assert compute_burstiness("") == 0.0

    def test_whitespace_only_returns_zero(self):
        assert compute_burstiness("   ") == 0.0

    def test_fewer_than_three_sentences_returns_zero(self):
        assert compute_burstiness("One sentence only.") == 0.0
        assert compute_burstiness("First sentence. Second sentence.") == 0.0

    def test_result_in_unit_interval(self):
        for text in (_UNIFORM_TEXT, _VARIED_TEXT, _REPETITIVE_TEXT):
            score = compute_burstiness(text)
            assert 0.0 <= score <= 1.0, f"Score out of [0,1]: {score}"

    def test_returns_float(self):
        assert isinstance(compute_burstiness(_UNIFORM_TEXT), float)


# ---------------------------------------------------------------------------
# compute_trigram_repetition
# ---------------------------------------------------------------------------

class TestComputeTrigramRepetition:
    def test_repetitive_text_high_score(self):
        score = compute_trigram_repetition(_REPETITIVE_TEXT)
        assert score > 0.05, f"Expected > 0.05, got {score}"

    def test_varied_text_low_score(self):
        score = compute_trigram_repetition(_VARIED_TRIGRAM_TEXT)
        assert score < 0.10, f"Expected < 0.10, got {score}"

    def test_empty_string_returns_zero(self):
        assert compute_trigram_repetition("") == 0.0

    def test_whitespace_only_returns_zero(self):
        assert compute_trigram_repetition("   ") == 0.0

    def test_fewer_than_four_words_returns_zero(self):
        assert compute_trigram_repetition("one two three") == 0.0

    def test_score_is_fraction(self):
        score = compute_trigram_repetition(_REPETITIVE_TEXT)
        assert 0.0 <= score <= 1.0, f"Score not in [0,1]: {score}"

    def test_returns_float(self):
        assert isinstance(compute_trigram_repetition(_REPETITIVE_TEXT), float)


# ---------------------------------------------------------------------------
# compute_sentence_cov
# ---------------------------------------------------------------------------

class TestComputeSentenceCov:
    def test_uniform_text_low_cov(self):
        score = compute_sentence_cov(_UNIFORM_TEXT)
        assert score < 0.15, f"Expected < 0.15, got {score}"

    def test_varied_text_high_cov(self):
        score = compute_sentence_cov(_VARIED_TEXT)
        assert score > 0.4, f"Expected > 0.4, got {score}"

    def test_empty_string_returns_zero(self):
        assert compute_sentence_cov("") == 0.0

    def test_whitespace_only_returns_zero(self):
        assert compute_sentence_cov("   ") == 0.0

    def test_fewer_than_three_sentences_returns_zero(self):
        assert compute_sentence_cov("Only one sentence here.") == 0.0
        assert compute_sentence_cov("Sentence one. Sentence two.") == 0.0

    def test_returns_float(self):
        assert isinstance(compute_sentence_cov(_UNIFORM_TEXT), float)

    def test_non_negative(self):
        for text in (_UNIFORM_TEXT, _VARIED_TEXT):
            assert compute_sentence_cov(text) >= 0.0


# ---------------------------------------------------------------------------
# analyze_statistical_signals
# ---------------------------------------------------------------------------

class TestAnalyzeStatisticalSignals:
    def test_returns_statistical_report(self):
        report = analyze_statistical_signals(_UNIFORM_TEXT)
        assert isinstance(report, StatisticalReport)

    def test_report_has_all_fields(self):
        report = analyze_statistical_signals(_UNIFORM_TEXT)
        assert hasattr(report, "burstiness")
        assert hasattr(report, "trigram_repetition")
        assert hasattr(report, "sentence_cov")
        assert hasattr(report, "ai_risk_flags")

    def test_flags_is_list(self):
        report = analyze_statistical_signals(_UNIFORM_TEXT)
        assert isinstance(report.ai_risk_flags, list)

    def test_ai_like_text_produces_flags(self):
        # Build a text that is long enough (> 50 words), metronomic (uniform
        # sentence lengths), and has repetitive trigrams.
        ai_text = _UNIFORM_TEXT * 3  # well above 50 words
        report = analyze_statistical_signals(ai_text)
        assert len(report.ai_risk_flags) >= 1, (
            f"Expected at least one flag; got {report.ai_risk_flags!r}. "
            f"burstiness={report.burstiness:.3f}, "
            f"trigram_rep={report.trigram_repetition:.3f}, "
            f"cov={report.sentence_cov:.3f}"
        )

    def test_highly_repetitive_text_flags_trigram(self):
        report = analyze_statistical_signals(_REPETITIVE_TEXT * 4)
        flag_text = " ".join(report.ai_risk_flags)
        assert "trigram" in flag_text.lower(), (
            f"Expected trigram flag; got {report.ai_risk_flags!r}"
        )

    def test_clean_human_text_fewer_flags(self):
        # Varied human-like text should produce fewer flags than AI-like text.
        human_report = analyze_statistical_signals(_VARIED_TEXT * 3)
        ai_report = analyze_statistical_signals(_UNIFORM_TEXT * 3)
        assert len(human_report.ai_risk_flags) <= len(ai_report.ai_risk_flags)

    def test_empty_text_no_flags(self):
        report = analyze_statistical_signals("")
        assert report.ai_risk_flags == []
        assert report.burstiness == 0.0
        assert report.trigram_repetition == 0.0
        assert report.sentence_cov == 0.0

    def test_metric_values_match_individual_functions(self):
        text = _UNIFORM_TEXT
        report = analyze_statistical_signals(text)
        assert report.burstiness == pytest.approx(compute_burstiness(text))
        assert report.trigram_repetition == pytest.approx(
            compute_trigram_repetition(text)
        )
        assert report.sentence_cov == pytest.approx(compute_sentence_cov(text))
