"""Tests for oxidizer.scoring.reporter."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from oxidizer.profiles.loader import load_profile_from_path
from oxidizer.scoring.reporter import StyleReport, compute_style_report

# ---------------------------------------------------------------------------
# Fixture: load the bundled jinchi profile
# ---------------------------------------------------------------------------

_PROFILE_PATH = (
    Path(__file__).parent.parent / "profiles" / "jinchi.yaml"
)


@pytest.fixture(scope="module")
def jinchi_profile():
    return load_profile_from_path(_PROFILE_PATH)


# ---------------------------------------------------------------------------
# Clean academic text (should score well)
# ---------------------------------------------------------------------------

_CLEAN_TEXT = (
    "We propose a registration framework that leverages spline-based deformable models. "
    "The method attains robust performance across diverse imaging conditions. "
    "Results demonstrate that the approach is scalable to large datasets (n=500). "
    "However, the technique requires calibrated equipment. "
    "As a result, we recommend additional validation in clinical settings. "
    "Similarly, the pipeline readily integrates with existing workflows. "
    "We analyzed 24 patients over two years of follow-up. "
    "The mean registration error was 1.43 ± 0.30 mm across all cases. "
    "These findings confirm the holistic utility of the system. "
    "Building upon prior work, we extend the framework to multi-modal inputs."
)

# ---------------------------------------------------------------------------
# Dirty text: many banned words + contractions
# ---------------------------------------------------------------------------

_DIRTY_TEXT = (
    "It's worth noting that this novel, groundbreaking framework can't be overstated. "
    "We delve into the multifaceted tapestry of methods, harnessing cutting-edge techniques. "
    "Don't overlook the pivotal role of transformative paradigms. "
    "The innovative landscape of this realm is a cornerstone of modern research. "
    "We won't ignore the myriad of synergy effects."
)


# ---------------------------------------------------------------------------
# StyleReport field existence
# ---------------------------------------------------------------------------

class TestStyleReportFields:
    EXPECTED_FIELDS = {
        "style_match_score",
        "sentence_length_mean",
        "sentence_length_std",
        "sentence_length_target_mean",
        "sentence_length_target_std",
        "active_voice_ratio",
        "active_voice_target",
        "banned_words_found",
        "semicolons_per_100",
        "semicolons_target",
        "parentheticals_per_100",
        "parentheticals_target",
        "contraction_count",
        "transition_score",
        "sub_scores",
    }

    def test_all_fields_present(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        actual_fields = {f.name for f in fields(report)}
        assert self.EXPECTED_FIELDS == actual_fields

    def test_style_match_score_is_float(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        assert isinstance(report.style_match_score, float)

    def test_sentence_length_mean_is_float(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        assert isinstance(report.sentence_length_mean, float)

    def test_banned_words_found_is_list(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        assert isinstance(report.banned_words_found, list)

    def test_sub_scores_has_all_dimensions(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        expected_dims = {
            "sentence_length_mean",
            "sentence_length_variance",
            "active_voice",
            "banned_words",
            "semicolons",
            "parentheticals",
            "transitions",
            "contractions",
        }
        assert expected_dims == set(report.sub_scores.keys())

    def test_sub_scores_in_range(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        for dim, score in report.sub_scores.items():
            assert 0.0 <= score <= 1.0, f"sub_score[{dim!r}] = {score} out of [0,1]"


# ---------------------------------------------------------------------------
# Clean text scores above 0.5
# ---------------------------------------------------------------------------

class TestCleanTextScoresHigh:
    def test_composite_above_threshold(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        assert report.style_match_score > 0.5, (
            f"Expected clean text score > 0.5, got {report.style_match_score}"
        )

    def test_no_contractions_in_clean_text(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        assert report.contraction_count == 0

    def test_no_banned_words_in_clean_text(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        assert report.banned_words_found == []

    def test_active_voice_ratio_is_float_in_range(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        assert 0.0 <= report.active_voice_ratio <= 1.0

    def test_targets_match_profile(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        assert report.sentence_length_target_mean == jinchi_profile.sentence_length.mean
        assert report.sentence_length_target_std == jinchi_profile.sentence_length.std
        assert report.active_voice_target == jinchi_profile.voice.active_ratio
        assert report.semicolons_target == jinchi_profile.punctuation.semicolons_per_100
        assert report.parentheticals_target == jinchi_profile.punctuation.parentheticals_per_100


# ---------------------------------------------------------------------------
# Dirty text scores lower than clean text
# ---------------------------------------------------------------------------

class TestDirtyTextScoresLower:
    def test_dirty_scores_lower_than_clean(self, jinchi_profile):
        clean_report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        dirty_report = compute_style_report(_DIRTY_TEXT, jinchi_profile)
        assert dirty_report.style_match_score < clean_report.style_match_score, (
            f"Dirty ({dirty_report.style_match_score}) should be less than "
            f"clean ({clean_report.style_match_score})"
        )

    def test_dirty_has_banned_words(self, jinchi_profile):
        report = compute_style_report(_DIRTY_TEXT, jinchi_profile)
        assert len(report.banned_words_found) > 0

    def test_dirty_has_contractions(self, jinchi_profile):
        report = compute_style_report(_DIRTY_TEXT, jinchi_profile)
        assert report.contraction_count > 0

    def test_dirty_banned_words_sub_score_lower(self, jinchi_profile):
        clean_report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        dirty_report = compute_style_report(_DIRTY_TEXT, jinchi_profile)
        assert dirty_report.sub_scores["banned_words"] < clean_report.sub_scores["banned_words"]

    def test_dirty_contraction_sub_score_zero(self, jinchi_profile):
        report = compute_style_report(_DIRTY_TEXT, jinchi_profile)
        assert report.sub_scores["contractions"] == 0.0


# ---------------------------------------------------------------------------
# to_dict is JSON-serializable
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_returns_dict(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        d = report.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_is_json_serializable(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        d = report.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_to_dict_round_trips_score(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        d = report.to_dict()
        assert d["style_match_score"] == report.style_match_score

    def test_to_dict_contains_sub_scores(self, jinchi_profile):
        report = compute_style_report(_CLEAN_TEXT, jinchi_profile)
        d = report.to_dict()
        assert "sub_scores" in d
        assert isinstance(d["sub_scores"], dict)

    def test_to_dict_banned_words_is_list(self, jinchi_profile):
        report = compute_style_report(_DIRTY_TEXT, jinchi_profile)
        d = report.to_dict()
        assert isinstance(d["banned_words_found"], list)


# ---------------------------------------------------------------------------
# Empty text edge case
# ---------------------------------------------------------------------------

class TestEmptyText:
    def test_empty_text_does_not_crash(self, jinchi_profile):
        report = compute_style_report("", jinchi_profile)
        assert isinstance(report, StyleReport)

    def test_empty_text_score_in_range(self, jinchi_profile):
        report = compute_style_report("", jinchi_profile)
        assert 0.0 <= report.style_match_score <= 1.0

    def test_empty_text_no_banned_words(self, jinchi_profile):
        report = compute_style_report("", jinchi_profile)
        assert report.banned_words_found == []

    def test_empty_text_contraction_count_zero(self, jinchi_profile):
        report = compute_style_report("", jinchi_profile)
        assert report.contraction_count == 0

    def test_empty_text_to_dict_json_serializable(self, jinchi_profile):
        report = compute_style_report("", jinchi_profile)
        d = report.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
