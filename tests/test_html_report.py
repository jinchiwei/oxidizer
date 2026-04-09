"""Tests for oxidizer/scoring/html_report.py."""
from __future__ import annotations

import pytest

from oxidizer.scoring.html_report import generate_html_report
from oxidizer.scoring.reporter import StyleReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_report(score: float = 0.82) -> StyleReport:
    """Build a minimal StyleReport for testing."""
    sub_scores = {
        "sentence_length_mean": 0.90,
        "sentence_length_variance": 0.80,
        "active_voice": 0.95,
        "banned_words": 0.70,
        "semicolons": 0.65,
        "parentheticals": 0.80,
        "transitions": 0.60,
        "contractions": 1.00,
    }
    return StyleReport(
        style_match_score=score,
        sentence_length_mean=18.5,
        sentence_length_std=5.2,
        sentence_length_target_mean=20.0,
        sentence_length_target_std=6.0,
        active_voice_ratio=0.88,
        active_voice_target=0.90,
        banned_words_found=[],
        semicolons_per_100=3.0,
        semicolons_target=4.0,
        parentheticals_per_100=8.0,
        parentheticals_target=8.0,
        contraction_count=0,
        transition_score=0.60,
        sub_scores=sub_scores,
    )


def _make_report_with_banned() -> StyleReport:
    """Build a StyleReport that contains banned words."""
    sub_scores = {
        "sentence_length_mean": 0.75,
        "sentence_length_variance": 0.70,
        "active_voice": 0.80,
        "banned_words": 0.60,
        "semicolons": 0.55,
        "parentheticals": 0.70,
        "transitions": 0.50,
        "contractions": 0.80,
    }
    return StyleReport(
        style_match_score=0.65,
        sentence_length_mean=22.0,
        sentence_length_std=7.0,
        sentence_length_target_mean=20.0,
        sentence_length_target_std=6.0,
        active_voice_ratio=0.75,
        active_voice_target=0.90,
        banned_words_found=["leverage", "paradigm shift", "robust"],
        semicolons_per_100=2.0,
        semicolons_target=4.0,
        parentheticals_per_100=5.0,
        parentheticals_target=8.0,
        contraction_count=2,
        transition_score=0.50,
        sub_scores=sub_scores,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_html_report_generates_valid_html():
    """generate_html_report should return a non-empty HTML string."""
    report = _make_report()
    html = generate_html_report(report)

    assert isinstance(html, str)
    assert len(html) > 100
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html


def test_html_report_contains_score():
    """HTML report should display the overall style match score."""
    report = _make_report(score=0.82)
    html = generate_html_report(report)

    # The score as a percentage should appear in the output
    assert "82.0" in html or "82" in html


def test_html_report_title_appears():
    """Custom title should appear in the HTML output."""
    report = _make_report()
    html = generate_html_report(report, title="My Custom Report")

    assert "My Custom Report" in html


def test_html_report_default_title():
    """Default title should be 'Style Report'."""
    report = _make_report()
    html = generate_html_report(report)

    assert "Style Report" in html


def test_html_report_contains_metric_labels():
    """HTML should contain all metric label text."""
    report = _make_report()
    html = generate_html_report(report)

    expected_labels = [
        "Sentence Length",
        "Active Voice",
        "Banned Words",
        "Semicolons",
        "Transitions",
        "Contractions",
    ]
    for label in expected_labels:
        assert label in html, f"Expected label '{label}' not found in HTML"


def test_html_report_banned_words_section():
    """HTML report should include a banned words section when words are found."""
    report = _make_report_with_banned()
    html = generate_html_report(report)

    assert "leverage" in html
    assert "paradigm shift" in html
    assert "robust" in html
    assert "Banned Words" in html


def test_html_report_no_banned_words():
    """HTML report should say no banned words when none are found."""
    report = _make_report()
    html = generate_html_report(report)

    assert "No banned words found" in html


def test_html_report_inline_css():
    """HTML should include inline CSS with no external stylesheets."""
    report = _make_report()
    html = generate_html_report(report)

    assert "<style>" in html
    # Should NOT reference external CSS files
    assert 'rel="stylesheet"' not in html
    assert "href=" not in html.replace("html>", "")  # no link[href] tags


def test_html_report_no_js():
    """HTML should contain no JavaScript."""
    report = _make_report()
    html = generate_html_report(report)

    assert "<script" not in html


def test_html_report_color_coding_green():
    """High scores (>0.85) should use green color."""
    report = _make_report(score=0.90)
    html = generate_html_report(report)

    assert "#27ae60" in html  # green hex


def test_html_report_color_coding_red():
    """Low scores (<0.60) should use red color."""
    report = _make_report(score=0.40)
    html = generate_html_report(report)

    assert "#e74c3c" in html  # red hex


def test_html_report_color_coding_yellow():
    """Mid-range scores (0.60-0.85) should use yellow/orange color."""
    report = _make_report(score=0.72)
    html = generate_html_report(report)

    assert "#f39c12" in html  # yellow/orange hex


def test_html_report_bar_chart_present():
    """HTML should include CSS bar chart elements."""
    report = _make_report()
    html = generate_html_report(report)

    assert "bar" in html.lower()
    assert "width:" in html


def test_html_report_self_contained():
    """HTML should be fully self-contained (no external dependencies)."""
    report = _make_report()
    html = generate_html_report(report)

    # No external script sources
    assert "src=" not in html
    # No CDN links
    assert "cdn" not in html.lower()
    assert "googleapis" not in html
    assert "bootstrapcdn" not in html


def test_html_report_escapes_special_chars():
    """generate_html_report should safely escape special characters in title."""
    report = _make_report()
    html = generate_html_report(report, title="Report <script>alert('xss')</script>")

    # Script tag should NOT appear raw
    assert "<script>alert" not in html
    # But the escaped version should be safe
    assert "&lt;script&gt;" in html
