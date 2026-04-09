"""Tests for oxidizer/diff.py."""
from __future__ import annotations

import pytest

from oxidizer.diff import DiffChange, DiffResult, compute_diff


# ---------------------------------------------------------------------------
# Basic diff detection
# ---------------------------------------------------------------------------

def test_diff_detects_changes():
    """compute_diff should detect changed lines between two texts."""
    original = "The results were obtained by performing a complex analysis.\nThis was done by our team."
    revised = "We performed a complex analysis of the results.\nOur team completed the work."

    result = compute_diff(original, revised)

    assert isinstance(result, DiffResult)
    assert len(result.changes) > 0

    for change in result.changes:
        assert isinstance(change, DiffChange)
        assert isinstance(change.line_num, int)
        assert change.line_num >= 1
        assert isinstance(change.annotation, str)
        assert len(change.annotation) > 0


def test_diff_empty_files():
    """compute_diff on two empty strings should produce no changes."""
    result = compute_diff("", "")
    assert isinstance(result, DiffResult)
    assert len(result.changes) == 0
    assert result.original_report is None
    assert result.revised_report is None


def test_diff_identical_files():
    """compute_diff on identical texts should produce no changes."""
    text = "We performed an analysis. The results show improvement."
    result = compute_diff(text, text)
    assert len(result.changes) == 0


def test_diff_preserves_reports():
    """compute_diff should carry through pre-computed reports."""
    orig_report = {"style_match_score": 0.72}
    rev_report = {"style_match_score": 0.88}

    result = compute_diff(
        "Original text.",
        "Revised text.",
        original_report=orig_report,
        revised_report=rev_report,
    )

    assert result.original_report == orig_report
    assert result.revised_report == rev_report


def test_diff_detects_passive_annotation():
    """Changes from passive to active should be annotated appropriately."""
    # Passive → active transition
    original = "The data was collected by researchers."
    revised = "Researchers collected the data."

    result = compute_diff(original, revised)

    assert len(result.changes) > 0
    # At least one change should mention passive or style change
    annotations = [c.annotation for c in result.changes]
    assert any(
        "passive" in a or "style change" in a or "sentence" in a
        for a in annotations
    )


def test_diff_with_banned_list():
    """Banned word removal should be annotated when banned_list is provided."""
    original = "This leverages a paradigm shift to showcase the robust approach."
    revised = "This uses a framework to demonstrate the reliable approach."

    result = compute_diff(
        original,
        revised,
        banned_list=["leverage", "leverages", "paradigm shift", "showcase", "robust"],
    )

    assert len(result.changes) > 0
    annotations = [c.annotation for c in result.changes]
    assert any("banned" in a for a in annotations)


def test_diff_line_added():
    """Lines added in revised should appear as DiffChange with empty original."""
    original = "First line."
    revised = "First line.\nSecond line added."

    result = compute_diff(original, revised)

    # Should have at least one insertion
    assert len(result.changes) > 0
    additions = [c for c in result.changes if c.original == "" and c.revised]
    assert len(additions) > 0


def test_diff_line_removed():
    """Lines removed from original should appear as DiffChange with empty revised."""
    original = "First line.\nSecond line to remove."
    revised = "First line."

    result = compute_diff(original, revised)

    assert len(result.changes) > 0
    removals = [c for c in result.changes if c.revised == "" and c.original]
    assert len(removals) > 0


def test_diff_change_dataclass_fields():
    """DiffChange should have all required fields with correct types."""
    change = DiffChange(
        line_num=5,
        original="The test was run.",
        revised="We ran the test.",
        annotation="passive → active",
    )

    assert change.line_num == 5
    assert change.original == "The test was run."
    assert change.revised == "We ran the test."
    assert change.annotation == "passive → active"


def test_diff_result_dataclass_fields():
    """DiffResult should have all required fields."""
    result = DiffResult()
    assert result.changes == []
    assert result.original_report is None
    assert result.revised_report is None

    result2 = DiffResult(
        changes=[DiffChange(1, "a", "b", "change")],
        original_report={"score": 0.5},
        revised_report={"score": 0.9},
    )
    assert len(result2.changes) == 1
    assert result2.original_report["score"] == 0.5
