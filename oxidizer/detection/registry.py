"""Detection registry for Oxidizer — combines all three detection layers.

Runs vocabulary, structural, and statistical detection and returns a single
:class:`DetectionReport` with an overall severity score and human-readable
summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from oxidizer.detection.vocabulary import (
    VocabFinding,
    Tier,
    scan_vocabulary,
)
from oxidizer.detection.structural import (
    StructuralFinding,
    detect_structural_patterns,
)
from oxidizer.detection.statistical import (
    StatisticalReport,
    analyze_statistical_signals,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Severity(IntEnum):
    CLEAN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DetectionReport:
    """Combined report from all three detection layers."""

    vocab_findings: list[VocabFinding]
    structural_findings: list[StructuralFinding]
    statistical_report: StatisticalReport
    overall_severity: Severity
    summary: str

    @property
    def vocab_findings_count(self) -> int:
        return len(self.vocab_findings)


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

def _compute_severity(
    vocab_findings: list[VocabFinding],
    structural_findings: list[StructuralFinding],
    statistical_report: StatisticalReport,
) -> Severity:
    """Compute overall severity from all detection layers.

    Score formula:
        score = p0_count * 3 + p1_count * 1 + struct_high * 2 + stat_flags * 2

    Thresholds:
        0      → CLEAN
        1-2    → LOW
        3-6    → MEDIUM
        7-12   → HIGH
        13+    → CRITICAL
    """
    # Only count non-exempt findings for P0/P1 in the score
    p0_count = sum(
        1 for f in vocab_findings
        if f.tier == Tier.P0 and not f.context_exempt
    )
    p1_count = sum(
        1 for f in vocab_findings
        if f.tier == Tier.P1 and not f.context_exempt
    )
    struct_high = sum(
        1 for f in structural_findings
        if f.severity == "high"
    )
    stat_flags = len(statistical_report.ai_risk_flags)

    score = p0_count * 3 + p1_count * 1 + struct_high * 2 + stat_flags * 2

    if score == 0:
        return Severity.CLEAN
    elif score <= 2:
        return Severity.LOW
    elif score <= 6:
        return Severity.MEDIUM
    elif score <= 12:
        return Severity.HIGH
    else:
        return Severity.CRITICAL


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    severity: Severity,
    vocab_findings: list[VocabFinding],
    structural_findings: list[StructuralFinding],
    statistical_report: StatisticalReport,
) -> str:
    """Build a human-readable summary string for the detection report."""
    if severity == Severity.CLEAN:
        return "No AI patterns detected."

    parts: list[str] = []

    # P0 vocabulary terms (non-exempt)
    p0_terms = [
        f.term for f in vocab_findings
        if f.tier == Tier.P0 and not f.context_exempt
    ]
    if p0_terms:
        parts.append(f"P0 vocabulary: {', '.join(p0_terms)}")

    # P1 vocabulary terms (non-exempt) — only include if triggered
    p1_terms = [
        f.term for f in vocab_findings
        if f.tier == Tier.P1 and not f.context_exempt
    ]
    if p1_terms:
        parts.append(f"P1 vocabulary: {', '.join(p1_terms)}")

    # Structural findings
    for sf in structural_findings:
        parts.append(f"{sf.pattern.value}: {sf.description}")

    # Statistical flags
    for flag in statistical_report.ai_risk_flags:
        # Condense to a short label with the burstiness value when relevant
        if "burstiness" in flag or "metronomic" in flag.lower():
            parts.append(f"Low burstiness ({statistical_report.burstiness:.2f})")
        else:
            parts.append(flag)

    detail = "; ".join(parts)
    return f"AI Detection [{severity.name}]: {detail}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_full_detection(
    text: str,
    context: str | None = None,
) -> DetectionReport:
    """Run all three detection layers and return a combined :class:`DetectionReport`.

    Parameters
    ----------
    text:
        The academic text to analyse.
    context:
        Optional section label (e.g. ``"methods"``, ``"results"``).
        Passed to the vocabulary scanner to apply context-based exemptions.

    Returns
    -------
    DetectionReport
        Populated with findings from all layers, an overall severity, and a
        human-readable summary.
    """
    vocab_findings = scan_vocabulary(text, context=context)
    structural_findings = detect_structural_patterns(text)
    statistical_report = analyze_statistical_signals(text)

    severity = _compute_severity(vocab_findings, structural_findings, statistical_report)
    summary = _build_summary(severity, vocab_findings, structural_findings, statistical_report)

    return DetectionReport(
        vocab_findings=vocab_findings,
        structural_findings=structural_findings,
        statistical_report=statistical_report,
        overall_severity=severity,
        summary=summary,
    )
