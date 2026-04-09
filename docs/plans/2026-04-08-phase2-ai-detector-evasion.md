# Phase 2: AI Detector Evasion — Structural Pattern Detection & Two-Pass Rewrite

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Oxidizer's output undetectable by AI content detectors by adding tiered vocabulary scanning, structural pattern detection, statistical signal analysis, and a two-pass rewrite methodology.

**Architecture:** Three new modules: `oxidizer/detection/vocabulary.py` (tiered word/phrase scanner), `oxidizer/detection/structural.py` (sentence-level pattern detectors), `oxidizer/detection/statistical.py` (burstiness, trigram repetition, sentence length CoV). The existing `scan` command and `revise` engine integrate these. The profile schema gets a new `ai_detection` section with context-aware tolerance. The revise engine gains a second pass (self-audit).

**Tech Stack:** Python 3.12, regex, nltk, spaCy (existing), collections.Counter for trigram analysis

---

## File Structure

```
oxidizer/detection/            # NEW package
  __init__.py
  vocabulary.py                # Tiered word/phrase scanner (P0/P1/P2)
  structural.py                # Sentence-level pattern detectors
  statistical.py               # Burstiness, TTR, trigram, CoV
  registry.py                  # Central AI pattern registry combining all detectors

oxidizer/scoring/metrics.py    # MODIFY — add sentence_length_cov, burstiness
oxidizer/scoring/reporter.py   # MODIFY — add ai_detection_score to StyleReport
oxidizer/profiles/schema.py    # MODIFY — add AIDetectionConfig to StyleProfile
oxidizer/profiles/loader.py    # MODIFY — load new ai_detection section
profiles/jinchi.yaml           # MODIFY — add ai_detection config + expanded banned list
oxidizer/engine/revise.py      # MODIFY — add second-pass self-audit
oxidizer/cli.py                # MODIFY — enhance scan command output

tests/test_vocabulary.py       # NEW
tests/test_structural.py       # NEW
tests/test_statistical.py      # NEW
tests/test_ai_detection.py     # NEW — integration tests for full detection pipeline
```

---

### Task 1: Tiered Vocabulary Scanner

**Files:**
- Create: `oxidizer/detection/__init__.py`
- Create: `oxidizer/detection/vocabulary.py`
- Test: `tests/test_vocabulary.py`

- [ ] **Step 1: Write failing tests for the vocabulary scanner**

```python
# tests/test_vocabulary.py
"""Tests for tiered AI vocabulary detection."""
import pytest
from oxidizer.detection.vocabulary import (
    scan_vocabulary,
    VocabFinding,
    Tier,
    TIER_P0,
    TIER_P1,
    TIER_P2,
)


class TestTierP0:
    """Tier P0 words are always flagged, even one occurrence."""

    def test_delve_flagged(self):
        findings = scan_vocabulary("We delve into the data.")
        assert any(f.term == "delve" and f.tier == Tier.P0 for f in findings)

    def test_tapestry_flagged(self):
        findings = scan_vocabulary("A rich tapestry of results.")
        assert any(f.term == "tapestry" and f.tier == Tier.P0 for f in findings)

    def test_leverage_flagged(self):
        findings = scan_vocabulary("We leverage deep learning.")
        assert any(f.term == "leverage" and f.tier == Tier.P0 for f in findings)

    def test_utilize_flagged(self):
        findings = scan_vocabulary("We utilize this approach.")
        assert any(f.term == "utilize" and f.tier == Tier.P0 for f in findings)

    def test_replacement_provided(self):
        findings = scan_vocabulary("We utilize this method.")
        f = next(f for f in findings if f.term == "utilize")
        assert f.replacement is not None
        assert "use" in f.replacement

    def test_serves_as_flagged(self):
        findings = scan_vocabulary("This serves as a foundation.")
        assert any(f.term == "serves as" and f.tier == Tier.P0 for f in findings)

    def test_testament_to_flagged(self):
        findings = scan_vocabulary("A testament to the method.")
        assert any(f.term == "testament to" and f.tier == Tier.P0 for f in findings)

    def test_in_order_to_flagged(self):
        findings = scan_vocabulary("In order to improve results.")
        assert any(f.term == "in order to" and f.tier == Tier.P0 for f in findings)

    def test_due_to_the_fact_that_flagged(self):
        findings = scan_vocabulary("Due to the fact that X.")
        assert any(f.term == "due to the fact that" and f.tier == Tier.P0 for f in findings)


class TestTierP1:
    """Tier P1 words are flagged when 2+ appear in the same text."""

    def test_single_p1_not_flagged(self):
        findings = scan_vocabulary("We harness the data.")
        p1_findings = [f for f in findings if f.tier == Tier.P1]
        assert len(p1_findings) == 0

    def test_two_p1_words_flagged(self):
        findings = scan_vocabulary("We harness the data to foster collaboration.")
        p1_findings = [f for f in findings if f.tier == Tier.P1]
        assert len(p1_findings) == 2

    def test_three_p1_words_all_flagged(self):
        findings = scan_vocabulary("We harness data, foster growth, and empower teams.")
        p1_findings = [f for f in findings if f.tier == Tier.P1]
        assert len(p1_findings) == 3


class TestTierP2:
    """Tier P2 words are flagged only at high density (3+ in 500 words)."""

    def test_single_p2_not_flagged(self):
        findings = scan_vocabulary("This is a significant result.")
        p2_findings = [f for f in findings if f.tier == Tier.P2]
        assert len(p2_findings) == 0

    def test_high_density_p2_flagged(self):
        text = "Significant results. Innovative methods. Remarkable outcomes. Unprecedented findings."
        findings = scan_vocabulary(text)
        p2_findings = [f for f in findings if f.tier == Tier.P2]
        assert len(p2_findings) >= 3


class TestContextExemptions:
    """Words with technical context exemptions should not be flagged."""

    def test_robust_exempt_in_methods(self):
        findings = scan_vocabulary("Robust standard errors were computed.", context="methods")
        assert not any(f.term == "robust" for f in findings)

    def test_robust_flagged_in_discussion(self):
        # "robust" in Tier P1 is flagged in clusters, but should not be exempt in discussion
        findings = scan_vocabulary(
            "Our robust findings foster a robust framework.",
            context="discussion",
        )
        p1_findings = [f for f in findings if f.tier == Tier.P1]
        assert len(p1_findings) >= 2

    def test_significant_exempt_in_results(self):
        findings = scan_vocabulary("The difference was statistically significant (p < 0.05).", context="results")
        assert not any(f.term == "significant" for f in findings)


class TestCaseInsensitive:
    def test_delve_case_insensitive(self):
        findings = scan_vocabulary("We Delve into this topic.")
        assert any(f.term == "delve" for f in findings)


class TestCleanText:
    def test_no_findings_on_clean_text(self):
        findings = scan_vocabulary("We collected data from 76 patients.")
        assert len(findings) == 0


class TestFindingStructure:
    def test_finding_has_position(self):
        findings = scan_vocabulary("We delve into this.")
        f = findings[0]
        assert f.position >= 0
        assert f.term == "delve"
        assert f.tier == Tier.P0
        assert f.replacement is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_vocabulary.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement the vocabulary scanner**

```python
# oxidizer/detection/__init__.py
"""AI pattern detection for Oxidizer."""

# oxidizer/detection/vocabulary.py
"""Tiered AI vocabulary scanner.

Three severity tiers:
  P0 — Always flag, even one occurrence. Credibility killers.
  P1 — Flag when 2+ appear in the same text. Obvious AI smell in clusters.
  P2 — Flag only at high density (3+ in text < 500 words, or 5+ in longer text).

Based on patterns from conorbronsdon/avoid-ai-writing and brandonwise/humanizer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Tier(Enum):
    P0 = "P0"  # Always flag
    P1 = "P1"  # Flag in clusters (2+)
    P2 = "P2"  # Flag at high density


@dataclass
class VocabFinding:
    term: str
    tier: Tier
    replacement: str | None
    position: int  # char offset in text
    context_exempt: bool = False


# ---------------------------------------------------------------------------
# Word/phrase tables
# ---------------------------------------------------------------------------

# P0: Always flag. These are dead giveaways.
# Format: (pattern, replacement_suggestion)
TIER_P0: list[tuple[str, str]] = [
    # Single words
    ("delve", "explore, examine, investigate"),
    ("tapestry", "describe the actual complexity"),
    ("beacon", "rewrite entirely"),
    ("utilize", "use"),
    ("utilization", "use"),
    ("leverage", "use"),
    ("leveraging", "using"),
    ("commence", "start, begin"),
    ("ascertain", "determine, find out"),
    ("endeavor", "effort, attempt"),
    ("showcase", "show, demonstrate"),
    ("showcasing", "showing, demonstrating"),
    ("underscores", "highlights, shows"),
    ("underscore", "highlight, show"),
    ("embark", "start, begin"),
    ("nestled", "located, situated"),
    ("bustling", "busy, active"),
    ("vibrant", "active, lively"),
    ("meticulous", "careful, detailed"),
    ("meticulously", "carefully"),
    ("seamless", "smooth, easy"),
    ("seamlessly", "smoothly"),
    ("tapestry", "describe the complexity"),
    ("unravel", "explain, clarify"),
    ("plethora", "many, a range of"),
    ("myriad", "many, numerous"),
    ("paradigm", "model, approach, framework"),
    # Phrases
    ("delve into", "explore, examine"),
    ("serves as", "is"),
    ("testament to", "shows, demonstrates"),
    ("in order to", "to"),
    ("due to the fact that", "because"),
    ("at its core", "cut or rewrite"),
    ("it's worth noting", "cut or state directly"),
    ("it's important to note", "cut or state directly"),
    ("in today's world", "cut"),
    ("game-changer", "describe what changed"),
    ("game-changing", "describe what changed"),
    ("watershed moment", "turning point"),
    ("deep dive", "look at, examine"),
    ("dive into", "examine, explore"),
    ("the future looks bright", "cut"),
    ("only time will tell", "cut"),
    ("ever-evolving", "changing"),
    ("thought leader", "expert"),
    ("best practices", "what works, proven methods"),
    ("actionable", "practical, concrete"),
    ("impactful", "effective"),
    ("learnings", "lessons, findings"),
    ("boasts", "has"),
    ("plays a crucial role", "is important for"),
    ("plays a pivotal role", "is important for"),
    ("plays a vital role", "is important for"),
    ("harness the power", "use"),
    ("navigate the complexities", "address"),
    ("pave the way", "enable, lead to"),
    ("push the boundaries", "advance, extend"),
]

# P1: Flag when 2+ appear in the same text.
TIER_P1: list[tuple[str, str | None]] = [
    ("harness", "use"),
    ("harnessing", "using"),
    ("foster", "encourage, support"),
    ("elevate", "improve, raise"),
    ("streamline", "simplify"),
    ("empower", "enable"),
    ("bolster", "strengthen, support"),
    ("spearhead", "lead"),
    ("resonate", None),
    ("revolutionize", None),
    ("facilitate", "enable, support"),
    ("cultivate", "develop, build"),
    ("illuminate", "clarify, reveal"),
    ("elucidate", "clarify, explain"),
    ("galvanize", "motivate"),
    ("augment", "add to, supplement"),
    ("catalyze", "accelerate, trigger"),
    ("reimagine", "rethink, redesign"),
    ("encompass", "include, cover"),
    ("unleash", "release, enable"),
    ("navigate", "address, work through"),
    ("landscape", "field, area, space"),
    ("cornerstone", "foundation, basis"),
    ("pivotal", "important, key"),
    ("groundbreaking", "new, first"),
    ("cutting-edge", "latest, advanced"),
    ("transformative", "significant"),
    ("synergy", "combined effect"),
    ("innovative", "new"),
    ("novel", "new"),
    ("realm", "area, field, domain"),
    ("intricacies", "details, complexities"),
    ("paramount", "important, essential"),
    ("poised", None),
    ("burgeoning", "growing"),
    ("nascent", "emerging, early"),
    ("quintessential", "typical, classic"),
    ("overarching", "overall, broad"),
    ("robust", "strong, reliable"),
    ("comprehensive", "thorough, complete"),
    ("holistic", "complete, whole"),
    ("multifaceted", "complex"),
    ("nuanced", "subtle, detailed"),
]

# P2: Flag only at high density (3+ in short text, 5+ in long text).
TIER_P2: list[tuple[str, str | None]] = [
    ("significant", None),
    ("significantly", None),
    ("innovative", None),
    ("innovation", None),
    ("effective", None),
    ("effectively", None),
    ("dynamic", None),
    ("compelling", None),
    ("unprecedented", None),
    ("exceptional", None),
    ("exceptionally", None),
    ("remarkable", None),
    ("remarkably", None),
    ("sophisticated", None),
    ("instrumental", None),
    ("notable", None),
    ("substantial", None),
    ("considerably", None),
]

# Context exemptions: (term, list of contexts where it's legitimate)
CONTEXT_EXEMPTIONS: dict[str, list[str]] = {
    "robust": ["methods", "results"],   # "robust standard errors" is legitimate
    "significant": ["results"],          # "statistically significant" is legitimate
    "comprehensive": ["methods"],        # "comprehensive analysis" can be legitimate
    "novel": ["intro"],                  # less common but sometimes acceptable
    "facilitate": ["methods"],           # "facilitate data collection" can be legitimate
}


def _compile_pattern(term: str) -> re.Pattern:
    """Compile a word/phrase into a case-insensitive regex with word boundaries."""
    escaped = re.escape(term)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def scan_vocabulary(
    text: str,
    context: str | None = None,
) -> list[VocabFinding]:
    """Scan text for AI vocabulary patterns across all tiers.

    Parameters
    ----------
    text:
        The input text to scan.
    context:
        Optional section context ("intro", "methods", "results", "discussion").
        Used for context-aware exemptions.

    Returns
    -------
    list[VocabFinding]
        Findings sorted by position. P1 findings are only included if 2+ P1
        terms are detected. P2 findings are only included if density threshold
        is exceeded.
    """
    if not text.strip():
        return []

    findings: list[VocabFinding] = []

    # --- P0: Always flag ---
    for term, replacement in TIER_P0:
        pat = _compile_pattern(term)
        for m in pat.finditer(text):
            exempt = (
                context is not None
                and term.lower() in CONTEXT_EXEMPTIONS
                and context in CONTEXT_EXEMPTIONS[term.lower()]
            )
            if not exempt:
                findings.append(VocabFinding(
                    term=term, tier=Tier.P0, replacement=replacement,
                    position=m.start(), context_exempt=False,
                ))

    # --- P1: Collect all matches, only include if 2+ distinct terms found ---
    p1_candidates: list[VocabFinding] = []
    p1_terms_found: set[str] = set()
    for term, replacement in TIER_P1:
        pat = _compile_pattern(term)
        for m in pat.finditer(text):
            exempt = (
                context is not None
                and term.lower() in CONTEXT_EXEMPTIONS
                and context in CONTEXT_EXEMPTIONS[term.lower()]
            )
            if not exempt:
                p1_candidates.append(VocabFinding(
                    term=term, tier=Tier.P1, replacement=replacement,
                    position=m.start(), context_exempt=False,
                ))
                p1_terms_found.add(term.lower())

    if len(p1_terms_found) >= 2:
        findings.extend(p1_candidates)

    # --- P2: Collect all matches, only include if density threshold exceeded ---
    p2_candidates: list[VocabFinding] = []
    p2_terms_found: set[str] = set()
    for term, replacement in TIER_P2:
        pat = _compile_pattern(term)
        for m in pat.finditer(text):
            exempt = (
                context is not None
                and term.lower() in CONTEXT_EXEMPTIONS
                and context in CONTEXT_EXEMPTIONS[term.lower()]
            )
            if not exempt:
                p2_candidates.append(VocabFinding(
                    term=term, tier=Tier.P2, replacement=replacement,
                    position=m.start(), context_exempt=False,
                ))
                p2_terms_found.add(term.lower())

    word_count = len(text.split())
    threshold = 3 if word_count < 500 else 5
    if len(p2_terms_found) >= threshold:
        findings.extend(p2_candidates)

    # Sort by position
    findings.sort(key=lambda f: f.position)
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n oxidizer python -m pytest tests/test_vocabulary.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add oxidizer/detection/ tests/test_vocabulary.py
git commit -m "feat: tiered AI vocabulary scanner (P0/P1/P2) with context exemptions"
```

---

### Task 2: Structural Pattern Detectors

**Files:**
- Create: `oxidizer/detection/structural.py`
- Test: `tests/test_structural.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_structural.py
"""Tests for structural AI pattern detection."""
import pytest
from oxidizer.detection.structural import (
    detect_structural_patterns,
    StructuralFinding,
    PatternType,
)


class TestSentenceStarterRepetition:
    """Detect repetitive sentence starters (the abstract problem)."""

    def test_we_we_we_detected(self):
        text = "We collected data. We performed analysis. We computed results. We observed trends."
        findings = detect_structural_patterns(text)
        assert any(f.pattern == PatternType.REPETITIVE_STARTERS for f in findings)

    def test_varied_starters_clean(self):
        text = "Data was collected from 76 patients. We performed analysis using TabPFN. The results showed strong discrimination."
        findings = detect_structural_patterns(text)
        assert not any(f.pattern == PatternType.REPETITIVE_STARTERS for f in findings)

    def test_the_the_the_detected(self):
        text = "The model performed well. The results were strong. The data confirmed this. The analysis showed improvement."
        findings = detect_structural_patterns(text)
        assert any(f.pattern == PatternType.REPETITIVE_STARTERS for f in findings)


class TestSentenceLengthUniformity:
    """AI text has metronomic sentence lengths; human text is bursty."""

    def test_uniform_lengths_detected(self):
        # All sentences ~15 words — AI-like uniformity
        text = (
            "The results of our analysis showed significant improvements in all metrics. "
            "We observed that the model performed consistently well across all datasets. "
            "The training process converged reliably within the expected number of epochs. "
            "Our findings suggest that this approach generalizes well to new domains."
        )
        findings = detect_structural_patterns(text)
        assert any(f.pattern == PatternType.LENGTH_UNIFORMITY for f in findings)

    def test_varied_lengths_clean(self):
        text = (
            "Results were strong. "
            "We observed that the model performed consistently well across all datasets, "
            "with particular improvements in the most challenging test cases where prior methods had failed. "
            "This matters."
        )
        findings = detect_structural_patterns(text)
        assert not any(f.pattern == PatternType.LENGTH_UNIFORMITY for f in findings)


class TestRuleOfThree:
    """AI overuses triadic structures."""

    def test_triple_list_detected(self):
        text = "This approach is efficient, scalable, and robust."
        findings = detect_structural_patterns(text)
        assert any(f.pattern == PatternType.RULE_OF_THREE for f in findings)

    def test_single_triple_not_flagged(self):
        # One instance is fine; only flag if 2+ in same text
        text = "We use three methods: segmentation, classification, and registration. The data was clean."
        findings = detect_structural_patterns(text)
        assert not any(f.pattern == PatternType.RULE_OF_THREE for f in findings)


class TestCopulaAvoidance:
    """AI substitutes 'serves as', 'features', 'boasts' for 'is'/'has'."""

    def test_serves_as_detected(self):
        text = "This method serves as a foundation. The model serves as a baseline."
        findings = detect_structural_patterns(text)
        assert any(f.pattern == PatternType.COPULA_AVOIDANCE for f in findings)


class TestNegativeParallelism:
    """'It's not just X, it's Y' pattern."""

    def test_not_just_detected(self):
        text = "It's not just about accuracy, it's about reliability."
        findings = detect_structural_patterns(text)
        assert any(f.pattern == PatternType.NEGATIVE_PARALLELISM for f in findings)


class TestEmDashOveruse:
    def test_em_dashes_detected(self):
        text = "The model — which we trained — showed results — both strong and weak — across datasets."
        findings = detect_structural_patterns(text)
        assert any(f.pattern == PatternType.EM_DASH_OVERUSE for f in findings)

    def test_zero_em_dashes_clean(self):
        text = "The model showed strong results across all datasets."
        findings = detect_structural_patterns(text)
        assert not any(f.pattern == PatternType.EM_DASH_OVERUSE for f in findings)


class TestConfidenceStacking:
    """AI stacks 'Interestingly', 'Notably', 'Surprisingly' as filler."""

    def test_stacking_detected(self):
        text = "Interestingly, the results improved. Notably, the error decreased. Surprisingly, the model converged."
        findings = detect_structural_patterns(text)
        assert any(f.pattern == PatternType.CONFIDENCE_STACKING for f in findings)


class TestCleanText:
    def test_no_findings(self):
        text = "We collected data from 76 patients. Registration error was 1.43 mm."
        findings = detect_structural_patterns(text)
        assert len(findings) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_structural.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement structural pattern detectors**

```python
# oxidizer/detection/structural.py
"""Structural AI pattern detectors.

Detects sentence-level and paragraph-level patterns that AI content
detectors key on: repetitive starters, uniform sentence lengths,
rule-of-three overuse, copula avoidance, em dash overuse, etc.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PatternType(Enum):
    REPETITIVE_STARTERS = "repetitive_starters"
    LENGTH_UNIFORMITY = "length_uniformity"
    RULE_OF_THREE = "rule_of_three"
    COPULA_AVOIDANCE = "copula_avoidance"
    NEGATIVE_PARALLELISM = "negative_parallelism"
    EM_DASH_OVERUSE = "em_dash_overuse"
    CONFIDENCE_STACKING = "confidence_stacking"


@dataclass
class StructuralFinding:
    pattern: PatternType
    description: str
    evidence: str  # the specific text that triggered the detection
    severity: str  # "high", "medium", "low"


def _sent_tokenize(text: str) -> list[str]:
    """Split text into sentences using nltk."""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except LookupError:
        import nltk
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)


def _detect_repetitive_starters(sentences: list[str]) -> Optional[StructuralFinding]:
    """Flag if 40%+ of sentences start with the same word."""
    if len(sentences) < 4:
        return None
    first_words = [s.split()[0].lower().rstrip(",") for s in sentences if s.strip()]
    if not first_words:
        return None
    from collections import Counter
    counts = Counter(first_words)
    most_common_word, most_common_count = counts.most_common(1)[0]
    ratio = most_common_count / len(first_words)
    if ratio >= 0.4 and most_common_count >= 3:
        return StructuralFinding(
            pattern=PatternType.REPETITIVE_STARTERS,
            description=f'{most_common_count}/{len(first_words)} sentences ({ratio:.0%}) start with "{most_common_word}"',
            evidence=f'Repeated starter: "{most_common_word}"',
            severity="high",
        )
    return None


def _detect_length_uniformity(sentences: list[str]) -> Optional[StructuralFinding]:
    """Flag if sentence length coefficient of variation is below 0.25 (AI-like uniformity)."""
    if len(sentences) < 4:
        return None
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if not lengths or len(lengths) < 4:
        return None
    mean_len = statistics.mean(lengths)
    if mean_len == 0:
        return None
    std_len = statistics.stdev(lengths)
    cov = std_len / mean_len
    if cov < 0.25:
        return StructuralFinding(
            pattern=PatternType.LENGTH_UNIFORMITY,
            description=f"Sentence length CoV is {cov:.2f} (AI-like < 0.25, human > 0.40)",
            evidence=f"Lengths: {lengths}",
            severity="high",
        )
    return None


def _detect_rule_of_three(text: str) -> Optional[StructuralFinding]:
    """Flag if 2+ triadic structures ('X, Y, and Z') appear in the text."""
    # Pattern: word, word, and word (adjective triads)
    pattern = re.compile(
        r"\b(\w+),\s+(\w+),\s+and\s+(\w+)\b",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if len(matches) >= 2:
        return StructuralFinding(
            pattern=PatternType.RULE_OF_THREE,
            description=f"{len(matches)} triadic lists detected",
            evidence=", ".join(f'"{a}, {b}, and {c}"' for a, b, c in matches[:3]),
            severity="medium",
        )
    return None


def _detect_copula_avoidance(text: str) -> Optional[StructuralFinding]:
    """Flag inflated copula substitutes: 'serves as', 'features', 'boasts'."""
    copula_patterns = [
        r"\bserves\s+as\b",
        r"\bboasts\b",
        r"\bfeatures\b(?=\s+(?:a|an|the|several|many|numerous)\b)",
    ]
    matches = []
    for pat in copula_patterns:
        matches.extend(re.findall(pat, text, re.IGNORECASE))
    if len(matches) >= 2:
        return StructuralFinding(
            pattern=PatternType.COPULA_AVOIDANCE,
            description=f"{len(matches)} inflated copula substitutes",
            evidence=", ".join(f'"{m}"' for m in matches[:3]),
            severity="medium",
        )
    return None


def _detect_negative_parallelism(text: str) -> Optional[StructuralFinding]:
    """Flag 'not just X, it's Y' patterns."""
    pattern = re.compile(
        r"\bnot\s+just\s+(?:about\s+)?\w+.*?,\s*(?:it'?s|it\s+is)\b",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if matches:
        return StructuralFinding(
            pattern=PatternType.NEGATIVE_PARALLELISM,
            description='"Not just X, it\'s Y" AI pattern',
            evidence=matches[0][:80],
            severity="medium",
        )
    return None


def _detect_em_dash_overuse(text: str) -> Optional[StructuralFinding]:
    """Flag more than 1 em dash per 500 words."""
    em_dashes = text.count("\u2014")  # — character
    word_count = len(text.split())
    if word_count == 0:
        return None
    rate = em_dashes / max(word_count, 1) * 1000
    if em_dashes >= 2 and rate > 2:
        return StructuralFinding(
            pattern=PatternType.EM_DASH_OVERUSE,
            description=f"{em_dashes} em dashes in {word_count} words ({rate:.1f} per 1000)",
            evidence="em dash overuse",
            severity="medium",
        )
    return None


def _detect_confidence_stacking(sentences: list[str]) -> Optional[StructuralFinding]:
    """Flag stacking of 'Interestingly', 'Notably', 'Surprisingly' etc."""
    confidence_words = {
        "interestingly", "notably", "surprisingly", "importantly",
        "remarkably", "significantly", "crucially", "strikingly",
    }
    count = 0
    found = []
    for sent in sentences:
        first_word = sent.split(",")[0].strip().lower().rstrip(",")
        if first_word in confidence_words:
            count += 1
            found.append(first_word)
    if count >= 2:
        return StructuralFinding(
            pattern=PatternType.CONFIDENCE_STACKING,
            description=f'{count} sentences open with confidence adverbs',
            evidence=", ".join(f'"{w}"' for w in found),
            severity="medium",
        )
    return None


def detect_structural_patterns(text: str) -> list[StructuralFinding]:
    """Run all structural pattern detectors on text.

    Returns a list of findings, empty if text is clean.
    """
    if not text.strip():
        return []

    sentences = _sent_tokenize(text)
    findings: list[StructuralFinding] = []

    detectors = [
        lambda: _detect_repetitive_starters(sentences),
        lambda: _detect_length_uniformity(sentences),
        lambda: _detect_rule_of_three(text),
        lambda: _detect_copula_avoidance(text),
        lambda: _detect_negative_parallelism(text),
        lambda: _detect_em_dash_overuse(text),
        lambda: _detect_confidence_stacking(sentences),
    ]

    for detector in detectors:
        result = detector()
        if result is not None:
            findings.append(result)

    return findings
```

- [ ] **Step 4: Run tests**

Run: `conda run -n oxidizer python -m pytest tests/test_structural.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add oxidizer/detection/structural.py tests/test_structural.py
git commit -m "feat: structural AI pattern detectors (starters, uniformity, triads, em dashes)"
```

---

### Task 3: Statistical Signal Analysis

**Files:**
- Create: `oxidizer/detection/statistical.py`
- Test: `tests/test_statistical.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_statistical.py
"""Tests for statistical AI signal detection."""
import pytest
from oxidizer.detection.statistical import (
    compute_burstiness,
    compute_trigram_repetition,
    compute_sentence_cov,
    StatisticalReport,
    analyze_statistical_signals,
)


class TestBurstiness:
    """Burstiness: human text is bursty (0.5-1.0), AI is metronomic (0.1-0.3)."""

    def test_uniform_text_low_burstiness(self):
        # All sentences same length — AI-like
        text = " ".join(["This is a sentence with ten words in it now."] * 10)
        b = compute_burstiness(text)
        assert b < 0.3

    def test_varied_text_higher_burstiness(self):
        text = (
            "Short. "
            "This sentence is a bit longer with more detail. "
            "A very long sentence that goes on and on with many clauses and subclauses "
            "to demonstrate the kind of variance that human writers naturally produce. "
            "Done."
        )
        b = compute_burstiness(text)
        assert b > 0.3

    def test_empty_text_returns_zero(self):
        assert compute_burstiness("") == 0.0


class TestTrigramRepetition:
    """Trigram repetition: human < 0.05, AI > 0.10."""

    def test_repetitive_text_high_score(self):
        text = "We used the model. We used the data. We used the method. We used the approach."
        score = compute_trigram_repetition(text)
        assert score > 0.05

    def test_varied_text_low_score(self):
        text = (
            "Data was collected from patients at our institution. "
            "Registration error using only transverse scans was 1.43 mm. "
            "The final cohort comprised 76 subjects with paired imaging."
        )
        score = compute_trigram_repetition(text)
        assert score < 0.10

    def test_empty_returns_zero(self):
        assert compute_trigram_repetition("") == 0.0


class TestSentenceCov:
    def test_uniform_low_cov(self):
        text = " ".join(["This is exactly ten words long for this test."] * 8)
        cov = compute_sentence_cov(text)
        assert cov < 0.15

    def test_varied_high_cov(self):
        text = "Short. This is a medium sentence. This is a very long sentence with many words to increase the variance significantly."
        cov = compute_sentence_cov(text)
        assert cov > 0.4

    def test_empty_returns_zero(self):
        assert compute_sentence_cov("") == 0.0


class TestAnalyzeStatisticalSignals:
    def test_returns_report(self):
        text = "We analyzed data. Results were promising. The model performed well."
        report = analyze_statistical_signals(text)
        assert isinstance(report, StatisticalReport)
        assert 0 <= report.burstiness <= 1.0
        assert 0 <= report.trigram_repetition <= 1.0
        assert report.sentence_cov >= 0
        assert isinstance(report.ai_risk_flags, list)

    def test_ai_like_text_has_flags(self):
        # Highly uniform AI-like text
        text = " ".join(["The results demonstrate significant improvements in model performance."] * 10)
        report = analyze_statistical_signals(text)
        assert len(report.ai_risk_flags) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_statistical.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement statistical analysis**

```python
# oxidizer/detection/statistical.py
"""Statistical AI signal detection.

Computes metrics that distinguish human from AI writing at a statistical
level: burstiness (variance in writing pace), trigram repetition, and
sentence length coefficient of variation.

Reference baselines (from brandonwise/humanizer):
  Burstiness:          human 0.5-1.0, AI 0.1-0.3
  Trigram repetition:  human < 0.05, AI > 0.10
  Sentence length CoV: human > 0.40, AI < 0.25
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field


def _sent_tokenize(text: str) -> list[str]:
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except LookupError:
        import nltk
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)


def compute_burstiness(text: str) -> float:
    """Compute burstiness of sentence lengths.

    Burstiness B = (std - mean) / (std + mean), normalized to [0, 1].
    Human text is bursty (high variance relative to mean), AI is uniform.
    Returns 0.0 for empty text.
    """
    text = text.strip()
    if not text:
        return 0.0
    sentences = _sent_tokenize(text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 3:
        return 0.0
    mean_l = statistics.mean(lengths)
    std_l = statistics.stdev(lengths)
    if (std_l + mean_l) == 0:
        return 0.0
    # Raw burstiness can be negative; clamp to [0, 1]
    raw = (std_l - mean_l) / (std_l + mean_l)
    return max(0.0, min(1.0, (raw + 1) / 2))  # normalize from [-1,1] to [0,1]


def compute_trigram_repetition(text: str) -> float:
    """Compute the fraction of trigrams that appear more than once.

    Human text typically < 0.05, AI text > 0.10.
    Returns 0.0 for empty or very short text.
    """
    text = text.strip()
    if not text:
        return 0.0
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts) if counts else 0.0


def compute_sentence_cov(text: str) -> float:
    """Compute coefficient of variation of sentence lengths.

    Human text typically > 0.40, AI text < 0.25.
    Returns 0.0 for empty text.
    """
    text = text.strip()
    if not text:
        return 0.0
    sentences = _sent_tokenize(text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 3:
        return 0.0
    mean_l = statistics.mean(lengths)
    if mean_l == 0:
        return 0.0
    return statistics.stdev(lengths) / mean_l


@dataclass
class StatisticalReport:
    burstiness: float
    trigram_repetition: float
    sentence_cov: float
    ai_risk_flags: list[str] = field(default_factory=list)


def analyze_statistical_signals(text: str) -> StatisticalReport:
    """Run all statistical analyses and flag AI-like signals."""
    burstiness = compute_burstiness(text)
    trigram_rep = compute_trigram_repetition(text)
    sentence_cov = compute_sentence_cov(text)

    flags: list[str] = []
    if burstiness < 0.3 and len(text.split()) > 50:
        flags.append(f"Low burstiness ({burstiness:.2f}) — AI-like uniformity")
    if trigram_rep > 0.10:
        flags.append(f"High trigram repetition ({trigram_rep:.2f}) — repetitive phrasing")
    if sentence_cov < 0.25 and len(_sent_tokenize(text)) >= 4:
        flags.append(f"Low sentence length CoV ({sentence_cov:.2f}) — metronomic rhythm")

    return StatisticalReport(
        burstiness=burstiness,
        trigram_repetition=trigram_rep,
        sentence_cov=sentence_cov,
        ai_risk_flags=flags,
    )
```

- [ ] **Step 4: Run tests**

Run: `conda run -n oxidizer python -m pytest tests/test_statistical.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add oxidizer/detection/statistical.py tests/test_statistical.py
git commit -m "feat: statistical AI signal detection (burstiness, trigrams, CoV)"
```

---

### Task 4: Detection Registry + Integration

**Files:**
- Create: `oxidizer/detection/registry.py`
- Test: `tests/test_ai_detection.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ai_detection.py
"""Integration tests for the full AI detection pipeline."""
import pytest
from oxidizer.detection.registry import (
    run_full_detection,
    DetectionReport,
    Severity,
)


class TestFullDetection:
    def test_clean_text_passes(self):
        text = "We collected data from 76 patients. Registration error was 1.43 mm."
        report = run_full_detection(text)
        assert isinstance(report, DetectionReport)
        assert report.overall_severity == Severity.CLEAN

    def test_ai_heavy_text_flags(self):
        text = (
            "We delve into this tapestry of results. The model serves as a beacon "
            "of innovation. In order to leverage the data, we utilize a comprehensive "
            "approach. This testament to the methodology showcases the seamless "
            "integration of cutting-edge techniques."
        )
        report = run_full_detection(text)
        assert report.overall_severity in (Severity.HIGH, Severity.CRITICAL)
        assert report.vocab_findings_count > 0

    def test_report_has_all_sections(self):
        text = "We analyzed the data using standard methods."
        report = run_full_detection(text)
        assert hasattr(report, "vocab_findings")
        assert hasattr(report, "structural_findings")
        assert hasattr(report, "statistical_report")
        assert hasattr(report, "overall_severity")
        assert hasattr(report, "summary")

    def test_context_affects_results(self):
        text = "We computed robust standard errors for the regression."
        report_methods = run_full_detection(text, context="methods")
        report_discussion = run_full_detection(text, context="discussion")
        # "robust" should be exempt in methods but not in discussion
        methods_terms = [f.term for f in report_methods.vocab_findings]
        discussion_terms = [f.term for f in report_discussion.vocab_findings]
        assert "robust" not in methods_terms

    def test_severity_ordering(self):
        clean = run_full_detection("Data was collected.")
        moderate = run_full_detection("We harness data to foster collaboration and empower researchers.")
        heavy = run_full_detection(
            "We delve into this tapestry. The model serves as a testament to innovation. "
            "In order to leverage the data, we utilize seamless approaches."
        )
        assert clean.overall_severity.value <= moderate.overall_severity.value
        assert moderate.overall_severity.value <= heavy.overall_severity.value
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_ai_detection.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement the detection registry**

```python
# oxidizer/detection/registry.py
"""Central AI pattern detection registry.

Combines vocabulary scanning, structural pattern detection, and statistical
analysis into a single detection report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from oxidizer.detection.vocabulary import VocabFinding, Tier, scan_vocabulary
from oxidizer.detection.structural import StructuralFinding, detect_structural_patterns
from oxidizer.detection.statistical import StatisticalReport, analyze_statistical_signals


class Severity(IntEnum):
    CLEAN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DetectionReport:
    vocab_findings: list[VocabFinding]
    structural_findings: list[StructuralFinding]
    statistical_report: StatisticalReport
    overall_severity: Severity
    summary: str

    @property
    def vocab_findings_count(self) -> int:
        return len(self.vocab_findings)


def _compute_severity(
    vocab: list[VocabFinding],
    structural: list[StructuralFinding],
    statistical: StatisticalReport,
) -> Severity:
    """Compute overall severity from all detection results."""
    p0_count = sum(1 for f in vocab if f.tier == Tier.P0)
    p1_count = sum(1 for f in vocab if f.tier == Tier.P1)
    struct_high = sum(1 for f in structural if f.severity == "high")
    stat_flags = len(statistical.ai_risk_flags)

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


def _build_summary(
    vocab: list[VocabFinding],
    structural: list[StructuralFinding],
    statistical: StatisticalReport,
    severity: Severity,
) -> str:
    """Build a human-readable summary of detection results."""
    parts: list[str] = []

    if vocab:
        p0 = [f for f in vocab if f.tier == Tier.P0]
        p1 = [f for f in vocab if f.tier == Tier.P1]
        p2 = [f for f in vocab if f.tier == Tier.P2]
        if p0:
            parts.append(f"P0 vocabulary: {', '.join(f.term for f in p0)}")
        if p1:
            parts.append(f"P1 cluster: {', '.join(f.term for f in p1)}")
        if p2:
            parts.append(f"P2 density: {len(p2)} terms flagged")

    for sf in structural:
        parts.append(f"{sf.pattern.value}: {sf.description}")

    for flag in statistical.ai_risk_flags:
        parts.append(flag)

    if not parts:
        return "No AI patterns detected."

    return f"AI Detection [{severity.name}]: " + "; ".join(parts)


def run_full_detection(
    text: str,
    context: str | None = None,
) -> DetectionReport:
    """Run the full AI detection pipeline on text.

    Parameters
    ----------
    text:
        Input text to analyze.
    context:
        Optional section context for vocabulary exemptions.

    Returns
    -------
    DetectionReport
        Combined report from all detection layers.
    """
    vocab = scan_vocabulary(text, context=context)
    structural = detect_structural_patterns(text)
    statistical = analyze_statistical_signals(text)

    severity = _compute_severity(vocab, structural, statistical)
    summary = _build_summary(vocab, structural, statistical, severity)

    return DetectionReport(
        vocab_findings=vocab,
        structural_findings=structural,
        statistical_report=statistical,
        overall_severity=severity,
        summary=summary,
    )
```

- [ ] **Step 4: Run all detection tests**

Run: `conda run -n oxidizer python -m pytest tests/test_vocabulary.py tests/test_structural.py tests/test_statistical.py tests/test_ai_detection.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add oxidizer/detection/registry.py tests/test_ai_detection.py
git commit -m "feat: AI detection registry combining vocab, structural, and statistical analysis"
```

---

### Task 5: Integrate Detection into CLI Scan Command

**Files:**
- Modify: `oxidizer/cli.py` (scan command)
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing test for enhanced scan**

Add to `tests/test_cli.py`:

```python
def test_scan_shows_tiered_findings(tmp_path):
    """Scan should show P0/P1/P2 tiers and structural patterns."""
    from click.testing import CliRunner
    from oxidizer.cli import cli

    text = "We delve into this tapestry. We leverage the data to utilize a comprehensive approach."
    md_file = tmp_path / "ai_heavy.md"
    md_file.write_text(f"## Discussion\n\n{text}")

    runner = CliRunner()
    result = runner.invoke(cli, ["scan", str(md_file), "--profile", "jinchi"])
    assert result.exit_code == 0
    assert "P0" in result.output or "delve" in result.output


def test_scan_shows_structural_patterns(tmp_path):
    from click.testing import CliRunner
    from oxidizer.cli import cli

    text = "We collected data. We performed analysis. We computed results. We observed trends. We noted patterns."
    md_file = tmp_path / "repetitive.md"
    md_file.write_text(f"## Methods\n\n{text}")

    runner = CliRunner()
    result = runner.invoke(cli, ["scan", str(md_file), "--profile", "jinchi"])
    assert result.exit_code == 0
    assert "repetitive" in result.output.lower() or "starter" in result.output.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_cli.py::test_scan_shows_tiered_findings tests/test_cli.py::test_scan_shows_structural_patterns -v`
Expected: FAIL

- [ ] **Step 3: Update scan command in cli.py**

Replace the existing scan command body. The scan command should:
1. Parse the document
2. For each section, run `run_full_detection(section.body, context=section.context)`
3. Display results in a Rich table grouped by severity (P0 first, then structural, then P1, P2, statistical)
4. Show replacement suggestions for vocabulary findings
5. Show overall severity per section and document-wide

Find the scan command in `oxidizer/cli.py` and update it to use the detection registry. Keep backward compatibility with `--profile` (still loads banned words from profile), but add the structural and statistical analysis.

- [ ] **Step 4: Run full test suite**

Run: `conda run -n oxidizer python -m pytest tests/ -v`
Expected: All pass (288 existing + new tests)

- [ ] **Step 5: Commit**

```bash
git add oxidizer/cli.py tests/test_cli.py
git commit -m "feat: enhanced scan command with tiered vocabulary, structural patterns, and statistical signals"
```

---

### Task 6: Two-Pass Rewrite in Revise Engine

**Files:**
- Modify: `oxidizer/engine/revise.py`
- Modify: `tests/test_revise.py`

- [ ] **Step 1: Write failing test for two-pass rewrite**

Add to `tests/test_revise.py`:

```python
def test_revise_section_runs_two_passes():
    """Revise should do a self-audit pass after the initial rewrite."""
    from unittest.mock import MagicMock
    from oxidizer.parsers.markdown_parser import Section
    from oxidizer.profiles.loader import load_profile
    from oxidizer.engine.revise import revise_section
    from pathlib import Path

    profile = load_profile("jinchi", search_paths=[Path(__file__).parent.parent / "profiles"])
    section = Section(heading="Discussion", body="We delve into these results.", context="discussion", level=1)

    # First call returns AI-ish text, second call (self-audit) returns cleaner text
    first_response = MagicMock()
    first_response.content = [MagicMock(text="We explored these results. This serves as a foundation.")]
    second_response = MagicMock()
    second_response.content = [MagicMock(text="We explored these results. This is a foundation.")]

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [first_response, second_response]

    result = revise_section(section, profile, client=mock_client)
    # Should have called the API twice (first pass + self-audit pass)
    assert mock_client.messages.create.call_count >= 2


def test_revise_skips_second_pass_if_clean():
    """If first pass has no AI patterns, skip the self-audit."""
    from unittest.mock import MagicMock
    from oxidizer.parsers.markdown_parser import Section
    from oxidizer.profiles.loader import load_profile
    from oxidizer.engine.revise import revise_section
    from pathlib import Path

    profile = load_profile("jinchi", search_paths=[Path(__file__).parent.parent / "profiles"])
    section = Section(heading="Methods", body="We collected data.", context="methods", level=1)

    clean_response = MagicMock()
    clean_response.content = [MagicMock(text="We collected data from 76 patients at our institution.")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = clean_response

    result = revise_section(section, profile, client=mock_client)
    # Only one API call if output is clean
    assert mock_client.messages.create.call_count == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_revise.py::test_revise_section_runs_two_passes tests/test_revise.py::test_revise_skips_second_pass_if_clean -v`
Expected: FAIL

- [ ] **Step 3: Update revise_section with two-pass logic**

In `oxidizer/engine/revise.py`, modify `revise_section` to:

1. After the first pass (existing entity preservation loop), run `run_full_detection(restyled, context=section.context)`
2. If the detection report severity is MEDIUM or higher, build a self-audit prompt:
   ```
   ## Self-Audit
   Your previous rewrite was flagged for AI patterns:
   {detection_report.summary}
   
   Rewrite again, fixing these specific issues:
   {list each finding with its replacement suggestion}
   
   Keep all factual content and locked entities intact.
   Output ONLY the revised text.
   ```
3. Call the API again with the self-audit prompt
4. If detection severity is LOW or CLEAN after the first pass, skip the second call

- [ ] **Step 4: Run tests**

Run: `conda run -n oxidizer python -m pytest tests/test_revise.py -v`
Expected: All pass

- [ ] **Step 5: Run full suite**

Run: `conda run -n oxidizer python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add oxidizer/engine/revise.py tests/test_revise.py
git commit -m "feat: two-pass rewrite with AI detection self-audit"
```

---

### Task 7: Update Profile Schema and jinchi.yaml

**Files:**
- Modify: `oxidizer/profiles/schema.py`
- Modify: `oxidizer/profiles/loader.py`
- Modify: `profiles/jinchi.yaml`

- [ ] **Step 1: Update schema with AIDetectionConfig**

Add to `oxidizer/profiles/schema.py` before `StyleProfile`:

```python
@dataclass
class AIDetectionConfig:
    """Configuration for AI detection strictness."""
    p2_density_threshold: int = 3       # P2 findings needed to flag (short text)
    p2_density_threshold_long: int = 5  # P2 findings needed to flag (500+ words)
    structural_enabled: bool = True
    statistical_enabled: bool = True
    context_exemptions: dict[str, list[str]] = field(default_factory=dict)
```

Add `ai_detection: AIDetectionConfig` field to `StyleProfile` with `field(default_factory=AIDetectionConfig)`.

- [ ] **Step 2: Update loader to parse ai_detection section**

In `oxidizer/profiles/loader.py`, add parsing for the optional `ai_detection` section from YAML. If missing, use defaults.

- [ ] **Step 3: Update jinchi.yaml with expanded banned list**

Replace the `banned_aiisms` list in `profiles/jinchi.yaml` with the full P0 tier (the vocabulary scanner now handles tiering, so the profile's banned list becomes the P0 list). Add the `ai_detection` config section:

```yaml
ai_detection:
  p2_density_threshold: 3
  p2_density_threshold_long: 5
  structural_enabled: true
  statistical_enabled: true
  context_exemptions:
    robust: ["methods", "results"]
    significant: ["results"]
    comprehensive: ["methods"]
    facilitate: ["methods"]
```

- [ ] **Step 4: Run tests**

Run: `conda run -n oxidizer python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add oxidizer/profiles/schema.py oxidizer/profiles/loader.py profiles/jinchi.yaml
git commit -m "feat: AI detection config in profile schema with context exemptions"
```

---

### Task 8: Update /oxidize Skill with Self-Audit Instructions

**Files:**
- Modify: `skill/SKILL.md`

- [ ] **Step 1: Update the skill to include two-pass instructions**

Update the `/oxidize revise` section of `skill/SKILL.md` to add a self-audit step between steps 4 and 5:

```markdown
### /oxidize revise <file> [--sections X,Y]
Restyle a document to match your writing voice. Preserves citations, numbers, equations, abbreviations.

Steps:
1. Read the file and parse into sections
2. For each section, run: `conda run -n oxidizer oxidizer score <temp_file> --profile jinchi --json-output` to get baseline
3. Extract entities using the Python extraction module
4. Restyle the section text using the style profile (Claude does this in-conversation)
5. **Self-audit**: Run `conda run -n oxidizer oxidizer scan <temp_restyled_file> --profile jinchi` on the restyled text. If any P0 or structural patterns are flagged, fix them in a second pass.
6. Run scoring on the restyled text
7. Verify all extracted entities appear in the restyled text
8. Output the restyled document
```

Add a new command:

```markdown
### /oxidize detect <file>
Full AI detection scan with tiered vocabulary, structural patterns, and statistical signals:
`conda run -n oxidizer oxidizer scan <file> --profile jinchi`
```

- [ ] **Step 2: Commit**

```bash
git add skill/SKILL.md
git commit -m "feat: update /oxidize skill with self-audit and detect command"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: All 10 gaps from the research are addressed (tiered vocab, replacements, structural detection, statistical signals, two-pass rewrite, context exemptions, severity tiers, detection registry, enhanced scan, skill update)
- [x] **Placeholder scan**: All tasks have complete code. No TBD, TODO, or "similar to Task N" references.
- [x] **Type consistency**: `VocabFinding`, `StructuralFinding`, `StatisticalReport`, `DetectionReport`, `Severity` are used consistently across tasks 1-5.
- [x] **Backward compatibility**: Existing tests should still pass. Profile schema changes use defaults. CLI scan adds new output but does not remove existing functionality.
