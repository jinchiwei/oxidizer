# Voice Learning Implementation Plan

> **For agentic workers:** Execute inline; tests precede behavior changes.

**Goal:** Make Oxidizer support Jinchi's personal and academic voices with evidence from his writing and a reproducible comparison workflow.

**Architecture:** Keep the existing academic editor and fix profile overrides. Add a separate `voice` workflow with a local corpus, content briefs, model-specific guides, order-swapped reference comparisons, bounded feedback-based tuning, and untouched test documents.

**Tech Stack:** Python, Click, YAML, pytest; existing Claude CLI or optional Anthropic API. Project `.venv` only.

**Spec:** User request dated 2026-11-16 and https://every.to/also-true-for-humans/how-i-finally-got-ai-to-write-like-me. Equal support for academic and personal writing. Implement the method independently; do not vendor the linked unlicensed script.

## Global Constraints

- Preserve existing local `skill/SKILL.md` change. No AI Git attribution.
- Personal corpora, chat context, model outputs and edits stay in ignored `.oxidizer/`; public repo contains code and generic examples only.
- No invented personal experiences, results, citations, or opinions. Context is not automatically a writing exemplar.
- Latest document groups are held out. Do not train on another section of a held-out paper.
- A heuristic style score is a diagnostic, not proof of authorship or a voice-quality metric.
- No silent model changes, retries, or unbounded optimization. Cache successful calls and record model IDs.

## Review Focus

- Duplicate posts / sections crossing split boundaries: group and normalized-content checks.
- Judge order bias or invalid output: swapped positions, explicit ties/disagreement, strict parsing.
- Model/provider changes: cache keyed on model and full prompt; candidate metadata checked on use.
- Missing substantive input: require brief thesis and evidence/experiences; forbid importing facts from style samples.
- Publication leakage: keep ignored raw data out of commits; generic fixtures only.

## Tasks

1. Profile behavior (`engine/write.py`, `engine/revise.py`, shared prompt rules, reporter). Write failing tests for personal pronouns, contractions, punctuation, and scoring; fix and run old tests.
2. Corpus and prompts (`voice/corpus.py`, `voice/prompts.py`). Test stable grouped splits, duplicate detection, source boundaries, brief validation, and train-only examples; implement local manifests and brief/draft/feedback prompts.
3. Model execution and evaluation (`voice/model.py`, `voice/evaluation.py`, `voice/workflow.py`). Test caching, CLI failure/truncation, swapped label accounting, keep/revert, held-out isolation; implement bounded runs and reporting.
4. CLI and onboarding (`voice/cli.py`, README, skill). Test end-to-end offline CLI preparation/prompt export, explicit model selection, output collision handling. Add personal/academic local corpus and context with provenance; do a small live pilot if the existing model login works.
5. Run full tests and review diffs. Commit intentional changes with configured human identity, push, report actual outcomes and limitations.
