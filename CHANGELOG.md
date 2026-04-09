# Changelog

All notable changes to Oxidizer will be documented in this file.

## [0.1.0.0] - 2026-04-08

### Added
- Style profile system with structured YAML schema and profile loader
- Document parsers for Markdown, DOCX, LaTeX, and PDF (best-effort)
- Entity extractor for citations, numbers, equations, abbreviations, and figure/table references
- Entity preservation checker with word-boundary matching to verify locked content survives restyling
- Style metrics engine computing 8 dimensions: sentence length, variance, active voice ratio, banned words, semicolons, parentheticals, transitions, contractions
- Weighted composite style-match scoring (0.0 to 1.0)
- Revise engine with entity locking, Claude API restyling, preservation verification, and retry logic
- Write engine for section drafting in the author's voice
- CLI with 7 commands: `score`, `revise`, `write`, `scan`, `diff`, `compare`, `validate-profile`
- De-AI scan mode detecting 28 banned AI-ism patterns with line-number context
- Style diff command showing annotated changes between original and revised text
- HTML visual style report with CSS bar charts and color-coded metrics
- Profile validation against writing samples
- Multi-profile comparison scoring
- Claude Code skill (`/oxidize`) for in-conversation restyling without API key
- Optional API mode for batch CLI processing (requires ANTHROPIC_API_KEY)
- Jinchi Wei style profile derived from 3 academic papers (2021-2025)
- 288 tests covering all modules
