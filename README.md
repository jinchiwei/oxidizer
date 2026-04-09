# Oxidizer

Academic writing style engine. Analyzes your writing voice and applies it to new or existing documents.

## What it does

- **revise**: Restyle an existing document to match your voice, preserving all citations, numbers, equations
- **write**: Generate new section drafts in your voice (requires API key or Claude Code)
- **score**: Evaluate how closely text matches your style profile
- **scan**: Detect AI-isms and banned patterns in any text
- **diff**: Compare original vs restyled text with style annotations
- **compare**: Score text against multiple profiles side by side
- **validate-profile**: Check how well a profile matches your actual writing samples

## Setup

```bash
conda activate oxidizer
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

## Usage

### Score text against your profile (local, no API)
```bash
oxidizer score paper.md --profile jinchi
```

### Scan for AI-isms (local, no API)
```bash
oxidizer scan paper.md --profile jinchi
```

### Revise a document (requires ANTHROPIC_API_KEY or use in Claude Code)
```bash
oxidizer revise paper.docx --profile jinchi --output restyled.md
```

### Generate HTML style report
```bash
oxidizer score paper.md --profile jinchi --html
```

## Style Profiles

YAML files in `profiles/`. Each captures sentence length targets, voice ratio, banned AI-isms, transition preferences, punctuation patterns, and few-shot examples.

## How Revise Works

1. Parse document into sections by headings
2. Extract entities to lock (citations, numbers, equations, abbreviations)
3. Send each section to Claude with style profile and locked entities
4. Verify all locked entities appear in output (retries if missing)
5. Compute style-match metrics
6. Output restyled text + JSON report
