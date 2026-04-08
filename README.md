# Oxidizer

Writing style analysis and replication toolkit. Analyzes writing samples to produce a comprehensive style profile and a ready-to-use Claude custom style prompt.

## What it does

1. **Statistical analysis** — sentence length, paragraph length, vocabulary complexity, voice ratios, contraction usage
2. **Pattern extraction** — transition words, paragraph openers/closers, hedging vs directness, punctuation habits, distinctive vocabulary
3. **Anti-AI audit** — cross-references writing against known AI-ism patterns to ensure the style guide reflects actual habits
4. **Output** — a detailed `style_profile.md` and a concise `custom_style_prompt.md` for use as Claude custom instructions

## Structure

```
oxidizer/
├── dev/src/
│   ├── style_profile.md         # Detailed analysis with stats and verbatim excerpts
│   └── custom_style_prompt.md   # Ready-to-paste Claude custom style prompt
├── *.pdf, *.docx                # Writing samples (not tracked in git)
└── README.md
```

## Setup

```bash
conda activate oxidizer
pip install python-docx nltk lxml
```

## Usage

Place writing samples (PDF, DOCX) in the root directory and run the analysis scripts in `dev/src/`.
