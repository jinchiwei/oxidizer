---
name: oxidize
description: Personal and academic writing voices. Draft from your own ideas, revise, and evaluate against your writing.
---

# /oxidize

Writing engine with separate personal and academic voices. Work from the author's actual argument and evidence.

## Preferred workflow: learned voices

Locate the Oxidizer checkout (normally `~/arcadia/repos/oxidizer`) and use its `.venv/bin/oxidizer` or an existing dedicated conda environment. Never install globally. Read `README.md` for the current CLI.

1. Choose personal or academic based on the requested document. Do not infer a manuscript voice from casual chat typos.
2. If local `.oxidizer/<genre>/baseline.json` exists, read that model-specific guide. Read `evaluation-*.json` before using the tuned `guide.json`; a tuning improvement may fail on held-out writing. Keep the baseline unless the author chooses a validated tuned guide.
3. Use `.oxidizer/context.md` only as background. Use only explicitly supplied author ideas, evidence, real experiences, results and quotes as the new piece's substance. Ask for missing substance instead of inventing an opinion or story.
4. Build the brief described in `examples/brief.json`. Export the prompt with `voice draft BRIEF --work .oxidizer/<genre> --genre <genre> --prompt-only`; this makes no model call. Draft from it in the current chat, noting if the current model differs from the guide's writer model.
5. For revision, preserve the author's meaning, hedging, citations, numbers and experiences. Do not force every sentence through a stylistic checklist. `score` and `scan` are diagnostics, not authorship tests or targets to maximize.
6. Save the author's edits with `voice feedback`, and supply that file through `voice draft --feedback` next time. Never tune against a held-out test after seeing its results.

Raw examples, chat context and generated outputs belong in Git-ignored `.oxidizer/`. Never commit them to this public repo. Both drafting and revision remain reviewable by the author.

## Commands

### /oxidize revise <file> [--sections X,Y]
Restyle a document to match your writing voice. Preserves citations, numbers, equations, abbreviations.

Steps:
1. Read the file and parse into sections
2. For each section, run: `.venv/bin/oxidizer score <temp_file> --profile jinchi --json-output` to get baseline
3. Extract entities using the Python extraction module
4. Restyle the section text using the style profile (Claude does this in-conversation)
5. **Diagnostic review**: Run `.venv/bin/oxidizer scan <temp_restyled_file> --profile jinchi` on the restyled text. Inspect flagged patterns against the author's actual examples; change them only if doing so improves the voice without altering meaning. Do not remove technical terms or natural author habits just to lower a score. Possible issues include:
   - Repetitive sentence starters (don't start 40%+ of sentences with the same word, especially "We")
   - P0 AI-isms that crept into the rewrite (delve, utilize, leverage, serves as, etc.)
   - Metronomic sentence lengths (vary between short punchy sentences and longer flowing ones)
6. Run scoring on the restyled text
7. Verify all extracted entities appear in the restyled text
8. Output the restyled document

### /oxidize write <topic> --section <type>
Draft a new section in your voice. Types: intro, methods, results, discussion.

### /oxidize score <file>
Score text against your profile (local, no API): `.venv/bin/oxidizer score <file> --profile jinchi`

### /oxidize scan <file>
Full AI detection scan with tiered vocabulary (P0/P1/P2), structural patterns (repetitive starters, length uniformity, rule-of-three, em dashes), and statistical signals (burstiness, trigram repetition). Local, no API:
`.venv/bin/oxidizer scan <file> --profile jinchi`

### /oxidize diff <original> <revised>
Show what changed and why: `.venv/bin/oxidizer diff <original> <revised> --profile jinchi`

## Profile Location
profiles/ directory at /Users/jinchiwei/arcadia/oxidizer/profiles/
