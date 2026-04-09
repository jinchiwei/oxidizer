---
name: oxidize
description: Academic writing style engine. Revise, write, score, and scan documents in your personal voice.
---

# /oxidize

Academic writing style engine. Works with your personal style profile to write and revise in your voice.

## Commands

### /oxidize revise <file> [--sections X,Y]
Restyle a document to match your writing voice. Preserves citations, numbers, equations, abbreviations.

Steps:
1. Read the file and parse into sections
2. For each section, run: `conda run -n oxidizer oxidizer score <temp_file> --profile jinchi --json-output` to get baseline
3. Extract entities using the Python extraction module
4. Restyle the section text using the style profile (Claude does this in-conversation)
5. **Self-audit**: Run `conda run -n oxidizer oxidizer scan <temp_restyled_file> --profile jinchi` on the restyled text. If any P0 vocabulary or structural patterns (repetitive starters, length uniformity) are flagged, fix them in a second pass. Pay special attention to:
   - Repetitive sentence starters (don't start 40%+ of sentences with the same word, especially "We")
   - P0 AI-isms that crept into the rewrite (delve, utilize, leverage, serves as, etc.)
   - Metronomic sentence lengths (vary between short punchy sentences and longer flowing ones)
6. Run scoring on the restyled text
7. Verify all extracted entities appear in the restyled text
8. Output the restyled document

### /oxidize write <topic> --section <type>
Draft a new section in your voice. Types: intro, methods, results, discussion.

### /oxidize score <file>
Score text against your profile (local, no API): `conda run -n oxidizer oxidizer score <file> --profile jinchi`

### /oxidize scan <file>
Full AI detection scan with tiered vocabulary (P0/P1/P2), structural patterns (repetitive starters, length uniformity, rule-of-three, em dashes), and statistical signals (burstiness, trigram repetition). Local, no API:
`conda run -n oxidizer oxidizer scan <file> --profile jinchi`

### /oxidize diff <original> <revised>
Show what changed and why: `conda run -n oxidizer oxidizer diff <original> <revised> --profile jinchi`

## Profile Location
profiles/ directory at /Users/jinchiwei/arcadia/oxidizer/profiles/
