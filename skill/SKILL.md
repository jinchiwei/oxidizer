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
5. Run scoring on the restyled text
6. Verify all extracted entities appear in the restyled text
7. Output the restyled document

### /oxidize write <topic> --section <type>
Draft a new section in your voice. Types: intro, methods, results, discussion.

### /oxidize score <file>
Score text against your profile (local, no API): `conda run -n oxidizer oxidizer score <file> --profile jinchi`

### /oxidize scan <file>
Detect AI-isms (local, no API): `conda run -n oxidizer oxidizer scan <file> --profile jinchi`

### /oxidize diff <original> <revised>
Show what changed and why: `conda run -n oxidizer oxidizer diff <original> <revised> --profile jinchi`

## Profile Location
profiles/ directory at /Users/jinchiwei/arcadia/oxidizer/profiles/
