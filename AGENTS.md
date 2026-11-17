# Working on Oxidizer

- Preserve configured human Git identity; never add AI authors, coauthors or attribution footers.
- Install dependencies only in this project's `.venv` or a dedicated conda environment. Use that environment's executables explicitly.
- Run `.venv/bin/python -m pytest -q` after behavior changes. Legacy NLP tests need the resources described in README.
- Keep all raw writing samples, background context, briefs, model responses and human edits in ignored `.oxidizer/`. This repository is public.
- Personal and academic voices are separate. For writing tasks, look for `.oxidizer/<genre>/baseline.json`, `.oxidizer/<genre>/guide.json` and evaluation reports. Initial guides are the default; a tuned guide must earn adoption through review.
- On the configured teal checkout, `.oxidizer/context.md` supplies user-approved background; `.oxidizer/skills/personal/SKILL.md` and `.oxidizer/skills/academic/SKILL.md` are portable local guides. These files are intentionally absent from Git. Do not invent their contents on a fresh clone.
- Use substantive author briefs, not chat typos or assistant prose as gold style examples. Do not fabricate opinions, experiences, references or research results.
- Group sections of a paper together; do not retune after a test is opened. Retain failed comparisons. Heuristic style scores and model preferences are not proof of human authorship.
