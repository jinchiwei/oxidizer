# Oxidizer

Write and revise in your own voice, with separate personal and academic writing workflows. Supply the ideas and evidence; Oxidizer handles phrasing and tests whether a writing guide helps.

The old `score` and `scan` commands are **heuristic editing diagnostics**, not proof that text sounds like you or was written by a human. The `voice` workflow compares generated drafts against actual writing samples and checks factual fidelity separately.

## Setup

Use a project environment. No system packages, `pip --user`, or global tool installs are needed.

```bash
cd ~/arcadia/repos/oxidizer
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

If your system Python lacks venv support, use an existing conda Python to create the venv **in an empty destination**, or create a dedicated conda environment. Do not overlay two Python versions in the same venv.

The new `voice` workflow does not need NLP model downloads. For legacy `score`, `scan`, `write` and `revise`:

```bash
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python -m nltk.downloader -d .venv/nltk_data punkt_tab
```

Scoring never downloads resources implicitly. A verified dependency snapshot for this change is in `docs/voice-validation.md`.

## Draft from your own substance

Start with [`examples/brief.json`](examples/brief.json), replacing its illustrative content with your actual material. Include your thesis, ordered points, supporting evidence, actual experiences, exact attributed quotes, and desired length. A one-line topic is insufficient for the new workflow.

With a trained local voice:

```bash
.venv/bin/oxidizer voice draft my-brief.json \
  --work .oxidizer/personal --genre personal \
  --context .oxidizer/context.md --output draft.md

.venv/bin/oxidizer voice draft research-brief.json \
  --work .oxidizer/academic --genre academic --output research-draft.md
```

The writer model is read from the trained artifact. `--variant baseline` is the default: tuning does not automatically replace the initial learned guide. Use `--variant tuned` only after reviewing the comparisons and the prose. A guide learned for one model cannot silently run under another.

To work in an existing chat without making a model call:

```bash
.venv/bin/oxidizer voice draft my-brief.json \
  --work .oxidizer/personal --genre personal --prompt-only > writing-prompt.txt
```

You can also supply `--guide my-guide.md` instead of `--work`. Actual generation then requires `--writer claude:YOUR_MODEL` or `--writer api:YOUR_MODEL`. The latter uses `ANTHROPIC_API_KEY` with the optional API dependency. All live calls send the selected text to that provider. `--context` adds explicitly selected background; it is not treated as a source of new facts or author-written style examples.

## Build and test a voice

Create a JSON manifest, with one entry per document or selected section:

```json
[
  {
    "id": "essay-01",
    "path": "sources/essay-01.md",
    "date": "2026-10-01",
    "group": "original-publication-01",
    "genre": "personal",
    "authorship": "author",
    "source": "My published essay or original manuscript"
  }
]
```

Paths resolve relative to the manifest. Collect at least three **independent publication groups**; 12–25 examples in the same format is a better starting point. `authorship: author` is your provenance assertion, not something the tool can verify. Do not use assistant chat replies, boilerplate, quotations by someone else, or papers you merely coauthored as unquestioned evidence of your voice. A collaboration-heavy corpus needs that limitation recorded.

Keep sections/revisions of the same paper in the same `group`. Exact normalized duplicates are rejected; manually check near-duplicates and co-written sections. Markdown frontmatter is stripped. The newest groups are reserved for testing; the next-newest groups are used for tuning. With five groups the split is 3/1/1. With three it is 1/1/1. Small runs are explicitly labeled pilots.

```bash
.venv/bin/oxidizer voice prepare .oxidizer/personal-manifest.json \
  --genre personal --work .oxidizer/personal
.venv/bin/oxidizer voice train --work .oxidizer/personal \
  --writer claude:YOUR_MODEL --max-calls 1
.venv/bin/oxidizer voice tune --work .oxidizer/personal \
  --writer claude:YOUR_MODEL --judge claude:YOUR_JUDGE \
  --rounds 1 --max-calls 12
.venv/bin/oxidizer voice evaluate --work .oxidizer/personal \
  --writer claude:YOUR_MODEL --judge claude:YOUR_JUDGE --against baseline
.venv/bin/oxidizer voice evaluate --work .oxidizer/personal \
  --writer claude:YOUR_MODEL --judge claude:YOUR_JUDGE --against none
```

Repeat with a separate academic manifest and work directory. Do not combine personal reflections and manuscript prose into a single universal guide.

`train` reads only learning examples. `tune` extracts substance-only briefs and runs a bounded feedback-driven keep/revert loop. It implements a simple optimizer, **not DSPy GEPA**. The judge sees a reference and two anonymous drafts, then sees them again in reverse order. Disagreements and ties score 0.5; an unfaithful candidate scores 0. Publication groups get equal weight. Judge notes and both verdicts are retained. A model judgment about fidelity is not a factual guarantee.

`evaluate` permanently locks tuning in that work directory as soon as the test is opened, even if a call fails. You may resume the same comparison or run the prespecified no-guide comparator, but cannot change the guide/models and retest. Keep the judge fixed. Inspect extracted briefs for copied stylistic language, outputs for fabricated content, and test results for regressions. Public examples may already be in model training data; this measures preference on these samples, not contamination-free authorship identification.

**Budget and resumption:** Each invocation has a maximum number of new calls **per distinct model** (default 40). Different writer/judge/optimizer models each have their own allowance. Successful calls are cached by model and complete prompt; failures are not retried automatically. Claude CLI calls have a 240-second timeout and a requested $1 per-call cap, use no tools/MCP, and use your existing login. Subscription quota accounting is provider-controlled; the cap is not an invoice prediction. API calls have an 8192-token output cap, no automatic retries, and no dollar ceiling. One tuning round on one document uses up to nine calls; a fresh test comparison uses five.

## Feed your edits back

```bash
.venv/bin/oxidizer voice feedback draft.md edited-by-me.md \
  --genre personal --note 'Keep my uncertainty; do not turn questions into claims.' \
  --output .oxidizer/edits/first-post.json
.venv/bin/oxidizer voice draft next-brief.json --work .oxidizer/personal \
  --genre personal --feedback .oxidizer/edits/first-post.json --output next-draft.md
```

This stores the actual diff and your reason, and can apply it to future drafts. It does not silently modify a validated guide or mix your edits into held-out data. Rebuild with fresh publications when you want another training cycle.

Export a guide for an agent/chat without installing anything globally:

```bash
.venv/bin/oxidizer voice export --work .oxidizer/personal \
  --variant baseline --output .oxidizer/skills/personal/SKILL.md
.venv/bin/oxidizer voice status --work .oxidizer/personal
```

The export includes the writer model and guide but no source documents. Existing files are not overwritten. The repository's [`skill/SKILL.md`](skill/SKILL.md) explains both workflows.

## Existing document tools

```bash
.venv/bin/oxidizer revise paper.docx --profile jinchi --output revised.md
.venv/bin/oxidizer score paper.md --profile jinchi
.venv/bin/oxidizer scan paper.md --profile jinchi
.venv/bin/oxidizer diff original.md revised.md --profile jinchi
```

The academic `jinchi` YAML profile remains available. Prompt rules now respect the profile's pronouns, contractions and punctuation, rather than imposing “we” on every writer. Allowed contractions are no longer penalized. Citation/number preservation checks remain, but cannot verify that the surrounding scientific interpretation is correct. Review publication drafts yourself.

## Local data and reproducibility

`.oxidizer/` is Git-ignored. Keep manifests, raw samples, context, briefs, generated drafts, judge notes, and human edits there. Do not force-add them to a public repository. Commit code and aggregate validation summaries only. Successful call caches contain generated text and should be treated as personal data.

The design was informed by [Mike Taylor's article](https://every.to/also-true-for-humans/how-i-finally-got-ai-to-write-like-me). This implementation is independent; the linked script was not vendored. Its headline results are not Oxidizer's results.

```bash
.venv/bin/python -m pytest -q
```
