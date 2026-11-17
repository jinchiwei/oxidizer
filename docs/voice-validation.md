# Voice workflow validation — 2026-11-16

## What changed

The original generator imposed academic first-person plural, banned contractions regardless of profile, and optimized surface features without testing resemblance to the author. The new workflow uses separate personal and academic corpora, content briefs, model-specific guides, and comparisons against real examples. The old tools remain available with profile-rule fixes.

## Actual pilot, not simulated results

Writer, judge and optimizer requested `claude:claude-opus-5-5`, high effort, through the existing Claude CLI login. One tuning round per genre, no automatic retries. Each genre produced 18 successful cached responses (36 total). No billed-cost claim is made. The pilot used the new workflow; it did not run the author's linked script or GEPA.

| Genre | Publication groups (learn / tune / test) | Tune: candidate vs initial guide | Test: tuned vs initial guide | Test: tuned vs no guide |
|---|---|---|---|---|
| Personal | 3 / 1 / 1 | Tuned preferred in both orders | Initial guide preferred in both orders | Order disagreement |
| Academic | 1 / 1 / 1 | Tuned preferred in both orders | Tuned preferred in both orders | Tuned preferred in both orders |

Every test candidate was judged faithful in both orders. This is a model judgment, not independent verification of factual fidelity. Two presentation orders of one document are **not two independent test examples**.

Personal sources were five Wei's Takes posts. Academic sources were selected first-author thesis, SPIE, and ARIA excerpts already associated with the original academic profile; published papers may reflect collaborator/editor contributions. The academic held-out excerpt was only 143 words. Each test set has **one publication**: these results cannot establish a reliable population win rate, statistical significance, generalization across writing formats, or superiority over human editing. The same model wrote and judged the drafts, another source of bias. Public papers/posts may have appeared in model training.

The personal result demonstrates why tuning gains must not automatically ship. Its initial learned guide was retained for use. The academic tuned guide was exported as a provisional option. The initial guide is the CLI default in both genres; choosing `--variant tuned` remains explicit. No prompts were tuned after opening either test set.

The comparisons are against a newly inferred one-shot guide and no guide, **not against the old Oxidizer academic prompt**. Do not describe these numbers as a measured before/after product improvement. More independently authored samples and blind human review are needed.

Raw examples, briefs, outputs, per-document judge notes, guides and context are excluded from this public repo. Local review files contain the reference, brief, baseline, tuned and no-guide drafts. The user approved processing the academic excerpts with Claude.

## Checks

- Complete test suite: **457 passed** in the project environment.
- New tests cover publication-group splits, duplicate rejection, authorship declarations, test isolation, model binding, swapped judge labels, order disagreement, fidelity gating, cache budgets, timeout/partial-response rejection, keep/revert behavior, interrupted tuning state, local traces, prompt export, and profile overrides.
- Real CLI checks: preparation, training, one tuning round, both test comparators, status, prompt-only drafting, and skill export for both genres.
- Dependency snapshot: Python 3.12.16; pytest 9.2.0; spaCy 3.8.18 with en_core_web_sm 3.8.0; NLTK 3.9.2; Click 8.5.1; Anthropic SDK 2.9.0. NLP data was installed inside the project environment. The live pilot used CLI, not the API backend.
- The API adapter is available but was not live-tested here. Provider billing ceilings and cross-provider reproducibility are not established.

## Reproduce with your own writing

Follow the README with a private local manifest. Do not copy individual sections into different split groups. Inspect briefs for style leakage before trusting results; the extraction prompt requests neutral notes but does not guarantee it. Keep the test locked, log failed experiments, compare no-guide performance, and use actual author edits for the next independent training cycle.
