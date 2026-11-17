"""Render the supplied profile, rather than imposing one academic voice on everyone."""
from oxidizer.profiles.schema import StyleProfile


def style_rules(profile: StyleProfile) -> list[str]:
    rules = ['## Style Rules',
             f'- When first person is appropriate, use "{profile.voice_rules.person}"; do not change who performed an action.',
             f'- Sentence length reference: mean ~{profile.sentence_length.mean:.0f} words; vary naturally, not to a quota.',
             '- Style examples illustrate phrasing only. Never import their facts, opinions or anecdotes.']
    rules.append('- Natural contractions are allowed.' if profile.contractions else '- Do NOT use contractions in this profile.')
    if profile.punctuation.em_dashes == 0:
        rules.append('- Do NOT use em dashes; prefer commas, semicolons or parentheses.')
    if profile.punctuation.semicolons_per_100 > 0:
        rules.append('- Semicolons may connect related clauses; do not add them to meet a numerical target.')
    if profile.voice.passive_contexts:
        rules.append('- Passive voice is appropriate in these contexts: ' + ', '.join(profile.voice.passive_contexts) + '.')
    if profile.vocabulary.banned_aiisms:
        rules.append('- Avoid these words unless preserving a quotation or necessary technical meaning: ' + ', '.join(profile.vocabulary.banned_aiisms) + '.')
    if profile.transitions.preferred:
        rules.append('- Optional transitions, only when logically needed: ' + ', '.join(profile.transitions.preferred) + '.')
    if profile.voice_rules.problem_before_solution:
        rules.append('- Establish the problem before proposing a solution.')
    if profile.voice_rules.quantitative_precision:
        rules.append('- Preserve supplied quantitative results exactly; never invent missing values.')
    rules.append('- Preserve uncertainty and limitations. Do not strengthen claims for rhetorical effect.')
    return rules
