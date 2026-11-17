"""Position-swapped reference judging, with separate fidelity checks."""
from .prompts import judge_prompt, parse_json


def compare(reference, brief, baseline, candidate, judge):
    verdicts, votes, faithful, notes = [], [], [], []
    for swapped in (False, True):
        a, b = (candidate, baseline) if swapped else (baseline, candidate)
        v = parse_json(judge(judge_prompt(reference, brief, a, b)))
        expected = {'closer', 'A_notes', 'B_notes', 'A_faithful', 'B_faithful'}
        if set(v) != expected or v['closer'] not in ('A', 'B', 'tie'):
            raise ValueError('Invalid judge verdict')
        if any(type(v[k]) is not bool for k in ('A_faithful', 'B_faithful')) or any(not isinstance(v[k], str) for k in ('A_notes', 'B_notes')):
            raise ValueError('Invalid judge field types')
        label = 'A' if swapped else 'B'
        votes.append(.5 if v['closer'] == 'tie' else float(v['closer'] == label))
        faithful.append(v[label + '_faithful'])
        notes.append(v[label + '_notes'])
        verdicts.append(v)
    return dict(score=sum(votes)/2 if all(faithful) else 0.0,
                style_score=sum(votes)/2, consistent=votes[0] == votes[1],
                faithful=all(faithful), notes=notes, verdicts=verdicts)
