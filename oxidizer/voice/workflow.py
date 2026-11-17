"""Learn on examples, tune with feedback, test once on untouched publications."""
import json
from pathlib import Path
from statistics import mean

from .corpus import digest, load, save_new
from .evaluation import compare
from .prompts import brief_prompt, draft_prompt, guide_prompt, packed, parse_json, validate_brief


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_state(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def bound_guide(work, corpus, writer_spec):
    guide = read(work / 'guide.json')
    if guide['writer'] != writer_spec:
        raise ValueError('Writer model differs from trained guide; use a separate work directory')
    if guide['corpus'] != corpus['fingerprint']:
        raise ValueError('Guide/corpus mismatch')
    return guide


def train(work: Path, writer, writer_spec):
    data = load(work)
    if (work / 'baseline.json').exists():
        raise ValueError('Already trained; use tune or a new work directory')
    samples = [p for p in data['posts'] if p['split'] == 'learn']
    guide = writer(guide_prompt(samples, data['genre']))
    artifact = dict(guide=guide, writer=writer_spec, corpus=data['fingerprint'], genre=data['genre'], status='unvalidated')
    save_new(work / 'baseline.json', artifact)
    save_new(work / 'guide.json', artifact)
    return artifact


def make_brief(work, post, writer):
    path = work / 'briefs' / (digest(post['id']) + '.json')
    if path.exists():
        return validate_brief(read(path))
    brief = validate_brief(parse_json(writer(brief_prompt(post))))
    save_new(path, brief)
    return brief


def compare_set(work, data, split, incumbent, candidate, writer, judge):
    rows = []
    for post in data['posts']:
        if post['split'] != split:
            continue
        brief = make_brief(work, post, writer)
        old = writer(draft_prompt(brief, incumbent, data['genre']))
        new = writer(draft_prompt(brief, candidate, data['genre']))
        result = compare(post['text'], brief, old, new, judge)
        trace = dict(id=post['id'], split=split, reference=post['text'], brief=brief,
                     baseline=old, candidate=new, comparison=result)
        trace_path = work / 'traces' / (digest(trace) + '.json')
        if not trace_path.exists():
            save_new(trace_path, trace)
        rows.append(dict(id=post['id'], group=post['group'], **result))
    # Multiple sections of one publication must not outvote independent publications.
    groups = sorted({r['group'] for r in rows})
    score = mean(mean(r['score'] for r in rows if r['group'] == g) for g in groups)
    return dict(n=len(groups), documents=len(rows), score=score,
                consistent=sum(r['consistent'] for r in rows),
                faithful=sum(r['faithful'] for r in rows), rows=rows)


def tune(work, writer, judge, optimizer, writer_spec, judge_spec, rounds=3):
    data = load(work)
    if (work / 'test-lock.json').exists():
        raise ValueError('Held-out test already opened; freeze this run and collect a new test set before tuning again')
    if not 1 <= rounds <= 10:
        raise ValueError('Use 1..10 total rounds')
    current = bound_guide(work, data, writer_spec)
    history_path = work / 'tuning.json'
    history = read(history_path) if history_path.exists() else []
    if history and history[0]['judge'] != judge_spec:
        raise ValueError('Judge model changed; use a new work directory')
    # Crash after the history write: recover the accepted guide from the persisted round.
    if history:
        current['guide'] = history[-1]['retained_guide']
        write_state(work / 'guide.json', current)
    for index in range(len(history), rounds):
        feedback = history[-1]['comparison']['rows'] if history else []
        # Establish real feedback before asking for the first change.
        if not feedback:
            feedback = compare_set(work, data, 'tune', '', current['guide'], writer, judge)['rows']
        proposal = optimizer(packed(
            'Improve this writing guide for this writer model. Return only a complete revised Markdown guide. '
            'Use concrete judge corrections, remove harmful rules, keep what works. Do not memorize task facts '
            'or add canned phrasing. Preserve factual fidelity and genre. Do not optimize for AI detectors.',
            dict(writer_model=writer_spec, genre=data['genre'], guide=current['guide'],
                 feedback=feedback, learning_samples=[p['text'] for p in data['posts'] if p['split'] == 'learn'])))
        result = compare_set(work, data, 'tune', current['guide'], proposal, writer, judge)
        accepted = result['score'] > .5 and result['faithful'] == result['documents']
        if accepted:
            current['guide'] = proposal
        history.append(dict(round=index+1, judge=judge_spec, candidate=proposal, accepted=accepted,
                            comparison=result, retained_guide=current['guide']))
        write_state(history_path, history)
        current['status'] = 'tuned; held-out validation pending'
        write_state(work / 'guide.json', current)
    return history


def evaluate(work, writer, judge, writer_spec, judge_spec, against='baseline'):
    data = load(work)
    current = bound_guide(work, data, writer_spec)
    if against not in ('baseline', 'none'):
        raise ValueError('Compare against baseline or none')
    baseline = read(work / 'baseline.json')['guide'] if against == 'baseline' else ''
    identity = dict(corpus=data['fingerprint'], guide=digest(current['guide']), writer=writer_spec,
                    judge=judge_spec)
    lock = work / 'test-lock.json'
    if lock.exists():
        if read(lock) != identity:
            raise ValueError('Held-out test is locked to different models/guide; no adaptive re-testing')
    else:
        save_new(lock, identity)  # lock even if a later model call fails
    path = work / f'evaluation-{against}.json'
    if path.exists():
        return read(path)
    result = compare_set(work, data, 'test', baseline, current['guide'], writer, judge)
    report = dict(**identity, **result, against=against, split='test', warnings=data['warnings'],
                  interpretation='Model-judged resemblance; not human validation or proof of authorship. Inspect briefs for style leakage.')
    save_new(path, report)
    return report
