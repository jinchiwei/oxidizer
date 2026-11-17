"""Behavior tests for corpus isolation, blind judging, and bounded tuning."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from oxidizer.cli import cli


def corpus(tmp_path):
    rows = []
    for i in range(5):
        p = tmp_path / f'post{i}.md'
        p.write_text(f'---\ntitle: Post {i}\n---\nOriginal prose about subject {i}.')
        rows.append(dict(id=f'p{i}', path=p.name, date=f'2026-01-0{i+1}',
                         group=f'publication{i}', genre='personal', source=f'own:{i}',
                         authorship='author'))
    manifest = tmp_path / 'corpus.json'
    manifest.write_text(json.dumps(rows))
    return manifest


def test_prepare_cli_freezes_newest_posts_without_model_calls(tmp_path):
    manifest = corpus(tmp_path)
    dest = tmp_path / 'work'
    result = CliRunner().invoke(cli, ['voice', 'prepare', str(manifest), '--genre', 'personal', '--work', str(dest)])
    assert result.exit_code == 0, result.output
    data = json.loads((dest / 'corpus.json').read_text())
    assert [p['id'] for p in data['posts'] if p['split'] == 'test'] == ['p4']
    assert [p['id'] for p in data['posts'] if p['split'] == 'tune'] == ['p3']
    assert 'title:' not in data['posts'][0]['text']
    assert data['warnings']
    result = CliRunner().invoke(cli, ['voice', 'prepare', str(manifest), '--genre', 'personal', '--work', str(dest)])
    assert result.exit_code != 0  # never silently reset frozen splits


def test_duplicate_documents_and_group_sections_cannot_leak(tmp_path):
    from oxidizer.voice.corpus import prepare
    manifest = corpus(tmp_path)
    rows = json.loads(manifest.read_text())
    rows[1]['group'] = rows[0]['group']
    manifest.write_text(json.dumps(rows))
    data = prepare(manifest, 'personal')
    assert len({p['split'] for p in data['posts'] if p['group'] == 'publication0'}) == 1
    (tmp_path / 'post4.md').write_text((tmp_path / 'post0.md').read_text())
    with pytest.raises(ValueError, match='duplicate'):
        prepare(manifest, 'personal')


def test_mixed_authorship_not_used_as_gold(tmp_path):
    from oxidizer.voice.corpus import prepare
    manifest = corpus(tmp_path)
    rows = json.loads(manifest.read_text())
    rows[0]['authorship'] = 'unknown'
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='authorship'):
        prepare(manifest, 'personal')


def test_swapped_judgments_map_to_original_candidate():
    from oxidizer.voice.evaluation import compare
    calls = []
    responses = iter([
        '{"closer":"B","A_notes":"formal","B_notes":"natural","A_faithful":true,"B_faithful":true}',
        '{"closer":"A","A_notes":"natural","B_notes":"formal","A_faithful":true,"B_faithful":true}',
    ])
    def judge(prompt):
        calls.append(json.loads(prompt.split('\nDATA\n')[1]))
        return next(responses)
    result = compare('gold', {'thesis': 'fact'}, 'baseline', 'candidate', judge)
    assert result['score'] == 1
    assert result['consistent'] is True
    assert calls[0]['A'] == calls[1]['B'] == 'baseline'
    assert 'candidate' not in calls[0]  # blinded keys, not model/arm names


def test_position_bias_is_disagreement_not_win():
    from oxidizer.voice.evaluation import compare
    result = compare('gold', {}, 'old', 'new', lambda p: json.dumps(dict(
        closer='A', A_notes='x', B_notes='y', A_faithful=True, B_faithful=True)))
    assert result['score'] == .5
    assert result['consistent'] is False


def test_unfaithful_candidate_cannot_win_and_bad_judge_fails():
    from oxidizer.voice.evaluation import compare
    answers = iter([
        dict(closer='B', A_notes='x', B_notes='invented experience', A_faithful=True, B_faithful=False),
        dict(closer='A', A_notes='invented experience', B_notes='x', A_faithful=False, B_faithful=True),
    ])
    assert compare('gold', {}, 'old', 'new', lambda p: json.dumps(next(answers)))['score'] == 0
    with pytest.raises(ValueError):
        compare('gold', {}, 'old', 'new', lambda p: '{"closer":"C"}')


def test_brief_requires_author_substance_and_no_gold_in_draft():
    from oxidizer.voice.prompts import draft_prompt, validate_brief
    with pytest.raises(ValueError):
        validate_brief({'topic': 'write something'})
    brief = dict(topic='Tools', thesis='Simple tools help.', points=['Use a short checklist.'],
                 evidence=['My checklist has 3 items.'], experiences=[], quotes=[], target_words=100)
    prompt = draft_prompt(brief, 'GUIDE', 'personal', context='BACKGROUND ONLY')
    assert 'GUIDE' in prompt and 'BACKGROUND ONLY' in prompt
    assert 'gold' not in json.loads(prompt.split('\nDATA\n')[1])


def test_prompt_export_and_feedback_are_local(tmp_path):
    brief = tmp_path / 'brief.json'
    brief.write_text(json.dumps(dict(topic='Test', thesis='My position', points=['A point'],
                                    evidence=['An observation'], experiences=[], quotes=[], target_words=100)))
    guide = tmp_path / 'guide.md'
    guide.write_text('Use a measured voice.')
    result = CliRunner().invoke(cli, ['voice', 'draft', str(brief), '--guide', str(guide),
                                     '--genre', 'academic', '--prompt-only'])
    assert result.exit_code == 0, result.output
    assert 'My position' in result.output
    assert 'Use a measured voice.' in result.output


def test_cached_model_never_reuses_another_model_or_counts_cache_as_spend(tmp_path):
    from oxidizer.voice.model import Model
    calls = []
    def transport(prompt, model):
        calls.append(model)
        return 'answer'
    m = Model('writer-a', tmp_path, max_calls=1, transport=transport)
    assert m('question') == m('question') == 'answer'
    with pytest.raises(RuntimeError, match='budget'):
        m('another question')
    other = Model('writer-b', tmp_path, max_calls=1, transport=transport)
    assert other('question') == 'answer'
    assert calls == ['writer-a', 'writer-b']


def test_train_tune_test_boundary_and_model_binding(tmp_path):
    from oxidizer.voice.corpus import prepare, save_new
    from oxidizer.voice.workflow import train, tune, evaluate
    work = tmp_path / 'work'
    save_new(work / 'corpus.json', prepare(corpus(tmp_path), 'personal'))
    seen = []
    brief = dict(topic='tools', thesis='Keep it simple', points=['Choose a short checklist'],
                 evidence=['3 items'], experiences=[], quotes=[], target_words=100)
    def writer(prompt):
        seen.append(prompt)
        if 'content-only brief' in prompt:
            return json.dumps(brief)
        if 'learning samples' in prompt:
            return 'Measured guide.'
        return 'Draft with three items.'
    def judge(prompt):
        seen.append(prompt)
        return json.dumps(dict(closer='tie', A_notes='Same', B_notes='Same', A_faithful=True, B_faithful=True))
    train(work, writer, 'model-a')
    assert all('subject 3' not in p and 'subject 4' not in p for p in seen)
    tune(work, writer, judge, lambda p: 'A changed guide.', 'model-a', 'judge-a', rounds=1)
    assert all('subject 4' not in p for p in seen)
    assert json.loads((work/'guide.json').read_text())['guide'] == 'Measured guide.'  # tie reverts
    with pytest.raises(ValueError, match='model'):
        evaluate(work, writer, judge, 'model-b', 'judge-a')
    result = evaluate(work, writer, judge, 'model-a', 'judge-a')
    assert result['n'] == 1 and result['score'] == .5
    assert result['split'] == 'test'
    traces = [json.loads(p.read_text()) for p in (work/'traces').glob('*.json')]
    assert any(t['reference'].endswith('subject 4.') and t['candidate'] == 'Draft with three items.' for t in traces)
    with pytest.raises(ValueError, match='test'):
        tune(work, writer, judge, writer, 'model-a', 'judge-a')


def test_export_keeps_model_binding_and_does_not_publish_samples(tmp_path):
    work = tmp_path/'work'
    work.mkdir()
    from oxidizer.voice.corpus import prepare, save_new
    data = prepare(corpus(tmp_path), 'personal')
    save_new(work/'corpus.json', data)
    save_new(work/'guide.json', dict(guide='An author-specific guide.', writer='claude:writer-a',
                                   corpus=data['fingerprint'], genre='personal', status='unvalidated'))
    out = tmp_path/'SKILL.md'
    result = CliRunner().invoke(cli, ['voice', 'export', '--work', str(work), '--output', str(out)])
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert 'claude:writer-a' in text and 'An author-specific guide.' in text
    assert 'Original prose' not in text


def test_human_edits_are_distinct_from_background_facts():
    from oxidizer.voice.prompts import draft_prompt
    brief = dict(topic='tools', thesis='Keep it simple', points=['Choose a short checklist'],
                 evidence=['3 items'], experiences=[], quotes=[], target_words=100)
    p = draft_prompt(brief, 'Guide', 'personal', context='Research background',
                     feedback='Keep my uncertainty instead of making the claim stronger.')
    data = json.loads(p.split('\nDATA\n')[1])
    assert data['author_feedback'].startswith('Keep my uncertainty')
    assert data['background'] == 'Research background'


def test_cli_model_error_and_timeout_do_not_cache_partial_output(tmp_path, monkeypatch):
    import subprocess
    from oxidizer.voice.model import Model
    def fail(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, json.dumps(dict(subtype='error_max_budget_usd', is_error=False, result='unfinished')), '')
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(RuntimeError):
        Model('claude:test-model', tmp_path)('task')
    assert not list(tmp_path.glob('*.json'))
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='timed out'):
        Model('claude:test-model', tmp_path)('task')


def test_tuning_resumes_the_saved_guide_after_interruption(tmp_path):
    from oxidizer.voice.corpus import prepare, save_new
    from oxidizer.voice.workflow import tune
    work = tmp_path/'work'
    data = prepare(corpus(tmp_path), 'personal')
    save_new(work/'corpus.json', data)
    save_new(work/'guide.json', dict(guide='stale', writer='writer', corpus=data['fingerprint']))
    save_new(work/'tuning.json', [dict(judge='judge', retained_guide='accepted guide')])
    tune(work, None, None, None, 'writer', 'judge', rounds=1)
    assert json.loads((work/'guide.json').read_text())['guide'] == 'accepted guide'
