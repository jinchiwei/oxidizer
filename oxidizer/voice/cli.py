"""CLI for local corpora, writing guides, and measured voice learning."""
import difflib
import functools
import json
from pathlib import Path

import click

from .corpus import prepare as prepare_corpus, save_new, load, digest
from .model import Model
from .prompts import draft_prompt, validate_brief
from . import workflow


def checked(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, OSError, RuntimeError, KeyError) as e:
            raise click.ClickException(str(e)) from e
    return run


def work_option(fn):
    return click.option('--work', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)(fn)


def model_options(fn):
    fn = click.option('--max-calls', type=click.IntRange(1, 500), default=40,
                      help='Maximum new calls per distinct model in this invocation; cached calls are free.')(fn)
    fn = click.option('--writer', required=True, help='Explicit claude:MODEL or api:MODEL.')(fn)
    return work_option(fn)


def models(work, max_calls, *specs):
    return {s: Model(s, work / 'cache', max_calls=max_calls) for s in set(specs)}


@click.group()
def voice():
    """Learn and test separate personal/academic voices from your own writing."""


@voice.command()
@click.argument('manifest', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--genre', type=click.Choice(['personal', 'academic']), required=True)
@click.option('--work', type=click.Path(path_type=Path), required=True)
@checked
def prepare(manifest, genre, work):
    """Freeze a JSON corpus manifest into learn/tune/test publication groups. No model calls."""
    if work.exists() and any(work.iterdir()):
        raise ValueError('Work directory must be empty; frozen runs are never overwritten')
    data = prepare_corpus(manifest, genre)
    save_new(work / 'corpus.json', data)
    click.echo(json.dumps(dict(genre=genre, splits={s: sum(p['split'] == s for p in data['posts'])
                                                 for s in ['learn', 'tune', 'test']}, warnings=data['warnings']), indent=2))


@voice.command()
@model_options
@checked
def train(work, writer, max_calls):
    """Infer the baseline guide from learning documents only (one model call)."""
    m = Model(writer, work / 'cache', max_calls=max_calls)
    result = workflow.train(work, m, writer)
    click.echo(f'Guide saved to {work / "guide.json"}; {result["status"]}.')


@voice.command()
@model_options
@click.option('--judge', required=True, help='Judge model, keep fixed across comparisons.')
@click.option('--optimizer', default=None, help='Defaults to judge model.')
@click.option('--rounds', type=click.IntRange(1, 10), default=3, help='Total rounds, including previous completed rounds.')
@checked
def tune(work, writer, judge, optimizer, rounds, max_calls):
    """Keep or revert guide changes using tune documents; never opens test documents."""
    optimizer = optimizer or judge
    m = models(work, max_calls, writer, judge, optimizer)
    history = workflow.tune(work, m[writer], m[judge], m[optimizer], writer, judge, rounds)
    click.echo(json.dumps([dict(round=h['round'], accepted=h['accepted'], score=h['comparison']['score']) for h in history], indent=2))


@voice.command()
@model_options
@click.option('--judge', required=True)
@click.option('--against', type=click.Choice(['baseline', 'none']), default='baseline')
@checked
def evaluate(work, writer, judge, against, max_calls):
    """Open the held-out test once; permanently locks further tuning in this work directory."""
    m = models(work, max_calls, writer, judge)
    report = workflow.evaluate(work, m[writer], m[judge], writer, judge, against)
    click.echo(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


@voice.command()
@click.argument('brief', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--guide', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--work', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--genre', type=click.Choice(['personal', 'academic']), required=True)
@click.option('--context', multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--feedback', multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--variant', type=click.Choice(['baseline', 'tuned']), default='baseline', help='Baseline is the conservative default; choose tuned after reviewing validation.')
@click.option('--writer', default=None)
@click.option('--output', type=click.Path(path_type=Path))
@click.option('--prompt-only', is_flag=True, help='Print the prompt for an existing chat; no model call.')
@checked
def draft(brief, guide, work, genre, context, feedback, variant, writer, output, prompt_only):
    """Draft from your actual substance, with optional background files kept separate."""
    if bool(guide) == bool(work):
        raise ValueError('Choose exactly one of --guide (Markdown) or --work (trained guide)')
    if output and output.exists():
        raise ValueError('Output already exists')
    if work:
        artifact = workflow.read(work / ('baseline.json' if variant == 'baseline' else 'guide.json'))
        if artifact['genre'] != genre:
            raise ValueError('Genre differs from trained guide')
        if writer and artifact['writer'] != writer:
            raise ValueError('Writer model differs from trained guide')
        writer = writer or artifact['writer']
        workflow.bound_guide(work, load(work), writer)
        guide_text = artifact['guide']
    else:
        guide_text = guide.read_text(encoding='utf-8')
    prompt = draft_prompt(workflow.read(brief), guide_text, genre,
                          '\n\n'.join(p.read_text(encoding='utf-8') for p in context),
                          '\n\n'.join(p.read_text(encoding='utf-8') for p in feedback))
    if prompt_only:
        text = prompt
    else:
        if not writer:
            raise ValueError('Supply --writer or use --prompt-only')
        cache = work / 'cache' if work else Path('.oxidizer/cache')
        text = Model(writer, cache, max_calls=1)(prompt)
    if output:
        with output.open('x', encoding='utf-8') as f:
            f.write(text + '\n')
    else:
        click.echo(text)


@voice.command()
@click.argument('before', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('after', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--genre', type=click.Choice(['personal', 'academic']), required=True)
@click.option('--note', required=True, help='Why you made these edits, in your own words.')
@click.option('--output', type=click.Path(path_type=Path), required=True)
@checked
def feedback(before, after, genre, note, output):
    """Record your actual edits for the next guide iteration; no model call."""
    a, b = before.read_text(encoding='utf-8'), after.read_text(encoding='utf-8')
    save_new(output, dict(genre=genre, note=note, before_sha256=digest(a), after_sha256=digest(b),
                         diff='\n'.join(difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile='draft', tofile='author edit')),
                         usage='Use as explicit feedback on future drafts; never use held-out test edits to retune the same run.'))


@voice.command()
@work_option
@checked
def status(work):
    """Show corpus counts and run artifacts without calling a model."""
    data = load(work)
    click.echo(json.dumps(dict(genre=data['genre'], warnings=data['warnings'],
        splits={s: sum(p['split'] == s for p in data['posts']) for s in ['learn', 'tune', 'test']},
        trained=(work/'guide.json').exists(), test_opened=(work/'test-lock.json').exists(),
        evaluations=[p.name for p in work.glob('evaluation-*.json')]), indent=2))


@voice.command()
@work_option
@click.option('--output', type=click.Path(path_type=Path), required=True)
@click.option('--variant', type=click.Choice(['baseline', 'tuned']), default='tuned')
@checked
def export(work, output, variant):
    """Export a portable chat skill, without copying corpus samples or installing globally."""
    data = load(work)
    current = workflow.read(work / 'guide.json')
    workflow.bound_guide(work, data, current['writer'])
    artifact = workflow.read(work / 'baseline.json') if variant == 'baseline' else current
    text = (
        '---\nname: oxidizer-' + data['genre'] + '\n'
        'description: Draft or revise in this author\'s ' + data['genre'] + ' voice from a substantive brief.\n---\n\n'
        '# Author voice: ' + data['genre'] + '\n\n'
        'Writer model: `' + artifact['writer'] + '`. Variant: ' + variant + '.\n'
        'Status: ' + artifact['status'] + '. Small corpora need human review.\n\n'
        'Use the author\'s actual thesis, points, evidence, experiences and attributed quotes. '
        'Ask for missing substance; never invent personal stories, results or opinions. '
        'Keep background context separate from writing samples. Preserve uncertainty and exact data. '
        'For revision, preserve the author\'s argument and factual content. '
        'Do not treat heuristic AI-detection scores as a measure of voice quality.\n\n'
        + artifact['guide'] + '\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        f.write(text)
    click.echo(str(output))
