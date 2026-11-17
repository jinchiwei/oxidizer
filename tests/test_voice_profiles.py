from dataclasses import replace
from pathlib import Path

from oxidizer.engine.write import build_write_prompt
from oxidizer.engine.revise import build_restyle_prompt
from oxidizer.parsers.markdown_parser import Section
from oxidizer.profiles.loader import load_profile_from_path
from oxidizer.scoring.reporter import compute_style_report


def personal():
    p = load_profile_from_path(Path('profiles/jinchi.yaml'))
    return replace(p, style_prompt=None, contractions=True,
                   voice_rules=replace(p.voice_rules, person='I'),
                   punctuation=replace(p.punctuation, em_dashes=1))


def test_personal_write_does_not_force_academic_voice():
    prompt = build_write_prompt('My supplied opinion.', 'personal', personal())
    assert 'Always use first-person plural' not in prompt
    assert 'Do NOT use contractions' not in prompt
    assert 'Do NOT use em dashes' not in prompt
    assert 'Generate new academic text' not in prompt
    assert '"I"' in prompt


def test_revise_respects_profile_voice():
    prompt = build_restyle_prompt(Section(heading='', level=0, body="I don't agree.", context='other'), personal(), [])
    assert 'Always use first-person plural' not in prompt
    assert 'Do NOT use contractions' not in prompt
    assert '"I"' in prompt


def test_allowed_contractions_are_not_penalized():
    report = compute_style_report("I don't agree. It's unclear.", personal())
    assert report.contraction_count == 2
    assert report.sub_scores['contractions'] == 1.0
