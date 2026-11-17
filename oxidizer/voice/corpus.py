"""Freeze local writing samples by publication, never by paragraph."""
import hashlib
import json
import re
from datetime import date
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(manifest: Path, genre: str) -> dict:
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('Corpus manifest must be a list of documents')
    posts, ids, hashes = [], set(), set()
    for row in rows:
        if row.get('genre') != genre:
            continue
        for key in ('id', 'group', 'source', 'path', 'date'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Missing document {key}')
        if row.get('authorship') != 'author':
            raise ValueError('Confirm authorship before using a document as voice evidence')
        date.fromisoformat(row['date'])
        if row['id'] in ids:
            raise ValueError('duplicate document id')
        ids.add(row['id'])
        path = manifest.parent / row['path']
        text = path.read_text(encoding='utf-8').strip()
        text = re.sub(r'\A---\r?\n.*?\r?\n---(?:\r?\n|$)', '', text, count=1, flags=re.S).strip()
        if len(text.split()) < 5:
            raise ValueError(f'Empty/too short document: {row["id"]}')
        sha = digest(' '.join(text.lower().split()))
        if sha in hashes:
            raise ValueError('duplicate document content; remove copied/revised versions or group them manually')
        hashes.add(sha)
        posts.append({k: row[k] for k in ('id', 'group', 'source', 'date')} | {'text': text, 'sha256': sha})
    groups = sorted({p['group'] for p in posts}, key=lambda g: (max(p['date'] for p in posts if p['group'] == g), g))
    if len(groups) < 3:
        raise ValueError('Need at least 3 independent publication groups for learn/tune/test')
    n = max(1, len(groups) // 5)
    test, tune = set(groups[-n:]), set(groups[-2*n:-n])
    for post in posts:
        post['split'] = 'test' if post['group'] in test else 'tune' if post['group'] in tune else 'learn'
    posts.sort(key=lambda p: (p['date'], p['id']))
    return dict(version=1, genre=genre, posts=posts, fingerprint=digest(posts), warnings=(
        ['Small corpus: pilot only; collect more independent author-written documents before claiming generalization.']
        if len(groups) < 12 else []))


def load(work: Path):
    data = json.loads((work / 'corpus.json').read_text(encoding='utf-8'))
    if digest(data['posts']) != data['fingerprint']:
        raise ValueError('Frozen corpus changed; prepare a new work directory')
    return data


def save_new(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
