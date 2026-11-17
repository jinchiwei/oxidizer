import pytest


def test_missing_tokenizer_does_not_download_into_user_home(monkeypatch):
    import nltk
    import nltk.tokenize
    from oxidizer.scoring.metrics import _sent_tokenize
    def missing(*args, **kwargs):
        raise LookupError('missing fixture resource')
    def download(*args, **kwargs):
        pytest.fail('Scoring must not install resources implicitly')
    monkeypatch.setattr(nltk.tokenize, 'sent_tokenize', missing)
    monkeypatch.setattr(nltk, 'download', download)
    with pytest.raises(RuntimeError, match='environment'):
        _sent_tokenize('A sentence.')
