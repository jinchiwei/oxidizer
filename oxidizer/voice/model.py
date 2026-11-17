"""Bounded model calls. No shell, tool access, silent retries or model fallback."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .corpus import digest, save_new


class Model:
    def __init__(self, spec, cache: Path, max_calls=40, transport=None, timeout=240):
        self.spec, self.cache, self.max_calls = spec, Path(cache), max_calls
        self.calls = self.hits = 0
        self.transport, self.timeout = transport, timeout

    def __call__(self, prompt):
        key = digest(dict(version=1, model=self.spec, prompt=prompt))
        path = self.cache / (key + '.json')
        if path.exists():
            value = json.loads(path.read_text(encoding='utf-8'))
            if value.get('key') != key or not isinstance(value.get('text'), str) or not value['text'].strip():
                raise ValueError('Invalid model cache entry')
            self.hits += 1
            return value['text']
        if self.calls >= self.max_calls:
            raise RuntimeError('Model call budget exhausted; successful calls are cached for resume')
        self.calls += 1
        try:
            text = self.transport(prompt, self.spec) if self.transport else self._request(prompt)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Model call timed out; partial output was not cached') from exc
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError('Model returned empty output')
        save_new(path, dict(key=key, model=self.spec, text=text.strip()))
        return text.strip()

    def _request(self, prompt):
        backend, sep, model = self.spec.partition(':')
        if not sep or not model or backend not in ('claude', 'api'):
            raise ValueError('Use an explicit claude:MODEL or api:MODEL')
        if backend == 'api':
            from anthropic import Anthropic
            response = Anthropic(max_retries=0, timeout=self.timeout).messages.create(
                model=model, max_tokens=8192, messages=[{'role': 'user', 'content': prompt}])
            if response.stop_reason != 'end_turn':
                raise RuntimeError(f'Incomplete API response: {response.stop_reason}')
            return '\n'.join(b.text for b in response.content if b.type == 'text')
        # Empty cwd plus explicit system prompt, no tools/MCP/settings or persisted session.
        # Use existing CLI login; --bare disables subscription login in recent CLI versions.
        with tempfile.TemporaryDirectory(prefix='oxidizer-voice-') as cwd:
            env = os.environ.copy()
            env.pop('CLAUDECODE', None)
            result = subprocess.run([
                'claude', '-p', '--model', model, '--effort', 'high', '--tools', '',
                '--setting-sources', '', '--strict-mcp-config', '--disable-slash-commands',
                '--no-session-persistence', '--output-format', 'json', '--max-budget-usd', '1',
                '--system-prompt', 'You are a writing and evaluation assistant. Follow the supplied task only.',
            ], input=prompt, text=True, capture_output=True, cwd=cwd, env=env, timeout=self.timeout)
        if result.returncode:
            raise RuntimeError(f'Claude CLI failed (exit {result.returncode}); no automatic retry')
        response = json.loads(result.stdout)
        if response.get('is_error') or response.get('subtype') != 'success':
            raise RuntimeError('Claude CLI did not complete successfully; check login, budget or usage limits')
        return response.get('result', '')
