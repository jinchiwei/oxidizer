"""Optional Claude API client. Only used when ANTHROPIC_API_KEY is set."""
import os


def is_api_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_client():
    if not is_api_available():
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        return None


def call_claude(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    client=None,
):
    if client is None:
        client = get_client()
    if client is None:
        raise RuntimeError(
            "Claude API not available. Either use Claude Code (/oxidize revise) "
            "or set ANTHROPIC_API_KEY for CLI batch mode."
        )
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
