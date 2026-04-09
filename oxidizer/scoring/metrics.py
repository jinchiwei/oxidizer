"""Style metrics computation for Oxidizer."""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy-loaded spaCy model
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    """Return the spaCy model, loading it on first call."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Common academic transitions
# ---------------------------------------------------------------------------

ALL_TRANSITIONS: list[str] = [
    # Contrast / concession
    "however",
    "nevertheless",
    "nonetheless",
    "although",
    "even though",
    "despite",
    "in contrast",
    "on the other hand",
    "conversely",
    "yet",
    "while",
    "whereas",
    # Addition / continuation
    "furthermore",
    "moreover",
    "additionally",
    "in addition",
    "also",
    "as well",
    "besides",
    "similarly",
    "likewise",
    "equally",
    # Cause / effect
    "therefore",
    "thus",
    "hence",
    "consequently",
    "as a result",
    "as such",
    "accordingly",
    "because",
    "since",
    "so that",
    # Illustration / example
    "for example",
    "for instance",
    "specifically",
    "notably",
    "in particular",
    "namely",
    # Sequence / structure
    "first",
    "second",
    "third",
    "finally",
    "subsequently",
    "initially",
    "then",
    "next",
    "lastly",
    # Summary / conclusion
    "in summary",
    "in conclusion",
    "overall",
    "collectively",
    "taken together",
    "to summarize",
    "in brief",
    # Concession / qualification
    "given that",
    "provided that",
    "assuming that",
    "in light of",
    # Purpose / goal
    "to address",
    "to this end",
    "to achieve",
    "in order to",
    # Temporal
    "previously",
    "recently",
    "currently",
    "subsequently",
    "simultaneously",
]


# ---------------------------------------------------------------------------
# Helper: sentence tokenizer with NLTK fallback
# ---------------------------------------------------------------------------

def _sent_tokenize(text: str) -> list[str]:
    """Tokenize *text* into sentences using NLTK, downloading if needed."""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except LookupError:
        import nltk
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_sentence_lengths(text: str) -> list[int]:
    """Return a list of word counts per sentence.

    Args:
        text: Input text (may be empty).

    Returns:
        List of non-negative integers, one per sentence. Returns ``[]`` for
        empty or whitespace-only input.
    """
    text = text.strip()
    if not text:
        return []
    sentences = _sent_tokenize(text)
    return [len(s.split()) for s in sentences if s.strip()]


def compute_active_voice_ratio(text: str) -> float:
    """Estimate the fraction of sentences written in active voice.

    Uses spaCy dependency parsing; sentences that contain an ``nsubjpass``
    token are counted as passive.

    Args:
        text: Input text.

    Returns:
        Float in [0, 1]. Returns ``1.0`` for empty text (no passive detected).
    """
    text = text.strip()
    if not text:
        return 1.0

    nlp = _get_nlp()
    doc = nlp(text)

    sentences = list(doc.sents)
    total = len(sentences)
    if total == 0:
        return 1.0

    passive_count = 0
    for sent in sentences:
        if any(tok.dep_ == "nsubjpass" for tok in sent):
            passive_count += 1

    return (total - passive_count) / total


def count_banned_words(text: str, banned_list: list[str]) -> list[str]:
    """Find banned words/phrases present in *text* (case-insensitive).

    Single-word entries use a word-boundary regex; multi-word phrases use a
    simple case-insensitive substring match.

    Args:
        text: Input text.
        banned_list: Words or phrases to search for.

    Returns:
        List of entries from *banned_list* that were found in *text*. Each
        entry appears at most once, in the order it appears in *banned_list*.
    """
    if not text.strip() or not banned_list:
        return []

    found: list[str] = []
    text_lower = text.lower()

    for entry in banned_list:
        entry_lower = entry.lower()
        if " " in entry_lower:
            # Phrase: substring match
            if entry_lower in text_lower:
                found.append(entry)
        else:
            # Single word: word-boundary regex
            pattern = r"\b" + re.escape(entry_lower) + r"\b"
            if re.search(pattern, text_lower):
                found.append(entry)

    return found


def count_semicolons_per_100(text: str) -> float:
    """Count semicolons per 100 sentences.

    Args:
        text: Input text.

    Returns:
        Float. Returns ``0.0`` for empty text or text with no sentences.
    """
    text = text.strip()
    if not text:
        return 0.0

    sentences = _sent_tokenize(text)
    n_sentences = len(sentences)
    if n_sentences == 0:
        return 0.0

    semicolon_count = text.count(";")
    return semicolon_count / n_sentences * 100


def count_parentheticals_per_100(text: str) -> float:
    """Count opening parentheses per 100 sentences.

    Each ``(`` is treated as one parenthetical occurrence.

    Args:
        text: Input text.

    Returns:
        Float. Returns ``0.0`` for empty text or text with no sentences.
    """
    text = text.strip()
    if not text:
        return 0.0

    sentences = _sent_tokenize(text)
    n_sentences = len(sentences)
    if n_sentences == 0:
        return 0.0

    paren_count = text.count("(")
    return paren_count / n_sentences * 100


# Common English contractions
_CONTRACTION_PATTERN = re.compile(
    r"\b(?:"
    r"can't|cannot|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't"
    r"|haven't|hasn't|hadn't|wouldn't|couldn't|shouldn't|mightn't|mustn't"
    r"|needn't|shan't|mayn't|oughtn't"
    r"|I'm|I've|I'll|I'd|I'ma"
    r"|you're|you've|you'll|you'd"
    r"|he's|he'll|he'd"
    r"|she's|she'll|she'd"
    r"|it's|it'll|it'd"
    r"|we're|we've|we'll|we'd"
    r"|they're|they've|they'll|they'd"
    r"|that's|that'll|that'd"
    r"|there's|there'll|there'd"
    r"|what's|what'll|what'd|what've"
    r"|who's|who'll|who'd|who've"
    r"|where's|where'll|where'd"
    r"|when's|when'll|when'd"
    r"|how's|how'll|how'd"
    r"|let's"
    r")\b",
    re.IGNORECASE,
)


def count_contractions(text: str) -> int:
    """Count the number of contractions in *text*.

    Args:
        text: Input text.

    Returns:
        Non-negative integer count of contractions found.
    """
    if not text.strip():
        return 0
    return len(_CONTRACTION_PATTERN.findall(text))


def compute_transition_score(text: str, preferred: list[str]) -> float:
    """Compute the fraction of detected transitions that are in *preferred*.

    Scans *text* for any transition from :data:`ALL_TRANSITIONS`.  Of those
    found, counts how many are also in *preferred*.

    Args:
        text: Input text.
        preferred: Subset of transitions that score positively.

    Returns:
        Float in [0, 1]. Returns ``0.0`` if no transitions are detected or
        if *text* is empty.
    """
    if not text.strip():
        return 0.0

    text_lower = text.lower()
    preferred_lower = {p.lower() for p in preferred}

    detected_total = 0
    detected_preferred = 0

    for transition in ALL_TRANSITIONS:
        t_lower = transition.lower()
        # Use word-boundary for single tokens, substring for phrases
        if " " in t_lower:
            found = t_lower in text_lower
        else:
            pattern = r"\b" + re.escape(t_lower) + r"\b"
            found = bool(re.search(pattern, text_lower))

        if found:
            detected_total += 1
            if t_lower in preferred_lower:
                detected_preferred += 1

    if detected_total == 0:
        return 0.0

    return detected_preferred / detected_total
