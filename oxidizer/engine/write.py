"""Write engine: generate new academic text from a topic/prompt."""
from __future__ import annotations

from dataclasses import dataclass, field

from oxidizer.llm import call_claude
from oxidizer.profiles.schema import StyleProfile
from oxidizer.scoring.reporter import StyleReport, compute_style_report


@dataclass
class WriteResult:
    """Result of a write operation."""

    text: str
    topic: str
    section_type: str
    style_report: StyleReport | None = None
    warnings: list[str] = field(default_factory=list)


def build_write_prompt(
    topic: str,
    section_type: str,
    profile: StyleProfile,
) -> str:
    """Construct the generation prompt with style instructions and section type.

    Parameters
    ----------
    topic:
        The subject or writing instruction for the new section.
    section_type:
        One of "intro", "methods", "results", "discussion", or "other".
    profile:
        The StyleProfile defining the target writing style.

    Returns
    -------
    str
        The full prompt to send to Claude.
    """
    lines: list[str] = []

    # ------------------------------------------------------------------ #
    # System-level style instructions from profile
    # ------------------------------------------------------------------ #
    if profile.style_prompt:
        lines.append(profile.style_prompt.strip())
        lines.append("")

    # ------------------------------------------------------------------ #
    # Core rules
    # ------------------------------------------------------------------ #
    lines.append("## Style Rules")
    lines.append('- Always use first-person plural: write "we" (not "I" or passive-only constructions).')
    lines.append(
        f"- Target sentence length: mean ~{profile.sentence_length.mean:.0f} words "
        f"(std ~{profile.sentence_length.std:.0f}). Range: "
        f"{profile.sentence_length.range_min}–{profile.sentence_length.range_max} words."
    )
    semi = profile.punctuation.semicolons_per_100
    if semi > 0:
        lines.append(f"- Use semicolons purposefully; target ~{semi:.0f} per 100 sentences.")
    lines.append("- Do NOT use em dashes (—). Rewrite any em-dash constructions as separate clauses.")
    lines.append("- Do NOT use contractions (e.g. don't → do not, it's → it is).")
    lines.append("- Do NOT use passive voice except in methods/experimental sections.")

    if profile.vocabulary.banned_aiisms:
        banned_str = ", ".join(f'"{w}"' for w in profile.vocabulary.banned_aiisms)
        lines.append(f"- NEVER use these banned AI-ism words/phrases: {banned_str}.")

    if profile.transitions.preferred:
        pref_str = ", ".join(f'"{t}"' for t in profile.transitions.preferred)
        lines.append(f"- Prefer these transition words: {pref_str}.")

    if profile.vocabulary.preferred:
        vocab_str = ", ".join(f'"{w}"' for w in profile.vocabulary.preferred)
        lines.append(f"- Preferred vocabulary: {vocab_str}.")

    lines.append("- State the problem before proposing the solution.")
    lines.append("- Be precise and quantitative; do not soften claims without evidence.")
    lines.append("- Do NOT hallucinate facts, statistics, or citations. Only include information provided in the topic/prompt.")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Section type context
    # ------------------------------------------------------------------ #
    section_guidance = {
        "intro": (
            "Write an Introduction section. Establish the clinical or scientific problem, "
            "motivate the work, and close with a clear statement of objectives."
        ),
        "methods": (
            "Write a Methods section. Describe experimental or computational procedures "
            "precisely and reproducibly. Use passive voice where conventional."
        ),
        "results": (
            "Write a Results section. Report findings with quantitative precision. "
            "State what was observed; reserve interpretation for Discussion."
        ),
        "discussion": (
            "Write a Discussion or Conclusion section. Interpret the results, compare to "
            "prior work, acknowledge limitations, and state implications."
        ),
        "other": (
            "Write an academic paragraph appropriate to the topic."
        ),
    }
    guidance = section_guidance.get(section_type, section_guidance["other"])
    lines.append(f"## Section Type: {section_type.capitalize()}")
    lines.append(guidance)
    lines.append("")

    # ------------------------------------------------------------------ #
    # Few-shot examples
    # ------------------------------------------------------------------ #
    if profile.few_shot_examples:
        lines.append("## Examples of Target Style")
        for ex in profile.few_shot_examples:
            lines.append(f"[{ex.category}] {ex.text}")
        lines.append("")

    # ------------------------------------------------------------------ #
    # Task
    # ------------------------------------------------------------------ #
    lines.append("## Task")
    lines.append(
        "Generate new academic text for the topic below. "
        "Match the style described above exactly. "
        "Output ONLY the generated text — no commentary, no preamble, no markdown headers."
    )
    lines.append("")
    lines.append("### Topic / Prompt")
    lines.append(topic)

    return "\n".join(lines)


def write_section(
    topic: str,
    section_type: str,
    profile: StyleProfile,
    client=None,
    model: str = "claude-sonnet-4-20250514",
) -> WriteResult:
    """Generate a new section of academic text matching the given StyleProfile.

    Parameters
    ----------
    topic:
        The subject or writing instruction for the new section.
    section_type:
        One of "intro", "methods", "results", "discussion", or "other".
    profile:
        Target StyleProfile.
    client:
        Optional pre-built Anthropic client (for testing / CLI use). If None,
        ``llm.get_client()`` is called.
    model:
        Claude model identifier.

    Returns
    -------
    WriteResult
        Contains the generated text, inputs, and style report.

    Raises
    ------
    RuntimeError
        If no API client is available (ANTHROPIC_API_KEY not set and no client
        passed in).
    """
    prompt = build_write_prompt(topic, section_type, profile)
    generated = call_claude(prompt, model=model, max_tokens=4096, client=client)

    style_report = compute_style_report(generated, profile)

    warnings: list[str] = []
    if style_report.banned_words_found:
        warnings.append(f"Banned AI-isms found: {style_report.banned_words_found}")

    return WriteResult(
        text=generated,
        topic=topic,
        section_type=section_type,
        style_report=style_report,
        warnings=warnings,
    )
