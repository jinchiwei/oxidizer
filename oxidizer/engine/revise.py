"""Revise engine: restyle existing text to match a StyleProfile."""
from __future__ import annotations

from oxidizer.engine.pipeline import (
    PipelineResult,
    extract_and_lock,
    verify_and_score,
)
from oxidizer.llm import call_claude
from oxidizer.parsers.markdown_parser import Section
from oxidizer.profiles.schema import StyleProfile


def build_restyle_prompt(
    section: Section,
    profile: StyleProfile,
    locked_entities: list[str],
) -> str:
    """Construct the restyle prompt with style instructions, locked entities, and examples.

    Parameters
    ----------
    section:
        The parsed markdown section containing heading, body, and context.
    profile:
        The StyleProfile defining the target writing style.
    locked_entities:
        Flat list of entity strings that must appear verbatim in the output.

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
    lines.append(f"- Always use first-person plural: write \"we\" (not \"I\" or passive-only constructions).")
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
    lines.append("")

    # ------------------------------------------------------------------ #
    # Section context
    # ------------------------------------------------------------------ #
    if section.heading:
        lines.append(f"## Section Context")
        lines.append(f"This is a **{section.context}** section titled \"{section.heading}\".")
        lines.append("")

    # ------------------------------------------------------------------ #
    # Locked entities
    # ------------------------------------------------------------------ #
    if locked_entities:
        lines.append("## Locked Entities (must appear verbatim in output)")
        lines.append(
            "The following entities MUST appear in your output exactly as shown. "
            "Do not paraphrase, abbreviate, or omit them:"
        )
        for entity in locked_entities:
            lines.append(f"  - {entity}")
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
        "Rewrite the following text to match the style described above. "
        "Preserve all factual content, data, and citations exactly. "
        "Output ONLY the restyled text — no commentary, no preamble, no markdown headers."
    )
    lines.append("")
    lines.append("### Original Text")
    lines.append(section.body)

    return "\n".join(lines)


def revise_section(
    section: Section,
    profile: StyleProfile,
    client=None,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 2,
) -> PipelineResult:
    """Restyle a single section to match the given StyleProfile.

    Parameters
    ----------
    section:
        The parsed Section to revise.
    profile:
        Target StyleProfile.
    client:
        Optional pre-built Anthropic client (for testing / CLI use). If None,
        ``llm.get_client()`` is called.
    model:
        Claude model identifier.
    max_retries:
        Number of additional attempts if entity preservation fails.

    Returns
    -------
    PipelineResult
        Contains the restyled text, entity info, preservation result, and
        style report.
    """
    # 1. Extract locked entities from the original text
    entities, locked = extract_and_lock(section.body)

    # 2. Build prompt
    prompt = build_restyle_prompt(section, profile, locked)

    # 3. Call API with retry loop for entity preservation
    restyled = ""
    retries = 0

    for attempt in range(max_retries + 1):
        restyled = call_claude(prompt, model=model, max_tokens=4096, client=client)

        # Check preservation — if passed or on last attempt, stop
        from oxidizer.preservation.checker import check_entity_preservation
        check = check_entity_preservation(entities, restyled)
        if check.passed or attempt == max_retries:
            break

        # Build a targeted retry prompt
        missing_str = "\n".join(f"  - {e}" for e in check.missing)
        prompt = (
            f"{prompt}\n\n"
            f"## Retry Instruction\n"
            f"Your previous output was missing these locked entities:\n{missing_str}\n"
            f"Rewrite the text again ensuring ALL listed entities appear verbatim."
        )
        retries += 1

    # 4. Verify and score final output
    preservation, style_report, warnings = verify_and_score(restyled, entities, profile)

    return PipelineResult(
        text=restyled,
        original_text=section.body,
        heading=section.heading,
        entities=entities,
        preservation=preservation,
        style_report=style_report,
        retries=retries,
        warnings=warnings,
    )
