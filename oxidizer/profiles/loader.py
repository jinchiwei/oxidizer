"""YAML profile loader for Oxidizer style profiles."""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional

from oxidizer.profiles.schema import (
    AIDetectionConfig,
    FewShotExample,
    ParagraphMetrics,
    PunctuationMetrics,
    SentenceLengthMetrics,
    StyleProfile,
    TransitionConfig,
    VocabularyConfig,
    VoiceMetrics,
    VoiceRules,
)

# Default search paths used when none are supplied
_DEFAULT_SEARCH_PATHS: list[Path] = [
    Path("./profiles"),
    Path.home() / ".oxidizer" / "profiles",
]


def resolve_profile_path(
    name: str,
    search_paths: Optional[list[Path]] = None,
) -> Optional[Path]:
    """Find a YAML profile file by name across the given search paths.

    Args:
        name: Profile stem name (e.g. ``"jinchi"``).
        search_paths: Directories to search. Falls back to ``_DEFAULT_SEARCH_PATHS``.

    Returns:
        The first matching :class:`Path`, or ``None`` if not found.
    """
    paths = search_paths if search_paths is not None else _DEFAULT_SEARCH_PATHS
    for directory in paths:
        candidate = Path(directory) / f"{name}.yaml"
        if candidate.exists():
            return candidate
    return None


def load_profile(
    name: str,
    search_paths: Optional[list[Path]] = None,
) -> StyleProfile:
    """Locate a profile by name and load it.

    Args:
        name: Profile stem name.
        search_paths: Directories to search.

    Returns:
        Parsed :class:`StyleProfile`.

    Raises:
        FileNotFoundError: If no matching YAML file is found.
    """
    path = resolve_profile_path(name, search_paths)
    if path is None:
        searched = search_paths or _DEFAULT_SEARCH_PATHS
        raise FileNotFoundError(
            f"Profile {name!r} not found in search paths: {[str(p) for p in searched]}"
        )
    return load_profile_from_path(path)


def load_profile_from_path(path: Path) -> StyleProfile:
    """Parse a YAML file into a :class:`StyleProfile`.

    Args:
        path: Absolute or relative path to the ``.yaml`` profile.

    Returns:
        Populated :class:`StyleProfile`.

    Raises:
        yaml.YAMLError: If the file contains invalid YAML.
        FileNotFoundError: If ``style_prompt_file`` is specified but missing.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)  # raises yaml.YAMLError on bad syntax

    metrics = data.get("metrics", {})

    # --- SentenceLengthMetrics ---
    sl = metrics["sentence_length"]
    sentence_length = SentenceLengthMetrics(
        mean=float(sl["mean"]),
        median=float(sl["median"]),
        std=float(sl["std"]),
        range_min=int(sl["range"][0]),
        range_max=int(sl["range"][1]),
    )

    # --- ParagraphMetrics ---
    pl = metrics["paragraph_length"]
    paragraph = ParagraphMetrics(
        mean_words=int(pl["mean"]),
        sentences_per_paragraph=list(pl["sentences_per_paragraph"]),
    )

    # --- VoiceMetrics ---
    vm = metrics["voice"]
    voice = VoiceMetrics(
        active_ratio=float(vm["active_ratio"]),
        passive_contexts=list(vm["passive_contexts"]),
    )

    # --- TransitionConfig ---
    tr = data.get("transitions", {})
    transitions = TransitionConfig(
        preferred=list(tr.get("preferred", [])),
        acceptable=list(tr.get("acceptable", [])),
    )

    # --- VocabularyConfig ---
    vc = data.get("vocabulary", {})
    vocabulary = VocabularyConfig(
        preferred=list(vc.get("preferred", [])),
        banned_aiisms=list(vc.get("banned_aiisms", [])),
    )

    # --- PunctuationMetrics (YAML key → dataclass field mapping) ---
    pm = data.get("punctuation", {})
    punctuation = PunctuationMetrics(
        semicolons_per_100=float(pm["semicolons_per_100_sentences"]),
        parentheticals_per_100=float(pm["parenthetical_pairs_per_100_sentences"]),
        em_dashes=int(pm["em_dashes"]),
        inline_enumerations=bool(pm["inline_enumerations"]),
    )

    # --- VoiceRules ---
    vr = data.get("voice_rules", {})
    voice_rules = VoiceRules(
        person=str(vr["person"]),
        hedging=list(vr.get("hedging", [])),
        reasoning=bool(vr["reasoning"]),
        problem_before_solution=bool(vr["problem_before_solution"]),
        quantitative_precision=bool(vr["quantitative_precision"]),
    )

    # --- style_prompt resolution ---
    style_prompt: Optional[str] = None
    prompt_filename = data.get("style_prompt_file")
    if prompt_filename:
        prompt_path = path.parent / prompt_filename
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"style_prompt_file {prompt_filename!r} not found relative to {path.parent}"
            )
        style_prompt = prompt_path.read_text(encoding="utf-8")

    # --- FewShotExamples ---
    few_shot_examples = [
        FewShotExample(category=ex["category"], text=ex["text"])
        for ex in data.get("few_shot_examples", [])
    ]

    # --- AIDetectionConfig ---
    ad = data.get("ai_detection", {})
    if ad:
        ai_detection = AIDetectionConfig(
            p2_density_threshold=int(ad.get("p2_density_threshold", 3)),
            p2_density_threshold_long=int(ad.get("p2_density_threshold_long", 5)),
            structural_enabled=bool(ad.get("structural_enabled", True)),
            statistical_enabled=bool(ad.get("statistical_enabled", True)),
            context_exemptions=dict(ad.get("context_exemptions", {})),
        )
    else:
        ai_detection = AIDetectionConfig()

    return StyleProfile(
        name=data["name"],
        version=int(data["version"]),
        source_documents=list(data.get("source_documents", [])),
        sentence_length=sentence_length,
        paragraph=paragraph,
        voice=voice,
        contractions=bool(metrics.get("contractions", False)),
        type_token_ratio=float(metrics.get("type_token_ratio", 0.0)),
        transitions=transitions,
        vocabulary=vocabulary,
        punctuation=punctuation,
        voice_rules=voice_rules,
        style_prompt=style_prompt,
        few_shot_examples=few_shot_examples,
        ai_detection=ai_detection,
    )
