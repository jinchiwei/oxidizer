# Oxidizer Phase 1: Expanded Authorship Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI + Claude Code skill for academic writing style transfer. Revise existing documents, write new sections, scan for AI-isms, diff changes, generate visual reports, validate and compare profiles. All in the author's personal voice.

**Architecture:** Two execution paths. **Claude Code-native (default, no API key needed):** The `/oxidize` skill handles revise and write by orchestrating Claude in-conversation. Python CLI handles local-only commands (score, scan, diff, compare, validate-profile). **Optional API mode:** If `ANTHROPIC_API_KEY` is set, revise/write also work from CLI for batch processing. The `anthropic` package is an optional dependency.

**Tech Stack:** Python 3.12, Click (CLI), python-docx (DOCX parsing), pdfplumber (PDF best-effort), spaCy (NER + voice detection), nltk (tokenization), PyYAML, Rich (output formatting), pytest (testing). Optional: Anthropic SDK (for API batch mode).

**CEO Review Decisions (2026-04-08):**
- 6 scope expansions accepted: write mode, de-AI scan, style diff, HTML report, profile self-test, profile compare
- LaTeX parser added (Codex recommendation)
- Claude Code-native restyling is default (no API key)
- Scoring weights documented as empirical with calibration plan
- See CEO plan: `~/.gstack/projects/jinchiwei-oxidizer/ceo-plans/2026-04-08-oxidizer-phase1.md`

**Eng Review Findings (2026-04-08):**
- Add `oxidizer/engine/pipeline.py` — shared orchestration layer for revise and write engines. Extract the parse → extract → restyle → verify → score flow from revise.py into a reusable pipeline. Both revise.py and write.py call pipeline functions.
- Add `tests/conftest.py` — shared `make_test_profile()` fixture (DRY: was duplicated in test_reporter.py and test_revise.py)
- Expand author-year citation regex to handle multi-author: `(Smith and Jones, 2024)`, `(Smith, Jones, & Wei, 2024)`
- Add 11 missing test cases:
  1. `test_loader.py`: invalid YAML file (malformed), missing style_prompt_file
  2. `test_parsers.py`: DOCX happy path (create a fixture DOCX file)
  3. `test_extractor.py`: multi-author citations, unicode ± vs ASCII +/-
  4. `test_metrics.py`: empty text input to all metric functions
  5. `test_revise.py`: retry on entity preservation failure (mock failing then succeeding)
  6. `test_write.py`: no API key error handling
  7. `test_diff.py`, `test_validate_compare.py`: deeper assertion tests for diff/compare/validate
- See eng review dashboard: CEO + ENG CLEARED

---

### Task 1: Package Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `oxidizer/__init__.py`
- Create: `oxidizer/profiles/__init__.py`
- Create: `oxidizer/engine/__init__.py`
- Create: `oxidizer/preservation/__init__.py`
- Create: `oxidizer/scoring/__init__.py`
- Create: `oxidizer/parsers/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/sample_methods.md`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "oxidizer"
version = "0.1.0"
description = "Academic writing style engine"
requires-python = ">=3.12"
dependencies = [
    "click>=8.0",
    "python-docx>=1.0",
    "pdfplumber>=0.11",
    "spacy>=3.7",
    "nltk>=3.8",
    "pyyaml>=6.0",
    "rich>=13.0",
]

[project.optional-dependencies]
api = [
    "anthropic>=0.40",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "anthropic>=0.40",
]

[project.scripts]
oxidizer = "oxidizer.cli:cli"

[tool.setuptools.packages.find]
include = ["oxidizer*"]
```

- [ ] **Step 2: Create package __init__.py files**

`oxidizer/__init__.py`:
```python
"""Oxidizer: Academic writing style engine."""

__version__ = "0.1.0"
```

Create empty `__init__.py` in each subpackage:
- `oxidizer/profiles/__init__.py`
- `oxidizer/engine/__init__.py`
- `oxidizer/preservation/__init__.py`
- `oxidizer/scoring/__init__.py`
- `oxidizer/parsers/__init__.py`
- `tests/__init__.py`

- [ ] **Step 3: Create test fixture**

`tests/fixtures/sample_methods.md`:
```markdown
## Methods

We collected data from 76 patients (42 female, 34 male; mean age 62.3 +/- 8.1 years) who underwent pretreatment MRI at UCSF between 2019 and 2023. All patients provided informed consent as approved by the institutional review board (IRB #19-28456).

Imaging was performed on a 3T Siemens Prisma scanner using a standardized protocol. T1-weighted images were acquired with the following parameters: TR = 2300 ms, TE = 2.32 ms, voxel size = 1.0 x 1.0 x 1.0 mm [1]. T2-FLAIR sequences were also obtained for each patient [2, 3].

Radiomics features were extracted using PyRadiomics (v3.0.1) from manually segmented regions of interest (ROIs). A total of 107 features were computed across five categories: first-order statistics, shape-based, gray-level co-occurrence matrix (GLCM), gray-level run-length matrix (GLRLM), and gray-level size-zone matrix (GLSZM).
```

- [ ] **Step 4: Install dependencies**

Run: `conda run -n oxidizer pip install -e ".[dev]" 2>&1 | tail -5`
Expected: Successful installation

Run: `conda run -n oxidizer python -m spacy download en_core_web_sm 2>&1 | tail -3`
Expected: Model downloaded successfully

Run: `conda run -n oxidizer python -c "import nltk; nltk.download('punkt_tab', quiet=True); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Verify package imports**

Run: `conda run -n oxidizer python -c "import oxidizer; print(oxidizer.__version__)"`
Expected: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml oxidizer/ tests/ docs/
git commit -m "scaffold: package structure with pyproject.toml and test fixtures"
```

---

### Task 2: Style Profile Schema

**Files:**
- Create: `oxidizer/profiles/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

`tests/test_schema.py`:
```python
from oxidizer.profiles.schema import (
    StyleProfile,
    SentenceLengthMetrics,
    ParagraphMetrics,
    VoiceMetrics,
    PunctuationMetrics,
    TransitionConfig,
    VocabularyConfig,
    VoiceRules,
    FewShotExample,
)


def test_sentence_length_metrics():
    m = SentenceLengthMetrics(mean=24.1, median=23, std=10.1, range_min=6, range_max=77)
    assert m.mean == 24.1
    assert m.std == 10.1


def test_style_profile_has_all_fields():
    profile = StyleProfile(
        name="Test Author",
        version=1,
        source_documents=["Paper 1"],
        sentence_length=SentenceLengthMetrics(
            mean=24.1, median=23, std=10.1, range_min=6, range_max=77
        ),
        paragraph=ParagraphMetrics(mean_words=71, sentences_per_paragraph=(3, 4)),
        voice=VoiceMetrics(active_ratio=0.90, passive_contexts=["methods"]),
        contractions=False,
        type_token_ratio=0.346,
        transitions=TransitionConfig(
            preferred=["while", "however"],
            acceptable=["furthermore"],
        ),
        vocabulary=VocabularyConfig(
            preferred=["robust"],
            banned_aiisms=["delve", "tapestry"],
        ),
        punctuation=PunctuationMetrics(
            semicolons_per_100=12,
            parentheticals_per_100=25,
            em_dashes=0,
            inline_enumerations=True,
        ),
        voice_rules=VoiceRules(
            person="we",
            hedging=["would likely"],
            reasoning=True,
            problem_before_solution=True,
            quantitative_precision=True,
        ),
        style_prompt="You are writing in the voice of...",
        few_shot_examples=[
            FewShotExample(category="problem_framing", text="Example text here.")
        ],
    )
    assert profile.name == "Test Author"
    assert profile.voice.active_ratio == 0.90
    assert len(profile.vocabulary.banned_aiisms) == 2
    assert profile.punctuation.em_dashes == 0


def test_banned_words_are_lowercase():
    vocab = VocabularyConfig(
        preferred=["robust"],
        banned_aiisms=["Delve", "TAPESTRY", "multifaceted"],
    )
    # All banned words should be stored lowercase for case-insensitive matching
    assert all(w == w.lower() for w in vocab.banned_aiisms)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oxidizer.profiles.schema'`

- [ ] **Step 3: Write the implementation**

`oxidizer/profiles/schema.py`:
```python
"""Style profile data structures."""

from dataclasses import dataclass, field


@dataclass
class SentenceLengthMetrics:
    mean: float
    median: float
    std: float
    range_min: int
    range_max: int


@dataclass
class ParagraphMetrics:
    mean_words: float
    sentences_per_paragraph: tuple[int, int]


@dataclass
class VoiceMetrics:
    active_ratio: float
    passive_contexts: list[str] = field(default_factory=list)


@dataclass
class PunctuationMetrics:
    semicolons_per_100: float
    parentheticals_per_100: float
    em_dashes: int
    inline_enumerations: bool


@dataclass
class TransitionConfig:
    preferred: list[str] = field(default_factory=list)
    acceptable: list[str] = field(default_factory=list)


@dataclass
class VocabularyConfig:
    preferred: list[str] = field(default_factory=list)
    banned_aiisms: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.banned_aiisms = [w.lower() for w in self.banned_aiisms]


@dataclass
class VoiceRules:
    person: str
    hedging: list[str] = field(default_factory=list)
    reasoning: bool = True
    problem_before_solution: bool = True
    quantitative_precision: bool = True


@dataclass
class FewShotExample:
    category: str
    text: str


@dataclass
class StyleProfile:
    name: str
    version: int
    source_documents: list[str]
    sentence_length: SentenceLengthMetrics
    paragraph: ParagraphMetrics
    voice: VoiceMetrics
    contractions: bool
    type_token_ratio: float
    transitions: TransitionConfig
    vocabulary: VocabularyConfig
    punctuation: PunctuationMetrics
    voice_rules: VoiceRules
    style_prompt: str
    few_shot_examples: list[FewShotExample] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n oxidizer python -m pytest tests/test_schema.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add oxidizer/profiles/schema.py tests/test_schema.py
git commit -m "feat: style profile schema dataclasses"
```

---

### Task 3: Profile YAML Loader

**Files:**
- Create: `oxidizer/profiles/loader.py`
- Create: `tests/test_loader.py`
- Create: `profiles/jinchi.yaml`
- Create: `profiles/custom_style_prompt.md` (copy from dev/src/)

- [ ] **Step 1: Create the jinchi.yaml profile**

`profiles/jinchi.yaml`:
```yaml
name: "Jinchi Wei"
version: 1
source_documents:
  - "MSE Thesis (2021)"
  - "SPIE Conference Paper (2024)"
  - "ARIA Perspective Paper (2025)"

metrics:
  sentence_length:
    mean: 24.1
    median: 23
    std: 10.1
    range: [6, 77]
  paragraph_length:
    mean: 71
    sentences_per_paragraph: [3, 4]
  voice:
    active_ratio: 0.90
    passive_contexts: ["methods", "experimental setup"]
  contractions: false
  type_token_ratio: 0.346

transitions:
  preferred: ["while", "however", "additionally", "as such", "as a result", "similarly", "to address", "collectively"]
  acceptable: ["furthermore", "notably", "conversely", "given that"]

vocabulary:
  preferred: ["attained", "robust", "scalable", "holistic", "readily", "promising", "building upon", "as well"]
  banned_aiisms:
    - "delve"
    - "tapestry"
    - "multifaceted"
    - "pivotal"
    - "groundbreaking"
    - "cutting-edge"
    - "transformative"
    - "unravel"
    - "plethora"
    - "myriad"
    - "foster"
    - "harness"
    - "harnessing"
    - "elevate"
    - "streamline"
    - "realm"
    - "intricacies"
    - "synergy"
    - "innovative"
    - "novel"
    - "embark"
    - "navigate"
    - "landscape"
    - "cornerstone"
    - "at its core"
    - "it's worth noting"
    - "it's important to note"
    - "in today's world"

punctuation:
  semicolons_per_100_sentences: 12
  parenthetical_pairs_per_100_sentences: 25
  em_dashes: 0
  inline_enumerations: true

voice_rules:
  person: "we"
  hedging: ["would likely", "could theoretically", "most likely"]
  reasoning: true
  problem_before_solution: true
  quantitative_precision: true

style_prompt_file: "custom_style_prompt.md"

few_shot_examples:
  - category: "problem_framing"
    text: "The negative results of these circumstances include inefficient delegation of physician efforts and time, billions of dollars in unnecessary clinical spending, and ultimately poorer health outcomes."
  - category: "methods_with_reasoning"
    text: "We realized there were two main problems with our goal of both segmenting and anatomically labeling individual vertebra: 1. Vertebra are not very distinct in appearance from their neighbors, creating problems distinguishing vertebra from each other, and 2. ground truth in 3D generally includes all of the vertebrae, including body and projections."
  - category: "results_with_precision"
    text: "Registration error using only the transverse scans was 1.43 +/- 0.30 mm."
  - category: "honest_limitations"
    text: "While only 2 images were correctly classified, 11/16 images were classified as Stages 3 or 5, showing that the network recognized the true labels were around 4 but was not precise enough to make the final distinction."
  - category: "clinical_review"
    text: "Compromised vessel integrity can result in vasogenic edema and extraparenchymal effusion (ARIA-E), microhemorrhages and superficial siderosis (ARIA-H), or a combination thereof."
```

- [ ] **Step 2: Copy custom_style_prompt.md to profiles/**

Run: `cp dev/src/custom_style_prompt.md profiles/custom_style_prompt.md`

- [ ] **Step 3: Write the failing test**

`tests/test_loader.py`:
```python
import os
from pathlib import Path

from oxidizer.profiles.loader import load_profile, resolve_profile_path
from oxidizer.profiles.schema import StyleProfile


FIXTURES_DIR = Path(__file__).parent.parent / "profiles"


def test_resolve_profile_path_local():
    """Profile resolution finds project-local profiles/ directory."""
    path = resolve_profile_path("jinchi", search_paths=[FIXTURES_DIR])
    assert path is not None
    assert path.name == "jinchi.yaml"
    assert path.exists()


def test_resolve_profile_path_not_found():
    """Missing profile returns None."""
    path = resolve_profile_path("nonexistent", search_paths=[FIXTURES_DIR])
    assert path is None


def test_load_profile_from_yaml():
    """Load jinchi.yaml and verify all fields are populated."""
    profile = load_profile("jinchi", search_paths=[FIXTURES_DIR])
    assert isinstance(profile, StyleProfile)
    assert profile.name == "Jinchi Wei"
    assert profile.version == 1
    assert len(profile.source_documents) == 3
    assert profile.sentence_length.mean == 24.1
    assert profile.sentence_length.std == 10.1
    assert profile.voice.active_ratio == 0.90
    assert profile.contractions is False
    assert profile.type_token_ratio == 0.346
    assert "while" in profile.transitions.preferred
    assert "delve" in profile.vocabulary.banned_aiisms
    assert profile.punctuation.semicolons_per_100 == 12
    assert profile.punctuation.em_dashes == 0
    assert profile.voice_rules.person == "we"
    assert len(profile.few_shot_examples) == 5
    assert profile.few_shot_examples[0].category == "problem_framing"


def test_load_profile_includes_style_prompt():
    """style_prompt_file is resolved and loaded into the style_prompt field."""
    profile = load_profile("jinchi", search_paths=[FIXTURES_DIR])
    assert "first person plural" in profile.style_prompt.lower()
    assert len(profile.style_prompt) > 100


def test_banned_words_normalized_to_lowercase():
    """All banned AI-isms are lowercased regardless of YAML casing."""
    profile = load_profile("jinchi", search_paths=[FIXTURES_DIR])
    for word in profile.vocabulary.banned_aiisms:
        assert word == word.lower(), f"Banned word not lowercase: {word}"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 5: Write the implementation**

`oxidizer/profiles/loader.py`:
```python
"""Load style profiles from YAML files."""

from pathlib import Path

import yaml

from oxidizer.profiles.schema import (
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

DEFAULT_SEARCH_PATHS = [
    Path("./profiles"),
    Path.home() / ".oxidizer" / "profiles",
]


def resolve_profile_path(
    name: str,
    search_paths: list[Path] | None = None,
) -> Path | None:
    """Find a profile YAML file by name across search paths."""
    paths = search_paths or DEFAULT_SEARCH_PATHS
    for base in paths:
        candidate = base / f"{name}.yaml"
        if candidate.exists():
            return candidate
    return None


def load_profile(
    name: str,
    search_paths: list[Path] | None = None,
) -> StyleProfile:
    """Load a style profile from YAML by name."""
    path = resolve_profile_path(name, search_paths)
    if path is None:
        raise FileNotFoundError(
            f"Profile '{name}' not found in: {search_paths or DEFAULT_SEARCH_PATHS}"
        )
    return load_profile_from_path(path)


def load_profile_from_path(path: Path) -> StyleProfile:
    """Load a style profile from a specific YAML file path."""
    with open(path) as f:
        data = yaml.safe_load(f)

    metrics = data["metrics"]
    sl = metrics["sentence_length"]
    sl_range = sl.get("range", [0, 100])

    # Resolve style_prompt_file relative to the profile directory
    style_prompt = ""
    prompt_file = data.get("style_prompt_file")
    if prompt_file:
        prompt_path = path.parent / prompt_file
        if prompt_path.exists():
            style_prompt = prompt_path.read_text()

    punct = data.get("punctuation", {})

    return StyleProfile(
        name=data["name"],
        version=data.get("version", 1),
        source_documents=data.get("source_documents", []),
        sentence_length=SentenceLengthMetrics(
            mean=sl["mean"],
            median=sl["median"],
            std=sl["std"],
            range_min=sl_range[0],
            range_max=sl_range[1],
        ),
        paragraph=ParagraphMetrics(
            mean_words=metrics["paragraph_length"]["mean"],
            sentences_per_paragraph=tuple(
                metrics["paragraph_length"]["sentences_per_paragraph"]
            ),
        ),
        voice=VoiceMetrics(
            active_ratio=metrics["voice"]["active_ratio"],
            passive_contexts=metrics["voice"].get("passive_contexts", []),
        ),
        contractions=metrics.get("contractions", False),
        type_token_ratio=metrics.get("type_token_ratio", 0.0),
        transitions=TransitionConfig(
            preferred=data.get("transitions", {}).get("preferred", []),
            acceptable=data.get("transitions", {}).get("acceptable", []),
        ),
        vocabulary=VocabularyConfig(
            preferred=data.get("vocabulary", {}).get("preferred", []),
            banned_aiisms=data.get("vocabulary", {}).get("banned_aiisms", []),
        ),
        punctuation=PunctuationMetrics(
            semicolons_per_100=punct.get("semicolons_per_100_sentences", 0),
            parentheticals_per_100=punct.get("parenthetical_pairs_per_100_sentences", 0),
            em_dashes=punct.get("em_dashes", 0),
            inline_enumerations=punct.get("inline_enumerations", False),
        ),
        voice_rules=VoiceRules(
            person=data.get("voice_rules", {}).get("person", "we"),
            hedging=data.get("voice_rules", {}).get("hedging", []),
            reasoning=data.get("voice_rules", {}).get("reasoning", True),
            problem_before_solution=data.get("voice_rules", {}).get(
                "problem_before_solution", True
            ),
            quantitative_precision=data.get("voice_rules", {}).get(
                "quantitative_precision", True
            ),
        ),
        style_prompt=style_prompt,
        few_shot_examples=[
            FewShotExample(category=ex["category"], text=ex["text"])
            for ex in data.get("few_shot_examples", [])
        ],
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `conda run -n oxidizer python -m pytest tests/test_loader.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add profiles/ oxidizer/profiles/loader.py tests/test_loader.py
git commit -m "feat: YAML profile loader with jinchi.yaml profile"
```

---

### Task 4: Document Parsers (Markdown + DOCX)

**Files:**
- Create: `oxidizer/parsers/markdown_parser.py`
- Create: `oxidizer/parsers/docx_parser.py`
- Create: `oxidizer/parsers/pdf_parser.py`
- Create: `tests/test_parsers.py`
- Create: `tests/fixtures/sample_paper.docx`

- [ ] **Step 1: Write the failing test**

`tests/test_parsers.py`:
```python
from pathlib import Path

from oxidizer.parsers.markdown_parser import parse_markdown
from oxidizer.parsers.docx_parser import parse_docx


FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_markdown_extracts_sections():
    """Parse a markdown file into named sections."""
    text = FIXTURES.joinpath("sample_methods.md").read_text()
    sections = parse_markdown(text)
    assert len(sections) >= 1
    assert sections[0].heading == "Methods"
    assert "76 patients" in sections[0].body
    assert "PyRadiomics" in sections[0].body


def test_parse_markdown_multiple_sections():
    """Parse markdown with multiple headings."""
    text = """## Introduction

This is the introduction.

## Methods

We collected data from 50 subjects.

## Results

Accuracy was 0.95.
"""
    sections = parse_markdown(text)
    assert len(sections) == 3
    assert sections[0].heading == "Introduction"
    assert sections[1].heading == "Methods"
    assert sections[2].heading == "Results"
    assert "0.95" in sections[2].body


def test_parse_markdown_no_headings():
    """Text without headings becomes a single unnamed section."""
    text = "Just a plain paragraph with no headings at all."
    sections = parse_markdown(text)
    assert len(sections) == 1
    assert sections[0].heading == ""
    assert "plain paragraph" in sections[0].body


def test_section_context_mapping():
    """Sections get mapped to context types for style adjustment."""
    text = """## Introduction

Intro text.

## Materials and Methods

Methods text.

## Results

Results text.

## Discussion

Discussion text.
"""
    sections = parse_markdown(text)
    assert sections[0].context == "intro"
    assert sections[1].context == "methods"
    assert sections[2].context == "results"
    assert sections[3].context == "discussion"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_parsers.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the markdown parser**

`oxidizer/parsers/markdown_parser.py`:
```python
"""Parse Markdown documents into sections."""

import re
from dataclasses import dataclass

CONTEXT_MAP = {
    "introduction": "intro",
    "background": "intro",
    "materials and methods": "methods",
    "methods": "methods",
    "materials": "methods",
    "experimental": "methods",
    "experimental setup": "methods",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "discussion",
    "conclusions": "discussion",
}


@dataclass
class Section:
    heading: str
    body: str
    context: str  # "intro", "methods", "results", "discussion", "other"
    level: int  # heading level (1-6), 0 if no heading


def _classify_context(heading: str) -> str:
    """Map a heading string to a section context type."""
    lower = heading.lower().strip()
    for key, ctx in CONTEXT_MAP.items():
        if key in lower:
            return ctx
    return "other"


def parse_markdown(text: str) -> list[Section]:
    """Parse markdown text into a list of Sections split on headings."""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [Section(heading="", body=text.strip(), context="other", level=0)]

    sections = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        context = _classify_context(heading)
        sections.append(Section(heading=heading, body=body, context=context, level=level))

    return sections
```

- [ ] **Step 4: Write the DOCX parser**

`oxidizer/parsers/docx_parser.py`:
```python
"""Parse DOCX documents into sections."""

from pathlib import Path

from docx import Document

from oxidizer.parsers.markdown_parser import Section, _classify_context


def parse_docx(path: Path) -> list[Section]:
    """Parse a .docx file into sections based on heading styles."""
    doc = Document(str(path))

    sections: list[Section] = []
    current_heading = ""
    current_body_parts: list[str] = []
    current_level = 0

    for para in doc.paragraphs:
        style_name = (para.style.name or "").lower()

        if style_name.startswith("heading"):
            # Save previous section if it has content
            if current_body_parts:
                body = "\n\n".join(current_body_parts)
                context = _classify_context(current_heading)
                sections.append(
                    Section(
                        heading=current_heading,
                        body=body,
                        context=context,
                        level=current_level,
                    )
                )
                current_body_parts = []

            current_heading = para.text.strip()
            # Extract heading level from style name (e.g., "heading 2" -> 2)
            try:
                current_level = int(style_name.split()[-1])
            except (ValueError, IndexError):
                current_level = 1

        elif para.text.strip():
            current_body_parts.append(para.text.strip())

    # Don't forget the last section
    if current_body_parts:
        body = "\n\n".join(current_body_parts)
        context = _classify_context(current_heading)
        sections.append(
            Section(
                heading=current_heading,
                body=body,
                context=context,
                level=current_level,
            )
        )

    # If no headings were found, return everything as one section
    if not sections:
        full_text = "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        sections = [Section(heading="", body=full_text, context="other", level=0)]

    return sections
```

- [ ] **Step 5: Create a stub PDF parser**

`oxidizer/parsers/pdf_parser.py`:
```python
"""Parse PDF documents into sections (best-effort)."""

from pathlib import Path

from oxidizer.parsers.markdown_parser import Section


def parse_pdf(path: Path) -> list[Section]:
    """Parse a PDF file into sections. Best-effort; prefer DOCX input.

    PDF parsing for academic papers with two-column layouts, figures, and
    equations is unreliable. This extracts raw text and attempts basic
    section detection via font-size heuristics.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF parsing: pip install pdfplumber")

    full_text_parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)

    if not full_text.strip():
        return [Section(heading="", body="", context="other", level=0)]

    # For now, return as a single section. Heading detection in PDFs
    # requires font-size analysis which is fragile. Users should use
    # --sections flag to specify sections manually, or convert to DOCX.
    return [Section(heading="", body=full_text.strip(), context="other", level=0)]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `conda run -n oxidizer python -m pytest tests/test_parsers.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add oxidizer/parsers/ tests/test_parsers.py
git commit -m "feat: markdown, DOCX, and PDF document parsers"
```

---

### Task 5: Entity Extractor

**Files:**
- Create: `oxidizer/preservation/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write the failing test**

`tests/test_extractor.py`:
```python
from oxidizer.preservation.extractor import extract_entities, LockedEntities


def test_extract_numbered_citations():
    text = "Previous work [1] showed that the method [2, 3] outperformed baselines [4-6]."
    entities = extract_entities(text)
    assert "[1]" in entities.citations
    assert "[2, 3]" in entities.citations
    assert "[4-6]" in entities.citations


def test_extract_author_year_citations():
    text = "As shown by Smith et al. (2024), the approach of Jones (2023) was validated."
    entities = extract_entities(text)
    assert any("Smith" in c for c in entities.citations)
    assert any("Jones" in c for c in entities.citations)


def test_extract_numbers_with_units():
    text = "Error was 1.43 +/- 0.30 mm. Accuracy reached 95.2%. Mean age was 62.3 years."
    entities = extract_entities(text)
    assert any("1.43" in n for n in entities.numbers)
    assert any("0.30" in n for n in entities.numbers)
    assert any("95.2" in n for n in entities.numbers)


def test_extract_error_margins():
    text = "Registration error was 1.22 +/- 0.39 mm and 16.62 +/- 7.04 mm."
    entities = extract_entities(text)
    # Error margins should be captured as complete expressions
    assert any("1.22" in n and "0.39" in n for n in entities.numbers)


def test_extract_abbreviations():
    text = (
        "We used gray-level co-occurrence matrix (GLCM) features. "
        "The GLCM values were normalized. "
        "Amyloid-related imaging abnormalities (ARIA) were detected."
    )
    entities = extract_entities(text)
    assert "GLCM" in entities.abbreviations
    assert "ARIA" in entities.abbreviations


def test_extract_equations():
    text = "The loss function is $L = -\\sum y \\log(p)$. We also used $\\alpha = 0.01$."
    entities = extract_entities(text)
    assert len(entities.equations) >= 1


def test_figure_table_references():
    text = "As shown in Figure 1, the results in Table 2 confirm our hypothesis."
    entities = extract_entities(text)
    assert any("Figure 1" in r for r in entities.figure_table_refs)
    assert any("Table 2" in r for r in entities.figure_table_refs)


def test_entities_are_deduplicated():
    text = "We used GLCM [1] and GLCM [1] features from the GLCM analysis."
    entities = extract_entities(text)
    # Citations and abbreviations should not have duplicates
    citation_count = sum(1 for c in entities.citations if c == "[1]")
    assert citation_count == 1


def test_all_entities_returns_flat_list():
    text = "Error was 1.43 +/- 0.30 mm [1]. See Figure 1."
    entities = extract_entities(text)
    all_ents = entities.all_entities()
    assert isinstance(all_ents, list)
    assert len(all_ents) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`oxidizer/preservation/extractor.py`:
```python
"""Extract lockable entities from academic text."""

import re
from dataclasses import dataclass, field


@dataclass
class LockedEntities:
    citations: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    abbreviations: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    figure_table_refs: list[str] = field(default_factory=list)

    def all_entities(self) -> list[str]:
        """Return a flat list of all extracted entities."""
        return (
            self.citations
            + self.numbers
            + self.abbreviations
            + self.equations
            + self.figure_table_refs
        )


# Regex patterns for entity extraction
NUMBERED_CITATION = re.compile(r"\[\d+(?:\s*[-\u2013,]\s*\d+)*\]")
AUTHOR_YEAR_CITATION = re.compile(
    r"(?:[A-Z][a-z]+(?:\s+et\s+al\.?)?\s*(?:,\s*)?(?:\(\d{4}\)|\d{4}))"
)
# Match numbers with optional +/- (± or +/-) and optional units
ERROR_MARGIN = re.compile(
    r"\d+\.?\d*\s*(?:\u00b1|\+/?-)\s*\d+\.?\d*(?:\s*(?:mm|cm|m|ms|s|Hz|dB|%|years?|kg|mL|mg))?"
)
NUMBER_WITH_UNIT = re.compile(
    r"\d+\.?\d*\s*(?:mm|cm|m|ms|s|Hz|dB|%|years?|kg|mL|mg|voxels?|patients?|subjects?)"
)
STANDALONE_NUMBER = re.compile(r"\b\d+\.?\d+\b")
ABBREVIATION_DEF = re.compile(r"[A-Za-z][A-Za-z\s-]+\(([A-Z]{2,}(?:-[A-Z])?)\)")
LATEX_EQUATION = re.compile(r"\$[^$]+\$")
FIGURE_TABLE_REF = re.compile(r"(?:Figure|Fig\.|Table|Supplementary\s+(?:Figure|Table))\s+\d+(?:\.\d+)?")


def extract_entities(text: str) -> LockedEntities:
    """Extract all lockable entities from text."""
    entities = LockedEntities()

    # Citations (numbered)
    for match in NUMBERED_CITATION.finditer(text):
        val = match.group()
        if val not in entities.citations:
            entities.citations.append(val)

    # Citations (author-year) - look for parenthetical patterns
    author_year_paren = re.compile(
        r"\(([A-Z][a-z]+(?:\s+et\s+al\.?)?(?:\s*,\s*)?\d{4})\)"
    )
    for match in author_year_paren.finditer(text):
        val = f"({match.group(1)})"
        if val not in entities.citations:
            entities.citations.append(val)

    # Error margins (must come before standalone numbers)
    for match in ERROR_MARGIN.finditer(text):
        val = match.group().strip()
        if val not in entities.numbers:
            entities.numbers.append(val)

    # Numbers with units
    for match in NUMBER_WITH_UNIT.finditer(text):
        val = match.group().strip()
        # Skip if already captured as part of an error margin
        if not any(val in em for em in entities.numbers):
            if val not in entities.numbers:
                entities.numbers.append(val)

    # Abbreviation definitions
    for match in ABBREVIATION_DEF.finditer(text):
        abbr = match.group(1)
        if abbr not in entities.abbreviations:
            entities.abbreviations.append(abbr)

    # LaTeX equations
    for match in LATEX_EQUATION.finditer(text):
        val = match.group()
        if val not in entities.equations:
            entities.equations.append(val)

    # Figure and table references
    for match in FIGURE_TABLE_REF.finditer(text):
        val = match.group()
        if val not in entities.figure_table_refs:
            entities.figure_table_refs.append(val)

    return entities
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n oxidizer python -m pytest tests/test_extractor.py -v`
Expected: 9 passed (some tests may need minor regex tuning)

- [ ] **Step 5: Commit**

```bash
git add oxidizer/preservation/extractor.py tests/test_extractor.py
git commit -m "feat: entity extractor for citations, numbers, equations, abbreviations"
```

---

### Task 6: Preservation Checker (Tier 1)

**Files:**
- Create: `oxidizer/preservation/checker.py`
- Create: `tests/test_checker.py`

- [ ] **Step 1: Write the failing test**

`tests/test_checker.py`:
```python
from oxidizer.preservation.checker import check_entity_preservation, PreservationResult
from oxidizer.preservation.extractor import LockedEntities


def test_all_entities_preserved():
    entities = LockedEntities(
        citations=["[1]", "[2, 3]"],
        numbers=["1.43 +/- 0.30 mm", "95.2%"],
        abbreviations=["GLCM", "ARIA"],
        equations=[],
        figure_table_refs=["Figure 1"],
    )
    output = (
        "The GLCM features showed accuracy of 95.2% [1]. "
        "Combined results [2, 3] demonstrated error of 1.43 +/- 0.30 mm. "
        "ARIA was detected in Figure 1."
    )
    result = check_entity_preservation(entities, output)
    assert result.passed is True
    assert len(result.missing) == 0
    assert result.preserved_count == 6
    assert result.total_count == 6


def test_missing_citation_detected():
    entities = LockedEntities(citations=["[1]", "[2]"], numbers=[], abbreviations=[])
    output = "The method [1] showed good results."
    result = check_entity_preservation(entities, output)
    assert result.passed is False
    assert "[2]" in result.missing


def test_missing_number_detected():
    entities = LockedEntities(
        citations=[], numbers=["1.43 +/- 0.30 mm"], abbreviations=[]
    )
    output = "The error was approximately 1.4 mm."
    result = check_entity_preservation(entities, output)
    assert result.passed is False
    assert any("1.43" in m for m in result.missing)


def test_empty_entities_always_passes():
    entities = LockedEntities()
    output = "Any text here."
    result = check_entity_preservation(entities, output)
    assert result.passed is True
    assert result.preserved_count == 0
    assert result.total_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_checker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`oxidizer/preservation/checker.py`:
```python
"""Verify entity preservation after restyling (Tier 1: string matching)."""

from dataclasses import dataclass, field

from oxidizer.preservation.extractor import LockedEntities


@dataclass
class PreservationResult:
    passed: bool
    preserved_count: int
    total_count: int
    missing: list[str] = field(default_factory=list)
    details: dict[str, list[str]] = field(default_factory=dict)


def check_entity_preservation(
    entities: LockedEntities,
    output_text: str,
) -> PreservationResult:
    """Check that all locked entities appear in the output text.

    This is Tier 1 verification: simple string matching. It checks that
    every extracted entity from the original text appears verbatim in the
    restyled output.
    """
    all_ents = entities.all_entities()
    total = len(all_ents)

    if total == 0:
        return PreservationResult(passed=True, preserved_count=0, total_count=0)

    missing: list[str] = []
    preserved = 0

    for entity in all_ents:
        if entity in output_text:
            preserved += 1
        else:
            missing.append(entity)

    return PreservationResult(
        passed=len(missing) == 0,
        preserved_count=preserved,
        total_count=total,
        missing=missing,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n oxidizer python -m pytest tests/test_checker.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add oxidizer/preservation/checker.py tests/test_checker.py
git commit -m "feat: Tier 1 entity preservation checker (string matching)"
```

---

### Task 7: Style Metrics Computation

**Files:**
- Create: `oxidizer/scoring/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
from oxidizer.scoring.metrics import (
    compute_sentence_lengths,
    compute_active_voice_ratio,
    count_banned_words,
    count_semicolons_per_100,
    count_parentheticals_per_100,
    count_contractions,
    compute_transition_score,
)


SAMPLE_TEXT = (
    "We collected data from 76 patients. "
    "The imaging was performed on a 3T scanner; "
    "T1-weighted images were acquired with standardized parameters. "
    "While the results showed improvement, additional validation is needed. "
    "However, the method (using PyRadiomics) demonstrated robust performance."
)


def test_sentence_lengths():
    lengths = compute_sentence_lengths(SAMPLE_TEXT)
    assert len(lengths) >= 4
    assert all(isinstance(n, int) for n in lengths)
    assert all(n > 0 for n in lengths)


def test_active_voice_ratio():
    # "We collected" = active, "images were acquired" = passive
    ratio = compute_active_voice_ratio(SAMPLE_TEXT)
    assert 0.0 <= ratio <= 1.0
    # Should detect at least some active voice
    assert ratio > 0.3


def test_banned_words_none_found():
    clean_text = "We analyzed the data and found good results."
    found = count_banned_words(clean_text, ["delve", "tapestry", "multifaceted"])
    assert found == []


def test_banned_words_detected():
    dirty_text = "We delve into the multifaceted landscape of this problem."
    found = count_banned_words(dirty_text, ["delve", "multifaceted", "landscape"])
    assert "delve" in found
    assert "multifaceted" in found
    assert "landscape" in found


def test_banned_words_case_insensitive():
    text = "We DELVE into the topic. The Landscape is complex."
    found = count_banned_words(text, ["delve", "landscape"])
    assert "delve" in found
    assert "landscape" in found


def test_semicolons_per_100():
    # One semicolon in ~5 sentences
    count = count_semicolons_per_100(SAMPLE_TEXT)
    assert count > 0


def test_parentheticals_per_100():
    count = count_parentheticals_per_100(SAMPLE_TEXT)
    assert count > 0  # "(using PyRadiomics)" is one


def test_contractions_zero():
    formal = "We did not find any significant differences."
    assert count_contractions(formal) == 0


def test_contractions_detected():
    informal = "We didn't find any. It's not significant. They're wrong."
    assert count_contractions(informal) >= 2


def test_transition_score():
    preferred = ["while", "however"]
    text = "While this works, however, another approach exists. Additionally, we tried X."
    score = compute_transition_score(text, preferred)
    # "while" and "however" are preferred; "additionally" is not in list
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

`oxidizer/scoring/metrics.py`:
```python
"""Compute style metrics on text."""

import re
import statistics

import nltk
import spacy

# Load models at module level (cached after first load)
_nlp = None

CONTRACTION_PATTERN = re.compile(
    r"\b(?:can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"hasn't|haven't|hadn't|couldn't|shouldn't|wouldn't|it's|that's|"
    r"there's|here's|what's|who's|they're|we're|you're|I'm|he's|she's)\b",
    re.IGNORECASE,
)

# Common transitions to detect (not just the preferred ones)
ALL_TRANSITIONS = [
    "while", "however", "additionally", "as such", "as a result",
    "similarly", "to address", "collectively", "furthermore", "notably",
    "conversely", "given that", "moreover", "nevertheless", "therefore",
    "consequently", "in contrast", "on the other hand", "specifically",
    "in particular", "for example", "for instance",
]


def _get_nlp():
    """Lazy-load spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _sent_tokenize(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(text)


def compute_sentence_lengths(text: str) -> list[int]:
    """Return a list of word counts per sentence."""
    sentences = _sent_tokenize(text)
    return [len(s.split()) for s in sentences if s.strip()]


def compute_active_voice_ratio(text: str) -> float:
    """Estimate active vs passive voice ratio using spaCy dependency parsing.

    Looks for nsubjpass (passive subject) dependencies. Sentences with
    nsubjpass are counted as passive; all others as active.
    """
    nlp = _get_nlp()
    sentences = _sent_tokenize(text)
    if not sentences:
        return 1.0

    passive_count = 0
    for sent in sentences:
        doc = nlp(sent)
        has_passive = any(tok.dep_ == "nsubjpass" for tok in doc)
        if has_passive:
            passive_count += 1

    total = len(sentences)
    return (total - passive_count) / total if total > 0 else 1.0


def count_banned_words(text: str, banned_list: list[str]) -> list[str]:
    """Find which banned words appear in the text. Case-insensitive."""
    text_lower = text.lower()
    found = []
    for word in banned_list:
        # Use word boundary matching for single words, substring for phrases
        if " " in word:
            if word.lower() in text_lower:
                found.append(word)
        else:
            pattern = re.compile(r"\b" + re.escape(word.lower()) + r"\b")
            if pattern.search(text_lower):
                found.append(word)
    return found


def count_semicolons_per_100(text: str) -> float:
    """Count semicolons per 100 sentences."""
    sentences = _sent_tokenize(text)
    n_sentences = len(sentences)
    if n_sentences == 0:
        return 0.0
    n_semicolons = text.count(";")
    return (n_semicolons / n_sentences) * 100


def count_parentheticals_per_100(text: str) -> float:
    """Count parenthetical pairs per 100 sentences."""
    sentences = _sent_tokenize(text)
    n_sentences = len(sentences)
    if n_sentences == 0:
        return 0.0
    # Count opening parentheses as proxy for pairs
    n_parens = text.count("(")
    return (n_parens / n_sentences) * 100


def count_contractions(text: str) -> int:
    """Count contractions in text."""
    return len(CONTRACTION_PATTERN.findall(text))


def compute_transition_score(text: str, preferred: list[str]) -> float:
    """Compute fraction of transitions that come from the preferred list.

    Returns a score between 0 and 1. Higher means more transitions are
    from the preferred list.
    """
    text_lower = text.lower()
    preferred_lower = [t.lower() for t in preferred]

    preferred_found = 0
    total_found = 0

    for t in ALL_TRANSITIONS:
        # Check if this transition appears in the text
        if t in text_lower:
            total_found += 1
            if t in preferred_lower:
                preferred_found += 1

    if total_found == 0:
        return 1.0  # No transitions found; not penalized

    return preferred_found / total_found
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n oxidizer python -m pytest tests/test_metrics.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add oxidizer/scoring/metrics.py tests/test_metrics.py
git commit -m "feat: style metrics computation (sentence length, voice, banned words, punctuation)"
```

---

### Task 8: Style Reporter

**Files:**
- Create: `oxidizer/scoring/reporter.py`
- Create: `tests/test_reporter.py`

- [ ] **Step 1: Write the failing test**

`tests/test_reporter.py`:
```python
import json

from oxidizer.scoring.reporter import compute_style_report, StyleReport
from oxidizer.profiles.schema import (
    StyleProfile,
    SentenceLengthMetrics,
    ParagraphMetrics,
    VoiceMetrics,
    PunctuationMetrics,
    TransitionConfig,
    VocabularyConfig,
    VoiceRules,
)


def _make_profile(**overrides) -> StyleProfile:
    defaults = dict(
        name="Test",
        version=1,
        source_documents=[],
        sentence_length=SentenceLengthMetrics(mean=24.1, median=23, std=10.1, range_min=6, range_max=77),
        paragraph=ParagraphMetrics(mean_words=71, sentences_per_paragraph=(3, 4)),
        voice=VoiceMetrics(active_ratio=0.90, passive_contexts=[]),
        contractions=False,
        type_token_ratio=0.346,
        transitions=TransitionConfig(preferred=["while", "however"], acceptable=[]),
        vocabulary=VocabularyConfig(preferred=[], banned_aiisms=["delve", "tapestry"]),
        punctuation=PunctuationMetrics(semicolons_per_100=12, parentheticals_per_100=25, em_dashes=0, inline_enumerations=True),
        voice_rules=VoiceRules(person="we", hedging=[], reasoning=True, problem_before_solution=True, quantitative_precision=True),
        style_prompt="",
        few_shot_examples=[],
    )
    defaults.update(overrides)
    return StyleProfile(**defaults)


def test_style_report_has_all_fields():
    profile = _make_profile()
    text = (
        "We collected data from 76 patients; the imaging protocol was standardized. "
        "While the results showed promise, further validation is required. "
        "The method demonstrated consistent performance across all subjects. "
        "However, additional samples would strengthen the conclusions."
    )
    report = compute_style_report(text, profile)
    assert isinstance(report, StyleReport)
    assert 0.0 <= report.style_match_score <= 1.0
    assert report.sentence_length_mean > 0
    assert report.sentence_length_std >= 0
    assert 0.0 <= report.active_voice_ratio <= 1.0
    assert isinstance(report.banned_words_found, list)
    assert report.contraction_count >= 0


def test_clean_text_scores_high():
    profile = _make_profile()
    # Text designed to match the profile closely
    text = (
        "We analyzed the imaging data from 50 subjects using a standardized protocol; "
        "the preprocessing pipeline included motion correction and spatial normalization. "
        "While the initial results were encouraging, we identified several limitations. "
        "However, the overall accuracy of 0.92 suggests the approach is viable. "
        "We collected additional samples to validate our findings across sites."
    )
    report = compute_style_report(text, profile)
    assert report.style_match_score > 0.5
    assert report.banned_words_found == []


def test_dirty_text_scores_lower():
    profile = _make_profile()
    text = "We delve into the multifaceted tapestry of this groundbreaking research."
    report = compute_style_report(text, profile)
    assert len(report.banned_words_found) > 0
    # Banned words should drag down the score
    clean_report = compute_style_report("We analyzed the data carefully.", profile)
    assert report.style_match_score < clean_report.style_match_score


def test_report_to_dict():
    profile = _make_profile()
    text = "We analyzed the data and found positive results."
    report = compute_style_report(text, profile)
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "style_match_score" in d
    # Should be JSON-serializable
    json.dumps(d)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_reporter.py -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

`oxidizer/scoring/reporter.py`:
```python
"""Generate style-match reports."""

import statistics
from dataclasses import dataclass, field, asdict

from oxidizer.profiles.schema import StyleProfile
from oxidizer.scoring.metrics import (
    compute_sentence_lengths,
    compute_active_voice_ratio,
    count_banned_words,
    count_semicolons_per_100,
    count_parentheticals_per_100,
    count_contractions,
    compute_transition_score,
)


@dataclass
class StyleReport:
    style_match_score: float
    sentence_length_mean: float
    sentence_length_std: float
    sentence_length_target_mean: float
    sentence_length_target_std: float
    active_voice_ratio: float
    active_voice_target: float
    banned_words_found: list[str] = field(default_factory=list)
    semicolons_per_100: float = 0.0
    semicolons_target: float = 0.0
    parentheticals_per_100: float = 0.0
    parentheticals_target: float = 0.0
    contraction_count: int = 0
    transition_score: float = 0.0
    sub_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_style_report(text: str, profile: StyleProfile) -> StyleReport:
    """Compute a full style-match report for text against a profile."""

    # Sentence length
    lengths = compute_sentence_lengths(text)
    if lengths:
        sl_mean = statistics.mean(lengths)
        sl_std = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    else:
        sl_mean = 0.0
        sl_std = 0.0

    target_mean = profile.sentence_length.mean
    target_std = profile.sentence_length.std

    sl_mean_score = _clamp(1 - abs(sl_mean - target_mean) / target_mean) if target_mean > 0 else 1.0
    sl_std_score = _clamp(1 - abs(sl_std - target_std) / target_std) if target_std > 0 else 1.0

    # Active voice
    av_ratio = compute_active_voice_ratio(text)
    av_target = profile.voice.active_ratio
    av_score = _clamp(1 - abs(av_ratio - av_target))

    # Banned words
    banned = count_banned_words(text, profile.vocabulary.banned_aiisms)
    banned_score = 1.0 if not banned else _clamp(1 - 0.1 * len(banned))

    # Semicolons
    sc_actual = count_semicolons_per_100(text)
    sc_target = profile.punctuation.semicolons_per_100
    sc_score = _clamp(1 - abs(sc_actual - sc_target) / sc_target) if sc_target > 0 else 1.0

    # Parentheticals
    pa_actual = count_parentheticals_per_100(text)
    pa_target = profile.punctuation.parentheticals_per_100
    pa_score = _clamp(1 - abs(pa_actual - pa_target) / pa_target) if pa_target > 0 else 1.0

    # Transitions
    tr_score = compute_transition_score(text, profile.transitions.preferred)

    # Contractions
    contraction_count = count_contractions(text)
    contr_score = 1.0 if contraction_count == 0 else 0.0

    # Weighted composite score
    sub_scores = {
        "sentence_length_mean": sl_mean_score,
        "sentence_length_variance": sl_std_score,
        "active_voice": av_score,
        "banned_words": banned_score,
        "semicolons": sc_score,
        "parentheticals": pa_score,
        "transitions": tr_score,
        "contractions": contr_score,
    }

    weights = {
        "sentence_length_mean": 0.20,
        "sentence_length_variance": 0.10,
        "active_voice": 0.15,
        "banned_words": 0.20,
        "semicolons": 0.10,
        "parentheticals": 0.10,
        "transitions": 0.10,
        "contractions": 0.05,
    }

    composite = sum(sub_scores[k] * weights[k] for k in sub_scores)

    return StyleReport(
        style_match_score=round(composite, 4),
        sentence_length_mean=round(sl_mean, 1),
        sentence_length_std=round(sl_std, 1),
        sentence_length_target_mean=target_mean,
        sentence_length_target_std=target_std,
        active_voice_ratio=round(av_ratio, 2),
        active_voice_target=av_target,
        banned_words_found=banned,
        semicolons_per_100=round(sc_actual, 1),
        semicolons_target=sc_target,
        parentheticals_per_100=round(pa_actual, 1),
        parentheticals_target=pa_target,
        contraction_count=contraction_count,
        transition_score=round(tr_score, 2),
        sub_scores={k: round(v, 4) for k, v in sub_scores.items()},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n oxidizer python -m pytest tests/test_reporter.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add oxidizer/scoring/reporter.py tests/test_reporter.py
git commit -m "feat: style-match reporter with weighted composite scoring"
```

---

### Task 9: Revise Engine

**Files:**
- Create: `oxidizer/engine/revise.py`
- Create: `tests/test_revise.py`

- [ ] **Step 1: Write the failing test**

`tests/test_revise.py`:
```python
from unittest.mock import patch, MagicMock
from pathlib import Path

from oxidizer.engine.revise import (
    build_restyle_prompt,
    ReviseResult,
    revise_section,
)
from oxidizer.parsers.markdown_parser import Section
from oxidizer.profiles.schema import (
    StyleProfile,
    SentenceLengthMetrics,
    ParagraphMetrics,
    VoiceMetrics,
    PunctuationMetrics,
    TransitionConfig,
    VocabularyConfig,
    VoiceRules,
    FewShotExample,
)


def _make_profile() -> StyleProfile:
    return StyleProfile(
        name="Test",
        version=1,
        source_documents=[],
        sentence_length=SentenceLengthMetrics(mean=24.1, median=23, std=10.1, range_min=6, range_max=77),
        paragraph=ParagraphMetrics(mean_words=71, sentences_per_paragraph=(3, 4)),
        voice=VoiceMetrics(active_ratio=0.90, passive_contexts=[]),
        contractions=False,
        type_token_ratio=0.346,
        transitions=TransitionConfig(preferred=["while", "however"], acceptable=[]),
        vocabulary=VocabularyConfig(preferred=["robust"], banned_aiisms=["delve", "tapestry"]),
        punctuation=PunctuationMetrics(semicolons_per_100=12, parentheticals_per_100=25, em_dashes=0, inline_enumerations=True),
        voice_rules=VoiceRules(person="we", hedging=["would likely"], reasoning=True, problem_before_solution=True, quantitative_precision=True),
        style_prompt="Write in Jinchi Wei's voice. Use 'we'. No em dashes.",
        few_shot_examples=[
            FewShotExample(category="results", text="Registration error was 1.43 +/- 0.30 mm.")
        ],
    )


def test_build_restyle_prompt_includes_style_instructions():
    profile = _make_profile()
    section = Section(heading="Methods", body="Data was collected from subjects.", context="methods", level=2)
    prompt = build_restyle_prompt(section, profile, locked_entities=["[1]", "50 subjects"])
    assert "Jinchi Wei" in prompt or "we" in prompt.lower()
    assert "[1]" in prompt
    assert "50 subjects" in prompt
    assert "methods" in prompt.lower()


def test_build_restyle_prompt_includes_locked_entities():
    profile = _make_profile()
    section = Section(heading="Results", body="Accuracy was 95.2% [1].", context="results", level=2)
    prompt = build_restyle_prompt(section, profile, locked_entities=["95.2%", "[1]"])
    assert "<lock>" in prompt or "LOCKED" in prompt.upper() or "verbatim" in prompt.lower()


def test_revise_section_calls_api(monkeypatch):
    """revise_section should call the Anthropic API and return a ReviseResult."""
    profile = _make_profile()
    section = Section(heading="Methods", body="The data was analyzed.", context="methods", level=2)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="We analyzed the data from 50 subjects.")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    result = revise_section(section, profile, client=mock_client)
    assert isinstance(result, ReviseResult)
    assert result.restyled_text == "We analyzed the data from 50 subjects."
    mock_client.messages.create.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_revise.py -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

`oxidizer/engine/revise.py`:
```python
"""Revise engine: restyle text sections to match a style profile."""

from dataclasses import dataclass, field

from oxidizer.parsers.markdown_parser import Section
from oxidizer.preservation.extractor import extract_entities, LockedEntities
from oxidizer.preservation.checker import check_entity_preservation, PreservationResult
from oxidizer.profiles.schema import StyleProfile
from oxidizer.scoring.reporter import compute_style_report, StyleReport


@dataclass
class ReviseResult:
    restyled_text: str
    original_text: str
    section_heading: str
    entities: LockedEntities
    preservation: PreservationResult | None = None
    style_report: StyleReport | None = None
    retries: int = 0
    warnings: list[str] = field(default_factory=list)


def build_restyle_prompt(
    section: Section,
    profile: StyleProfile,
    locked_entities: list[str] | None = None,
) -> str:
    """Build the prompt for Claude to restyle a section."""
    locked = locked_entities or []

    locked_section = ""
    if locked:
        entity_list = "\n".join(f"  - {e}" for e in locked)
        locked_section = (
            f"\n\nLOCKED ENTITIES (must appear verbatim in your output):\n{entity_list}\n"
            "These are citations, numbers, equations, and abbreviations that must be "
            "preserved exactly as written. Do not paraphrase, round, or omit any of them."
        )

    few_shots = ""
    if profile.few_shot_examples:
        examples = "\n".join(
            f"  [{ex.category}]: \"{ex.text}\"" for ex in profile.few_shot_examples
        )
        few_shots = f"\n\nFew-shot examples of the target style:\n{examples}"

    return f"""Restyle the following academic text to match the writing voice described below.
Do NOT change the meaning, claims, or factual content. Only change the wording, sentence structure, and style.

STYLE PROFILE:
{profile.style_prompt}

SECTION CONTEXT: {section.context}
(Adjust style accordingly: e.g., methods sections may use more passive voice for experimental descriptions.)
{locked_section}
{few_shots}

RULES:
- Use first person plural ("we") as the subject.
- Average sentence length should be around {profile.sentence_length.mean} words with high variance.
- Use semicolons to join related clauses. Use parenthetical asides for specifications.
- NEVER use em dashes.
- NEVER use these words: {", ".join(profile.vocabulary.banned_aiisms[:10])}...
- Do not add contractions.
- Preserve all technical terminology, abbreviations, and quantitative values exactly.

TEXT TO RESTYLE:
{section.body}

OUTPUT: Return only the restyled text. No explanations, no preamble."""


def revise_section(
    section: Section,
    profile: StyleProfile,
    client=None,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 2,
) -> ReviseResult:
    """Restyle a single section using the Claude API.

    Args:
        section: The parsed section to restyle.
        profile: The target style profile.
        client: An Anthropic client instance. If None, creates one.
        model: The Claude model to use.
        max_retries: Max retries if entity preservation fails.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    # Step 1: Extract entities to lock
    entities = extract_entities(section.body)
    locked = entities.all_entities()

    # Step 2: Build prompt and call API
    prompt = build_restyle_prompt(section, profile, locked_entities=locked)

    restyled_text = ""
    retries = 0
    preservation = None
    warnings = []

    for attempt in range(1 + max_retries):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        restyled_text = response.content[0].text.strip()

        # Step 3: Verify entity preservation (Tier 1)
        preservation = check_entity_preservation(entities, restyled_text)

        if preservation.passed:
            break

        retries = attempt + 1
        if retries <= max_retries:
            # Add stricter instructions for retry
            missing_str = ", ".join(preservation.missing)
            prompt += (
                f"\n\nPREVIOUS ATTEMPT FAILED: The following entities were missing "
                f"from your output: {missing_str}\n"
                "You MUST include ALL locked entities verbatim. Try again."
            )
        else:
            warnings.append(
                f"Entity preservation failed after {max_retries} retries. "
                f"Missing: {preservation.missing}"
            )

    # Step 4: Compute style report
    style_report = compute_style_report(restyled_text, profile)

    # Check for banned words and warn
    if style_report.banned_words_found:
        warnings.append(
            f"Banned AI-isms found in output: {style_report.banned_words_found}"
        )

    return ReviseResult(
        restyled_text=restyled_text,
        original_text=section.body,
        section_heading=section.heading,
        entities=entities,
        preservation=preservation,
        style_report=style_report,
        retries=retries,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n oxidizer python -m pytest tests/test_revise.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add oxidizer/engine/revise.py tests/test_revise.py
git commit -m "feat: revise engine with entity locking, verification, and style scoring"
```

---

### Task 10: CLI (revise + score commands)

**Files:**
- Create: `oxidizer/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from click.testing import CliRunner
from pathlib import Path

from oxidizer.cli import cli


FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_score_command():
    """The 'score' command should score text against a profile."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        "score",
        str(FIXTURES / "sample_methods.md"),
        "--profile", "jinchi",
    ])
    assert result.exit_code == 0
    assert "style_match_score" in result.output.lower() or "Style Match" in result.output


def test_cli_score_missing_file():
    """Score command with missing file should error gracefully."""
    runner = CliRunner()
    result = runner.invoke(cli, ["score", "nonexistent.md", "--profile", "jinchi"])
    assert result.exit_code != 0


def test_cli_revise_missing_profile():
    """Revise with missing profile should error gracefully."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        "revise",
        str(FIXTURES / "sample_methods.md"),
        "--profile", "nonexistent_profile",
    ])
    assert result.exit_code != 0


def test_cli_help():
    """CLI should show help text."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "oxidizer" in result.output.lower() or "Usage" in result.output


def test_cli_score_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["score", "--help"])
    assert result.exit_code == 0
    assert "profile" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n oxidizer python -m pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

`oxidizer/cli.py`:
```python
"""Oxidizer CLI."""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from oxidizer.profiles.loader import load_profile
from oxidizer.parsers.markdown_parser import parse_markdown
from oxidizer.parsers.docx_parser import parse_docx
from oxidizer.scoring.reporter import compute_style_report

console = Console()


def _parse_document(path: Path):
    """Parse a document based on its file extension."""
    suffix = path.suffix.lower()
    if suffix == ".md":
        return parse_markdown(path.read_text())
    elif suffix == ".docx":
        return parse_docx(path)
    elif suffix == ".pdf":
        from oxidizer.parsers.pdf_parser import parse_pdf
        return parse_pdf(path)
    else:
        # Try as markdown/plain text
        return parse_markdown(path.read_text())


def _display_report(report, section_name: str = ""):
    """Display a style report using Rich tables."""
    title = f"Style Report: {section_name}" if section_name else "Style Report"
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Actual", style="white")
    table.add_column("Target", style="green")
    table.add_column("Score", style="bold")

    table.add_row(
        "Style Match Score",
        f"{report.style_match_score:.2f}",
        ">0.85",
        _score_color(report.style_match_score),
    )
    table.add_row(
        "Sentence Length (mean)",
        f"{report.sentence_length_mean:.1f}",
        f"{report.sentence_length_target_mean:.1f}",
        _score_color(report.sub_scores.get("sentence_length_mean", 0)),
    )
    table.add_row(
        "Sentence Length (std)",
        f"{report.sentence_length_std:.1f}",
        f"{report.sentence_length_target_std:.1f}",
        _score_color(report.sub_scores.get("sentence_length_variance", 0)),
    )
    table.add_row(
        "Active Voice Ratio",
        f"{report.active_voice_ratio:.0%}",
        f"{report.active_voice_target:.0%}",
        _score_color(report.sub_scores.get("active_voice", 0)),
    )
    table.add_row(
        "Banned Words",
        str(report.banned_words_found) if report.banned_words_found else "None",
        "None",
        _score_color(report.sub_scores.get("banned_words", 0)),
    )
    table.add_row(
        "Semicolons/100 sent",
        f"{report.semicolons_per_100:.1f}",
        f"{report.semicolons_target:.1f}",
        _score_color(report.sub_scores.get("semicolons", 0)),
    )
    table.add_row(
        "Parentheticals/100 sent",
        f"{report.parentheticals_per_100:.1f}",
        f"{report.parentheticals_target:.1f}",
        _score_color(report.sub_scores.get("parentheticals", 0)),
    )
    table.add_row(
        "Contractions",
        str(report.contraction_count),
        "0",
        _score_color(report.sub_scores.get("contractions", 0)),
    )
    table.add_row(
        "Transition Score",
        f"{report.transition_score:.0%}",
        "preferred list",
        _score_color(report.sub_scores.get("transitions", 0)),
    )

    console.print(table)


def _score_color(score: float) -> str:
    if score >= 0.85:
        return f"[green]{score:.2f}[/green]"
    elif score >= 0.6:
        return f"[yellow]{score:.2f}[/yellow]"
    else:
        return f"[red]{score:.2f}[/red]"


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Oxidizer: Academic writing style engine."""
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--profile", "-p", required=True, help="Style profile name")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file path")
@click.option("--sections", "-s", help="Comma-separated section names to revise")
@click.option("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
def revise(file: Path, profile: str, output: Path | None, sections: str | None, model: str):
    """Revise a document to match a style profile."""
    try:
        style_profile = load_profile(profile)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    parsed_sections = _parse_document(file)

    if sections:
        section_names = [s.strip().lower() for s in sections.split(",")]
        parsed_sections = [
            s for s in parsed_sections
            if s.heading.lower() in section_names or s.context in section_names
        ]

    if not parsed_sections:
        console.print("[red]No sections found to revise.[/red]")
        sys.exit(1)

    from oxidizer.engine.revise import revise_section

    results = []
    for section in parsed_sections:
        console.print(f"\n[cyan]Revising:[/cyan] {section.heading or '(untitled)'}")
        result = revise_section(section, style_profile, model=model)
        results.append(result)

        if result.warnings:
            for w in result.warnings:
                console.print(f"  [yellow]Warning:[/yellow] {w}")

        if result.preservation and not result.preservation.passed:
            console.print(f"  [red]Entities missing:[/red] {result.preservation.missing}")

        if result.style_report:
            _display_report(result.style_report, section.heading)

    # Write output
    output_parts = []
    for r in results:
        if r.section_heading:
            output_parts.append(f"## {r.section_heading}\n\n{r.restyled_text}")
        else:
            output_parts.append(r.restyled_text)

    output_text = "\n\n".join(output_parts)

    if output:
        output.write_text(output_text)
        console.print(f"\n[green]Written to {output}[/green]")

        # Also write report.json alongside the output
        report_path = output.with_suffix(".report.json")
        report_data = {
            "sections": [
                {
                    "heading": r.section_heading,
                    "style_report": r.style_report.to_dict() if r.style_report else None,
                    "preservation": {
                        "passed": r.preservation.passed if r.preservation else None,
                        "preserved": r.preservation.preserved_count if r.preservation else 0,
                        "total": r.preservation.total_count if r.preservation else 0,
                        "missing": r.preservation.missing if r.preservation else [],
                    },
                    "retries": r.retries,
                    "warnings": r.warnings,
                }
                for r in results
            ],
        }
        report_path.write_text(json.dumps(report_data, indent=2))
        console.print(f"[green]Report written to {report_path}[/green]")
    else:
        console.print("\n--- RESTYLED OUTPUT ---\n")
        console.print(output_text)


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--profile", "-p", required=True, help="Style profile name")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def score(file: Path, profile: str, json_output: bool):
    """Score existing text against a style profile (no rewriting)."""
    try:
        style_profile = load_profile(profile)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    parsed_sections = _parse_document(file)

    for section in parsed_sections:
        report = compute_style_report(section.body, style_profile)

        if json_output:
            click.echo(json.dumps(report.to_dict(), indent=2))
        else:
            _display_report(report, section.heading or file.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n oxidizer python -m pytest tests/test_cli.py -v`
Expected: 5 passed

- [ ] **Step 5: Verify CLI works end-to-end**

Run: `conda run -n oxidizer python -m oxidizer.cli score tests/fixtures/sample_methods.md --profile jinchi`
Expected: A style report table showing metrics

- [ ] **Step 6: Commit**

```bash
git add oxidizer/cli.py tests/test_cli.py
git commit -m "feat: CLI with revise and score commands"
```

---

### Task 11: Claude Code Skill Definition

**Files:**
- Create: `skill/SKILL.md`

- [ ] **Step 1: Create the skill file**

`skill/SKILL.md`:
```markdown
---
name: oxidize
description: Revise or score academic text against a personal writing style profile
---

# /oxidize

Academic writing style engine. Two modes:

## Commands

### /oxidize revise <file> [--sections X,Y]
Restyle a document to match your writing voice. Preserves all citations, numbers, equations, and abbreviations.

### /oxidize score <file>
Score text against your style profile without changing anything. Shows how close it matches your voice.

## How It Works

1. Reads the document and splits into sections
2. Extracts entities to lock (citations, numbers, equations, abbreviations)
3. Sends each section to Claude with your style profile for restyling
4. Verifies all locked entities are preserved in the output
5. Computes style-match metrics (sentence length, voice ratio, banned words, punctuation)
6. Outputs restyled text + a report

## Usage from Claude Code

When the user says "/oxidize revise" or "/oxidize score", run the corresponding CLI command:

```bash
conda run -n oxidizer oxidizer revise <file> --profile jinchi --output <output_path>
```

or

```bash
conda run -n oxidizer oxidizer score <file> --profile jinchi
```

The profile "jinchi" is the default. For other profiles, specify with --profile <name>.

## Profile Location

Profiles are YAML files in `profiles/` directory of the oxidizer repo at /Users/jinchiwei/arcadia/oxidizer/profiles/.
```

- [ ] **Step 2: Commit**

```bash
git add skill/
git commit -m "feat: Claude Code skill definition for /oxidize"
```

---

### Task 12: Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write an integration test**

`tests/test_integration.py`:
```python
"""Integration tests for the full revise pipeline.

These tests mock the Claude API but exercise the full pipeline:
parse -> extract -> restyle -> verify -> score -> report
"""

from unittest.mock import MagicMock
from pathlib import Path

from oxidizer.parsers.markdown_parser import parse_markdown
from oxidizer.engine.revise import revise_section
from oxidizer.profiles.loader import load_profile
from oxidizer.scoring.reporter import compute_style_report


PROFILES_DIR = Path(__file__).parent.parent / "profiles"

SAMPLE_INPUT = """We collected data from 76 patients (42 female, 34 male; mean age 62.3 +/- 8.1 years) who underwent pretreatment MRI at UCSF between 2019 and 2023 [1]. All patients provided informed consent as approved by the institutional review board (IRB #19-28456).

Radiomics features were extracted using PyRadiomics (v3.0.1) from manually segmented regions of interest (ROIs) [2, 3]. A total of 107 features were computed across five categories."""

# Simulated Claude output that preserves entities and matches style
MOCK_RESTYLED = """We collected data from 76 patients (42 female, 34 male; mean age 62.3 +/- 8.1 years) who underwent pretreatment MRI at UCSF between 2019 and 2023 [1]. All patients provided informed consent as approved by the institutional review board (IRB #19-28456).

We extracted radiomics features using PyRadiomics (v3.0.1) from manually segmented regions of interest (ROIs) [2, 3]; a total of 107 features were computed across five categories."""


def test_full_pipeline_with_mock_api():
    """Exercise the complete pipeline with a mocked Claude response."""
    profile = load_profile("jinchi", search_paths=[PROFILES_DIR])

    sections = parse_markdown(f"## Methods\n\n{SAMPLE_INPUT}")
    assert len(sections) == 1

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=MOCK_RESTYLED)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    result = revise_section(sections[0], profile, client=mock_client)

    # Check entity preservation
    assert result.preservation is not None
    assert result.preservation.passed is True, f"Missing: {result.preservation.missing}"

    # Check style report
    assert result.style_report is not None
    assert result.style_report.style_match_score > 0

    # Check no banned words
    assert result.style_report.banned_words_found == []

    # Check no warnings
    assert result.warnings == []


def test_score_only_no_api():
    """Score command works without any API calls."""
    profile = load_profile("jinchi", search_paths=[PROFILES_DIR])
    report = compute_style_report(SAMPLE_INPUT, profile)

    assert report.style_match_score > 0
    assert report.sentence_length_mean > 0
    assert report.banned_words_found == []
    assert report.contraction_count == 0
```

- [ ] **Step 2: Run integration tests**

Run: `conda run -n oxidizer python -m pytest tests/test_integration.py -v`
Expected: 2 passed

- [ ] **Step 3: Run full test suite**

Run: `conda run -n oxidizer python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration tests for full revise pipeline"
```

---

### Task 13: Final Wiring and README Update

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Update README.md**

Replace the existing README.md with updated content reflecting the actual built tool:

```markdown
# Oxidizer

Academic writing style engine. Analyzes your writing voice and applies it to new or existing documents.

## What it does

- **Revise mode**: Restyle an existing document to match your personal writing voice while preserving all citations, numbers, equations, and technical content
- **Score mode**: Evaluate how closely a text matches your style profile without changing anything

## Setup

```bash
conda activate oxidizer
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

## Usage

### Score text against your profile

```bash
oxidizer score paper.md --profile jinchi
```

### Revise a document

```bash
oxidizer revise paper.docx --profile jinchi --output restyled.md
```

### Revise specific sections

```bash
oxidizer revise paper.md --profile jinchi --sections "methods,results" --output restyled.md
```

## Style Profiles

Profiles are YAML files in `profiles/`. Each profile captures:

- Sentence length distribution targets
- Active/passive voice ratio
- Banned AI-ism words
- Preferred transition words
- Punctuation patterns (semicolons, parentheticals, em dashes)
- Voice rules (person, hedging, reasoning patterns)
- Few-shot examples from real writing

## How Revise Works

1. **Parse**: Split document into sections by headings
2. **Lock**: Extract citations, numbers, equations, abbreviations
3. **Restyle**: Send each section to Claude with style profile and locked entities
4. **Verify**: Check all locked entities appear in output (retries if missing)
5. **Score**: Compute style-match metrics
6. **Report**: Output restyled text + JSON report with metrics

## Project Structure

```
oxidizer/
├── oxidizer/          # Python package
│   ├── cli.py         # CLI entry point
│   ├── engine/        # Revise engine
│   ├── parsers/       # Document parsers (MD, DOCX, PDF)
│   ├── preservation/  # Entity extraction and verification
│   ├── profiles/      # Profile schema and loader
│   └── scoring/       # Style metrics and reporting
├── profiles/          # Style profiles (YAML)
├── skill/             # Claude Code skill definition
└── tests/             # Test suite
```
```

- [ ] **Step 2: Update .gitignore**

Add to `.gitignore`:
```
# Test outputs
*.report.json
restyled*.md

# spaCy models
en_core_web_*
```

- [ ] **Step 3: Run full test suite one more time**

Run: `conda run -n oxidizer python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add README.md .gitignore
git commit -m "docs: update README with usage instructions and project structure"
```

---

## CEO Review Expansions (Tasks 14-20)

### Task 14: LaTeX Parser

**Files:**
- Create: `oxidizer/parsers/latex_parser.py`
- Create: `tests/test_latex_parser.py`

- [ ] **Step 1: Write failing test**

`tests/test_latex_parser.py`:
```python
from oxidizer.parsers.latex_parser import parse_latex

def test_parse_latex_sections():
    text = r"""
\section{Introduction}
This is the introduction text.

\section{Methods}
We collected data from 50 subjects.

\subsection{Data Collection}
The data was collected at UCSF.

\section{Results}
Accuracy was 0.95.
"""
    sections = parse_latex(text)
    assert len(sections) >= 3
    assert sections[0].heading == "Introduction"
    assert sections[1].heading == "Methods"
    assert "0.95" in sections[-1].body

def test_parse_latex_context_mapping():
    text = r"\section{Materials and Methods}\nWe used a 3T scanner."
    sections = parse_latex(text)
    assert sections[0].context == "methods"

def test_parse_latex_no_sections():
    text = "Just plain text with no LaTeX sections."
    sections = parse_latex(text)
    assert len(sections) == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `conda run -n oxidizer python -m pytest tests/test_latex_parser.py -v`

- [ ] **Step 3: Implement**

`oxidizer/parsers/latex_parser.py`:
```python
"""Parse LaTeX documents into sections."""

import re
from oxidizer.parsers.markdown_parser import Section, _classify_context

SECTION_PATTERN = re.compile(
    r"\\(section|subsection|subsubsection)\{([^}]+)\}", re.MULTILINE
)

def parse_latex(text: str) -> list[Section]:
    """Parse LaTeX text into sections based on \\section{} commands."""
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [Section(heading="", body=text.strip(), context="other", level=0)]

    level_map = {"section": 1, "subsection": 2, "subsubsection": 3}
    sections = []
    for i, match in enumerate(matches):
        level = level_map.get(match.group(1), 1)
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        context = _classify_context(heading)
        sections.append(Section(heading=heading, body=body, context=context, level=level))
    return sections
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Update `cli.py` `_parse_document()` to handle `.tex` files**
- [ ] **Step 6: Commit**

```bash
git add oxidizer/parsers/latex_parser.py tests/test_latex_parser.py oxidizer/cli.py
git commit -m "feat: LaTeX document parser"
```

---

### Task 15: LLM Client Wrapper (Optional API Mode)

**Files:**
- Create: `oxidizer/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing test**

`tests/test_llm.py`:
```python
import os
from unittest.mock import patch
from oxidizer.llm import get_client, is_api_available

def test_api_not_available_without_key():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        assert is_api_available() is False

def test_api_available_with_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        assert is_api_available() is True

def test_get_client_returns_none_without_key():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        assert get_client() is None
```

- [ ] **Step 2: Implement**

`oxidizer/llm.py`:
```python
"""Optional Claude API client. Only used when ANTHROPIC_API_KEY is set."""

import os

def is_api_available() -> bool:
    """Check if the Anthropic API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))

def get_client():
    """Get an Anthropic client, or None if API key is not set."""
    if not is_api_available():
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        return None

def call_claude(prompt: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 4096, client=None):
    """Call Claude API. Returns response text or raises if no client."""
    if client is None:
        client = get_client()
    if client is None:
        raise RuntimeError(
            "Claude API not available. Either use Claude Code (/oxidize revise) "
            "or set ANTHROPIC_API_KEY for CLI batch mode."
        )
    response = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
```

- [ ] **Step 3: Update `engine/revise.py` to use `llm.py`**
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add oxidizer/llm.py tests/test_llm.py oxidizer/engine/revise.py
git commit -m "feat: optional LLM client wrapper (API mode)"
```

---

### Task 16: Write Mode Engine

**Files:**
- Create: `oxidizer/engine/write.py`
- Create: `tests/test_write.py`

- [ ] **Step 1: Write failing test**

`tests/test_write.py`:
```python
from unittest.mock import MagicMock
from oxidizer.engine.write import build_write_prompt, write_section, WriteResult
from oxidizer.profiles.loader import load_profile
from pathlib import Path

PROFILES = Path(__file__).parent.parent / "profiles"

def test_build_write_prompt_includes_profile():
    profile = load_profile("jinchi", search_paths=[PROFILES])
    prompt = build_write_prompt(
        topic="Methods section about MRI data collection at UCSF",
        section_type="methods",
        profile=profile,
    )
    assert "we" in prompt.lower()
    assert "methods" in prompt.lower()
    assert "banned" in prompt.lower() or "never use" in prompt.lower()

def test_write_section_returns_result():
    profile = load_profile("jinchi", search_paths=[PROFILES])
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="We collected data from 76 patients.")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    result = write_section(
        topic="MRI data collection methods",
        section_type="methods",
        profile=profile,
        client=mock_client,
    )
    assert isinstance(result, WriteResult)
    assert "76 patients" in result.text
    assert result.style_report is not None
```

- [ ] **Step 2: Implement**

`oxidizer/engine/write.py`:
```python
"""Write engine: generate new section drafts in the author's voice."""

from dataclasses import dataclass
from oxidizer.profiles.schema import StyleProfile
from oxidizer.scoring.reporter import compute_style_report, StyleReport

@dataclass
class WriteResult:
    text: str
    topic: str
    section_type: str
    style_report: StyleReport | None = None

def build_write_prompt(topic: str, section_type: str, profile: StyleProfile) -> str:
    """Build prompt for Claude to write a new section."""
    return f"""Write an academic paper section in the following voice and style.

STYLE PROFILE:
{profile.style_prompt}

SECTION TYPE: {section_type}
TOPIC: {topic}

RULES:
- Use first person plural ("we").
- Target ~{profile.sentence_length.mean} words per sentence with high variance.
- Use semicolons to join related clauses. Use parenthetical asides.
- NEVER use em dashes.
- NEVER use: {", ".join(profile.vocabulary.banned_aiisms[:10])}...
- No contractions.
- Include specific quantitative details where relevant.
- Do NOT hallucinate citations, statistics, or claims. Use placeholders like [CITE] or [X patients] for facts you don't know.

OUTPUT: Return only the section text. No explanations."""

def write_section(topic: str, section_type: str, profile: StyleProfile, client=None, model: str = "claude-sonnet-4-20250514") -> WriteResult:
    """Generate a new section draft using Claude API."""
    if client is None:
        from oxidizer.llm import get_client
        client = get_client()
    if client is None:
        raise RuntimeError("Claude API not available. Use /oxidize write in Claude Code.")

    prompt = build_write_prompt(topic, section_type, profile)
    response = client.messages.create(
        model=model, max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    report = compute_style_report(text, profile)
    return WriteResult(text=text, topic=topic, section_type=section_type, style_report=report)
```

- [ ] **Step 3: Add `write` command to CLI**
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add oxidizer/engine/write.py tests/test_write.py oxidizer/cli.py
git commit -m "feat: write mode engine for section drafting"
```

---

### Task 17: De-AI Scan Command

**Files:**
- Create: `tests/test_scan.py`

- [ ] **Step 1: Write failing test**

`tests/test_scan.py`:
```python
from click.testing import CliRunner
from oxidizer.cli import cli
from pathlib import Path
import tempfile

def test_scan_detects_banned_words():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("We delve into the multifaceted tapestry of this groundbreaking research.")
        f.flush()
        result = runner.invoke(cli, ["scan", f.name, "--profile", "jinchi"])
    assert result.exit_code == 0
    assert "delve" in result.output.lower()

def test_scan_clean_text():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("We analyzed the data and found positive results.")
        f.flush()
        result = runner.invoke(cli, ["scan", f.name, "--profile", "jinchi"])
    assert result.exit_code == 0
    # Should report no AI-isms found
```

- [ ] **Step 2: Add `scan` command to CLI** that reads a document, runs `count_banned_words`, and displays results with line numbers and context using Rich highlighting.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git add tests/test_scan.py oxidizer/cli.py
git commit -m "feat: de-AI scan command"
```

---

### Task 18: Style Diff Command

**Files:**
- Create: `oxidizer/diff.py`
- Create: `tests/test_diff.py`

- [ ] **Step 1: Write failing test**

`tests/test_diff.py`:
```python
from oxidizer.diff import compute_style_diff, DiffResult

def test_diff_detects_changes():
    original = "The data was analyzed by our team. It's important to note the results."
    revised = "We analyzed the data; the results showed significant improvement."
    result = compute_style_diff(original, revised)
    assert isinstance(result, DiffResult)
    assert len(result.changes) > 0

def test_diff_annotations():
    original = "The data was analyzed."
    revised = "We analyzed the data."
    result = compute_style_diff(original, revised)
    # Should annotate passive->active voice change
    assert any("voice" in c.annotation.lower() or "passive" in c.annotation.lower() for c in result.changes)
```

- [ ] **Step 2: Implement `oxidizer/diff.py`** with `DiffResult` dataclass and `compute_style_diff` function using Python's `difflib` for text comparison plus style annotations.

- [ ] **Step 3: Add `diff` command to CLI**
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add oxidizer/diff.py tests/test_diff.py oxidizer/cli.py
git commit -m "feat: style diff command with annotations"
```

---

### Task 19: HTML Style Report

**Files:**
- Create: `oxidizer/scoring/html_report.py`
- Create: `tests/test_html_report.py`

- [ ] **Step 1: Write failing test**

`tests/test_html_report.py`:
```python
from oxidizer.scoring.html_report import generate_html_report
from oxidizer.scoring.reporter import StyleReport

def test_html_report_generates_valid_html():
    report = StyleReport(
        style_match_score=0.87,
        sentence_length_mean=23.5,
        sentence_length_std=9.8,
        sentence_length_target_mean=24.1,
        sentence_length_target_std=10.1,
        active_voice_ratio=0.88,
        active_voice_target=0.90,
        banned_words_found=[],
        semicolons_per_100=10.0,
        semicolons_target=12.0,
        parentheticals_per_100=22.0,
        parentheticals_target=25.0,
        contraction_count=0,
        transition_score=0.75,
        sub_scores={"sentence_length_mean": 0.97},
    )
    html = generate_html_report(report, title="Test Report")
    assert "<html" in html
    assert "0.87" in html
    assert "Style Match" in html
```

- [ ] **Step 2: Implement** using inline HTML/CSS (no external dependencies). Bar charts via CSS widths. Self-contained single-file HTML.

- [ ] **Step 3: Add `--html` flag to `score` and `revise` CLI commands**
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add oxidizer/scoring/html_report.py tests/test_html_report.py oxidizer/cli.py
git commit -m "feat: HTML style report with visual metrics"
```

---

### Task 20: Profile Validate and Compare Commands

**Files:**
- Create: `tests/test_validate_compare.py`

- [ ] **Step 1: Write failing tests**

`tests/test_validate_compare.py`:
```python
from click.testing import CliRunner
from oxidizer.cli import cli
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

def test_validate_profile_command():
    runner = CliRunner()
    result = runner.invoke(cli, [
        "validate-profile", "jinchi",
        "--samples", str(FIXTURES / "sample_methods.md"),
    ])
    assert result.exit_code == 0
    assert "score" in result.output.lower() or "Style" in result.output

def test_compare_command():
    runner = CliRunner()
    result = runner.invoke(cli, [
        "compare",
        str(FIXTURES / "sample_methods.md"),
        "--profiles", "jinchi",
    ])
    assert result.exit_code == 0
```

- [ ] **Step 2: Add `validate-profile` command** that scores writing samples against a profile and shows per-sample metrics.

- [ ] **Step 3: Add `compare` command** that scores text against multiple profiles and shows a comparison table.

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add tests/test_validate_compare.py oxidizer/cli.py
git commit -m "feat: profile validate and compare commands"
```

---

### Task 21: Claude Code Skill (Updated)

**Files:**
- Create: `skill/SKILL.md`

- [ ] **Step 1: Create the skill**

`skill/SKILL.md`:
```markdown
---
name: oxidize
description: Academic writing style engine. Revise, write, score, and scan documents in your personal voice.
---

# /oxidize

Academic writing style engine. Works with your personal style profile to write and revise in your voice.

## Commands

### /oxidize revise <file> [--sections X,Y]
Restyle a document to match your writing voice. Preserves citations, numbers, equations, abbreviations.

Steps:
1. Read the file and parse into sections
2. For each section, run: `conda run -n oxidizer oxidizer score <temp_section_file> --profile jinchi --json-output` to get baseline metrics
3. Extract entities using the Python extraction module
4. Restyle the section text using the style profile (Claude does this in-conversation)
5. Run scoring on the restyled text to verify style match
6. Verify all extracted entities appear in the restyled text
7. Output the restyled document

### /oxidize write <topic> --section <type>
Draft a new section in your voice. Types: intro, methods, results, discussion.

Steps:
1. Load the style profile from profiles/jinchi.yaml
2. Draft the section using the style instructions and few-shot examples
3. Score the output: `conda run -n oxidizer oxidizer score <output_file> --profile jinchi`
4. Show the style report

### /oxidize score <file>
Score text against your profile (local, no API): `conda run -n oxidizer oxidizer score <file> --profile jinchi`

### /oxidize scan <file>
Detect AI-isms (local, no API): `conda run -n oxidizer oxidizer scan <file> --profile jinchi`

### /oxidize diff <original> <revised>
Show what changed and why: `conda run -n oxidizer oxidizer diff <original> <revised> --profile jinchi`

## Profile Location
profiles/ directory at /Users/jinchiwei/arcadia/oxidizer/profiles/

## Scoring Note
Style scoring weights are empirical (not rigorously calibrated). Run `oxidizer validate-profile jinchi --samples <your_papers>` to check how well the profile matches your actual writing. Tune weights in the YAML if needed.
```

- [ ] **Step 2: Commit**

```bash
git add skill/
git commit -m "feat: Claude Code skill for /oxidize (revise, write, score, scan, diff)"
```

---

### Task 22: Expanded Integration Tests

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Add integration tests for all new commands**

Add to `tests/test_integration.py`:
```python
from oxidizer.scoring.metrics import count_banned_words
from oxidizer.profiles.loader import load_profile
from oxidizer.parsers.latex_parser import parse_latex
from oxidizer.llm import is_api_available
from pathlib import Path

PROFILES_DIR = Path(__file__).parent.parent / "profiles"

def test_scan_pipeline_local():
    """Scan detects banned words without API."""
    profile = load_profile("jinchi", search_paths=[PROFILES_DIR])
    text = "We delve into this multifaceted problem."
    found = count_banned_words(text, profile.vocabulary.banned_aiisms)
    assert "delve" in found
    assert "multifaceted" in found

def test_latex_parsing():
    """LaTeX parser extracts sections."""
    text = r"\section{Methods}\nWe collected data.\n\section{Results}\nAccuracy was 0.95."
    sections = parse_latex(text)
    assert len(sections) == 2

def test_api_mode_detection():
    """API availability check works."""
    result = is_api_available()
    assert isinstance(result, bool)

def test_all_cli_commands_have_help():
    """Every CLI command responds to --help."""
    from click.testing import CliRunner
    from oxidizer.cli import cli
    runner = CliRunner()
    for cmd in ["score", "scan", "revise", "diff", "validate-profile", "compare"]:
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed"
```

- [ ] **Step 2: Run full test suite**

Run: `conda run -n oxidizer python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: expanded integration tests for all commands"
```
