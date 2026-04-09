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
