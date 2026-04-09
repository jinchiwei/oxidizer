"""Parse LaTeX documents into sections."""
import re
from oxidizer.parsers.markdown_parser import Section, _classify_context

SECTION_PATTERN = re.compile(
    r"\\(section|subsection|subsubsection)\{([^}]+)\}", re.MULTILINE
)


def parse_latex(text: str) -> list[Section]:
    r"""Parse LaTeX text into sections based on \section{} commands."""
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
