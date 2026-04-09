"""Parse PDF files into sections — best-effort stub using pdfplumber.

WARNING: PDF is an inherently unreliable format for academic papers. Structural
information (headings, sections) is not preserved in the PDF byte stream, so
this parser returns the entire document as a single section. For high-fidelity
parsing prefer DOCX, LaTeX, or Markdown sources.
"""
import warnings

import pdfplumber

from oxidizer.parsers.markdown_parser import Section


def parse_pdf(path: str) -> list[Section]:
    """Extract all text from a PDF and return it as a single Section.

    Parameters
    ----------
    path:
        Filesystem path to the PDF file.

    Returns
    -------
    list[Section]
        A one-element list containing all extracted text with heading="",
        context="other", and level=0.
    """
    warnings.warn(
        "PDF parsing is a best-effort stub. Structural information (headings, "
        "sections) cannot be reliably extracted from PDF files. Use DOCX, "
        "LaTeX, or Markdown sources for accurate section parsing.",
        UserWarning,
        stacklevel=2,
    )

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

    body = "\n\n".join(pages).strip()
    return [Section(heading="", body=body, context="other", level=0)]
