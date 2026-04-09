"""Document parsers for Oxidizer."""
from oxidizer.parsers.markdown_parser import Section, parse_markdown
from oxidizer.parsers.docx_parser import parse_docx
from oxidizer.parsers.pdf_parser import parse_pdf
from oxidizer.parsers.latex_parser import parse_latex

__all__ = ["Section", "parse_markdown", "parse_docx", "parse_pdf", "parse_latex"]
