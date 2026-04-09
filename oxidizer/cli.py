"""Oxidizer CLI — academic writing style engine.

Commands:
  score            Compute style report for a document.
  revise           AI-powered restyling of a document.
  write            AI-powered generation of a new section.
  scan             Scan a document for banned words.
  diff             Show styled diff between two files.
  validate-profile Score sample files against a profile.
  compare          Compare a document against multiple profiles.
"""
from __future__ import annotations

import json
import sys
import webbrowser
import tempfile
import difflib
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Document parsing helper
# ---------------------------------------------------------------------------

def _parse_document(path: Path):
    """Detect file format by extension and parse into sections.

    Args:
        path: Path to the document file.

    Returns:
        List of Section objects.
    """
    suffix = path.suffix.lower()

    if suffix == ".md":
        from oxidizer.parsers.markdown_parser import parse_markdown
        return parse_markdown(path.read_text(encoding="utf-8"))

    elif suffix == ".docx":
        from oxidizer.parsers.docx_parser import parse_docx
        return parse_docx(str(path))

    elif suffix in (".tex", ".latex"):
        from oxidizer.parsers.latex_parser import parse_latex
        return parse_latex(path.read_text(encoding="utf-8"))

    elif suffix == ".pdf":
        from oxidizer.parsers.pdf_parser import parse_pdf
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return parse_pdf(str(path))

    else:
        # Fallback: try as markdown
        from oxidizer.parsers.markdown_parser import parse_markdown
        return parse_markdown(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Shared score display helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    """Return rich color markup name for a score."""
    if score > 0.85:
        return "green"
    elif score >= 0.60:
        return "yellow"
    else:
        return "red"


def _display_report_table(report, section_heading: str = "") -> None:
    """Render a StyleReport as a Rich table to the console."""
    from oxidizer.scoring.reporter import StyleReport

    title = f"Style Report"
    if section_heading:
        title += f" — {section_heading}"

    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )
    table.add_column("Metric", style="bold", min_width=30)
    table.add_column("Actual", justify="right", min_width=10)
    table.add_column("Target", justify="right", min_width=10)
    table.add_column("Score", justify="right", min_width=8)

    sub = report.sub_scores

    def fmt_score(s: float) -> str:
        color = _score_color(s)
        return f"[{color}]{s:.2f}[/{color}]"

    def fmt_val(v: float) -> str:
        return f"{v:.2f}"

    table.add_row(
        "Sentence Length (Mean)",
        fmt_val(report.sentence_length_mean),
        fmt_val(report.sentence_length_target_mean),
        fmt_score(sub.get("sentence_length_mean", 0.0)),
    )
    table.add_row(
        "Sentence Length (Std Dev)",
        fmt_val(report.sentence_length_std),
        fmt_val(report.sentence_length_target_std),
        fmt_score(sub.get("sentence_length_variance", 0.0)),
    )
    table.add_row(
        "Active Voice Ratio",
        fmt_val(report.active_voice_ratio),
        fmt_val(report.active_voice_target),
        fmt_score(sub.get("active_voice", 0.0)),
    )
    banned_count = len(report.banned_words_found)
    table.add_row(
        f"Banned Words ({banned_count} found)",
        str(banned_count),
        "0",
        fmt_score(sub.get("banned_words", 0.0)),
    )
    table.add_row(
        "Semicolons per 100 Sentences",
        fmt_val(report.semicolons_per_100),
        fmt_val(report.semicolons_target),
        fmt_score(sub.get("semicolons", 0.0)),
    )
    table.add_row(
        "Parentheticals per 100 Sentences",
        fmt_val(report.parentheticals_per_100),
        fmt_val(report.parentheticals_target),
        fmt_score(sub.get("parentheticals", 0.0)),
    )
    table.add_row(
        "Transitions",
        fmt_val(report.transition_score),
        "preferred",
        fmt_score(sub.get("transitions", 0.0)),
    )
    table.add_row(
        "Contractions",
        str(report.contraction_count),
        "0",
        fmt_score(sub.get("contractions", 0.0)),
    )

    overall = report.style_match_score
    overall_color = _score_color(overall)
    table.add_section()
    table.add_row(
        "[bold]Overall Style Match[/bold]",
        "",
        "",
        f"[bold {overall_color}]{overall:.2f}[/bold {overall_color}]",
    )

    console.print(table)

    if report.banned_words_found:
        console.print(
            f"[red]Banned words found:[/red] {', '.join(report.banned_words_found)}"
        )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="oxidizer")
def cli():
    """Oxidizer — Academic Writing Style Engine."""
    pass


# ---------------------------------------------------------------------------
# score command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--profile", "-p", required=True, help="Profile name (e.g. jinchi)")
@click.option("--json-output", is_flag=True, default=False, help="Output JSON instead of table")
@click.option("--html", "html_output", is_flag=True, default=False, help="Generate and open HTML report")
def score(file: Path, profile: str, json_output: bool, html_output: bool):
    """Compute style report for FILE against a profile."""
    from oxidizer.profiles.loader import load_profile
    from oxidizer.scoring.reporter import compute_style_report

    # Load profile
    try:
        prof = load_profile(profile)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Parse document
    try:
        sections = _parse_document(file)
    except Exception as e:
        console.print(f"[red]Error parsing document:[/red] {e}")
        sys.exit(1)

    if not sections:
        console.print("[yellow]Warning:[/yellow] No sections found in document.")
        sys.exit(0)

    all_reports = []
    for section in sections:
        text = section.body
        if not text.strip():
            continue
        report = compute_style_report(text, prof)
        all_reports.append((section.heading or "(untitled)", report))

    if not all_reports:
        console.print("[yellow]No content to score.[/yellow]")
        sys.exit(0)

    if json_output:
        output = [
            {"section": heading, "report": rep.to_dict()}
            for heading, rep in all_reports
        ]
        click.echo(json.dumps(output, indent=2))
        return

    if html_output:
        from oxidizer.scoring.html_report import generate_html_report
        # Use aggregate or first section
        if len(all_reports) == 1:
            heading, rep = all_reports[0]
            html_str = generate_html_report(rep, title=f"{file.name} — Style Report")
        else:
            # Use the first non-trivial report
            heading, rep = all_reports[0]
            html_str = generate_html_report(rep, title=f"{file.name} — Style Report ({heading})")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html_str)
            tmp_path = f.name

        console.print(f"[green]HTML report written to:[/green] {tmp_path}")
        webbrowser.open(f"file://{tmp_path}")
        return

    # Default: Rich table output
    console.print(f"\n[bold]Document:[/bold] {file}")
    console.print(f"[bold]Profile:[/bold] {prof.name}\n")

    for heading, rep in all_reports:
        _display_report_table(rep, section_heading=heading)
        console.print()


# ---------------------------------------------------------------------------
# revise command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--profile", "-p", required=True, help="Profile name (e.g. jinchi)")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output file path")
@click.option("--sections", default=None, help="Comma-separated section numbers to revise (1-indexed)")
@click.option("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
def revise(file: Path, profile: str, output: Optional[Path], sections: Optional[str], model: str):
    """Revise FILE to match a writing profile using AI."""
    from oxidizer.llm import is_api_available
    from oxidizer.profiles.loader import load_profile
    from oxidizer.engine.revise import revise_section

    if not is_api_available():
        console.print(
            "[red]Error:[/red] No API key found.\n"
            "Set [bold]ANTHROPIC_API_KEY[/bold] in your environment to use the revise command.\n"
            "Example: [dim]export ANTHROPIC_API_KEY=sk-ant-...[/dim]"
        )
        sys.exit(1)

    # Load profile
    try:
        prof = load_profile(profile)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Parse document
    try:
        all_sections = _parse_document(file)
    except Exception as e:
        console.print(f"[red]Error parsing document:[/red] {e}")
        sys.exit(1)

    # Filter sections if requested
    if sections:
        try:
            indices = [int(x.strip()) - 1 for x in sections.split(",")]
            selected = [all_sections[i] for i in indices if 0 <= i < len(all_sections)]
        except (ValueError, IndexError) as e:
            console.print(f"[red]Error parsing --sections:[/red] {e}")
            sys.exit(1)
    else:
        selected = all_sections

    if not selected:
        console.print("[yellow]No sections to revise.[/yellow]")
        sys.exit(0)

    console.print(f"\n[bold]Revising:[/bold] {file}")
    console.print(f"[bold]Profile:[/bold] {prof.name}")
    console.print(f"[bold]Sections:[/bold] {len(selected)}\n")

    revised_parts: list[str] = []
    all_results = []

    for i, section in enumerate(selected, 1):
        heading_display = section.heading or f"Section {i}"
        console.print(f"[cyan]Revising:[/cyan] {heading_display} ...")

        if not section.body.strip():
            console.print(f"  [dim]Skipping empty section.[/dim]")
            continue

        result = revise_section(section, prof, model=model)
        all_results.append(result)

        if section.heading:
            revised_parts.append(f"## {section.heading}\n\n{result.text}")
        else:
            revised_parts.append(result.text)

        if result.style_report:
            _display_report_table(result.style_report, section_heading=heading_display)

        if result.warnings:
            for w in result.warnings:
                console.print(f"  [yellow]Warning:[/yellow] {w}")

        console.print()

    # Determine output path
    if output is None:
        output = file.parent / (file.stem + "_revised" + file.suffix)

    output.write_text("\n\n".join(revised_parts), encoding="utf-8")
    console.print(f"[green]Revised document written to:[/green] {output}")

    # Write report.json alongside
    report_path = output.parent / (output.stem + "_report.json")
    report_data = []
    for result in all_results:
        entry = {
            "heading": result.heading,
            "retries": result.retries,
            "warnings": result.warnings,
        }
        if result.style_report:
            entry["style_report"] = result.style_report.to_dict()
        if result.preservation:
            entry["preservation"] = {
                "passed": result.preservation.passed,
                "missing": result.preservation.missing,
            }
        report_data.append(entry)

    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    console.print(f"[green]Report JSON written to:[/green] {report_path}")


# ---------------------------------------------------------------------------
# write command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("topic")
@click.option("--profile", "-p", required=True, help="Profile name (e.g. jinchi)")
@click.option("--section", "section_type", required=True,
              type=click.Choice(["intro", "methods", "results", "discussion", "other"]),
              help="Section type to generate")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output file path")
@click.option("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
def write(topic: str, profile: str, section_type: str, output: Optional[Path], model: str):
    """Generate new academic text for TOPIC using AI."""
    from oxidizer.llm import is_api_available
    from oxidizer.profiles.loader import load_profile
    from oxidizer.engine.write import write_section

    if not is_api_available():
        console.print(
            "[red]Error:[/red] No API key found.\n"
            "Set [bold]ANTHROPIC_API_KEY[/bold] in your environment to use the write command.\n"
            "Example: [dim]export ANTHROPIC_API_KEY=sk-ant-...[/dim]"
        )
        sys.exit(1)

    # Load profile
    try:
        prof = load_profile(profile)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print(f"\n[bold]Generating:[/bold] {section_type} section")
    console.print(f"[bold]Profile:[/bold] {prof.name}")
    console.print(f"[bold]Topic:[/bold] {topic}\n")

    result = write_section(topic, section_type, prof, model=model)

    console.print("[bold cyan]Generated Text:[/bold cyan]")
    console.print(result.text)
    console.print()

    if result.style_report:
        _display_report_table(result.style_report, section_heading=f"{section_type.capitalize()} (generated)")

    if result.warnings:
        for w in result.warnings:
            console.print(f"[yellow]Warning:[/yellow] {w}")

    if output:
        output.write_text(result.text, encoding="utf-8")
        console.print(f"\n[green]Output written to:[/green] {output}")


# ---------------------------------------------------------------------------
# scan command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--profile", "-p", required=True, help="Profile name (e.g. jinchi)")
def scan(file: Path, profile: str):
    """Scan FILE for banned words defined in the profile."""
    from oxidizer.profiles.loader import load_profile
    from oxidizer.scoring.metrics import count_banned_words

    # Load profile
    try:
        prof = load_profile(profile)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Parse document
    try:
        sections = _parse_document(file)
    except Exception as e:
        console.print(f"[red]Error parsing document:[/red] {e}")
        sys.exit(1)

    banned_list = prof.vocabulary.banned_aiisms
    if not banned_list:
        console.print(
            f"[yellow]Profile '{prof.name}' has no banned words configured.[/yellow]"
        )
        sys.exit(0)

    console.print(f"\n[bold]Scanning:[/bold] {file}")
    console.print(f"[bold]Profile:[/bold] {prof.name}")
    console.print(f"[bold]Banned word list:[/bold] {', '.join(banned_list)}\n")

    total_found = 0
    any_found = False

    for section in sections:
        heading = section.heading or "(untitled)"
        text = section.body
        if not text.strip():
            continue

        found = count_banned_words(text, banned_list)
        if not found:
            continue

        any_found = True
        total_found += len(found)

        table = Table(
            title=f"Section: {heading}",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold red",
        )
        table.add_column("Banned Word", style="red bold", min_width=20)
        table.add_column("Context", min_width=60)

        lines = text.splitlines()
        for banned_word in found:
            # Find lines containing this word (case-insensitive)
            word_lower = banned_word.lower()
            context_lines: list[str] = []
            for line_num, line in enumerate(lines, 1):
                if word_lower in line.lower():
                    # Show truncated context
                    truncated = line.strip()
                    if len(truncated) > 80:
                        idx = truncated.lower().find(word_lower)
                        start = max(0, idx - 30)
                        end = min(len(truncated), idx + len(word_lower) + 30)
                        truncated = ("..." if start > 0 else "") + truncated[start:end] + ("..." if end < len(truncated) else "")
                    context_lines.append(f"[dim]L{line_num}:[/dim] {truncated}")

            context_str = "\n".join(context_lines[:3])  # Show max 3 occurrences
            table.add_row(banned_word, context_str)

        console.print(table)

    if not any_found:
        console.print(f"[green]No banned words found in {file.name}.[/green]")
    else:
        console.print(f"\n[red]Total banned word occurrences found:[/red] {total_found}")


# ---------------------------------------------------------------------------
# diff command
# ---------------------------------------------------------------------------

@cli.command(name="diff")
@click.argument("original", type=click.Path(exists=True, path_type=Path))
@click.argument("revised", type=click.Path(exists=True, path_type=Path))
@click.option("--profile", "-p", default=None, help="Profile name for style annotation")
def diff_cmd(original: Path, revised: Path, profile: Optional[str]):
    """Show styled diff between ORIGINAL and REVISED files."""
    from oxidizer.diff import compute_diff

    original_text = original.read_text(encoding="utf-8")
    revised_text = revised.read_text(encoding="utf-8")

    original_report = None
    revised_report = None
    banned_list = None

    if profile:
        from oxidizer.profiles.loader import load_profile
        from oxidizer.scoring.reporter import compute_style_report

        try:
            prof = load_profile(profile)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

        banned_list = prof.vocabulary.banned_aiisms

        orig_sections = _parse_document(original)
        rev_sections = _parse_document(revised)

        orig_text_full = " ".join(s.body for s in orig_sections if s.body.strip())
        rev_text_full = " ".join(s.body for s in rev_sections if s.body.strip())

        if orig_text_full.strip():
            original_report = compute_style_report(orig_text_full, prof).to_dict()
        if rev_text_full.strip():
            revised_report = compute_style_report(rev_text_full, prof).to_dict()

    diff_result = compute_diff(
        original_text,
        revised_text,
        banned_list=banned_list,
        original_report=original_report,
        revised_report=revised_report,
    )

    console.print(f"\n[bold]Diff:[/bold] {original} → {revised}\n")

    if not diff_result.changes:
        console.print("[green]Files are identical.[/green]")
        return

    # Display changes
    for change in diff_result.changes:
        console.print(f"[dim]Line {change.line_num}[/dim] [italic yellow]({change.annotation})[/italic yellow]")
        if change.original:
            console.print(f"  [red]- {change.original}[/red]")
        if change.revised:
            console.print(f"  [green]+ {change.revised}[/green]")
        console.print()

    console.print(f"[bold]Total changes:[/bold] {len(diff_result.changes)}")

    # Style report comparison if available
    if original_report and revised_report:
        orig_score = original_report.get("style_match_score", 0.0)
        rev_score = revised_report.get("style_match_score", 0.0)
        delta = rev_score - orig_score
        delta_color = "green" if delta >= 0 else "red"
        delta_sign = "+" if delta >= 0 else ""
        console.print(
            f"\n[bold]Style Score:[/bold] "
            f"[dim]{orig_score:.2f}[/dim] → "
            f"[bold]{rev_score:.2f}[/bold] "
            f"[{delta_color}]({delta_sign}{delta:.2f})[/{delta_color}]"
        )


# ---------------------------------------------------------------------------
# validate-profile command
# ---------------------------------------------------------------------------

@cli.command("validate-profile")
@click.argument("name")
@click.option("--samples", required=True, type=click.Path(exists=True, path_type=Path),
              help="Path to a directory or single file with sample documents")
def validate_profile(name: str, samples: Path):
    """Validate profile NAME against sample documents."""
    from oxidizer.profiles.loader import load_profile
    from oxidizer.scoring.reporter import compute_style_report

    # Load profile
    try:
        prof = load_profile(name)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Collect sample files
    if samples.is_dir():
        sample_files: list[Path] = []
        for ext in ("*.md", "*.docx", "*.tex", "*.latex"):
            sample_files.extend(samples.glob(ext))
        if not sample_files:
            console.print(f"[yellow]No .md/.docx/.tex files found in {samples}[/yellow]")
            sys.exit(0)
    else:
        sample_files = [samples]

    console.print(f"\n[bold]Validating profile:[/bold] {prof.name}")
    console.print(f"[bold]Samples:[/bold] {len(sample_files)} file(s)\n")

    table = Table(
        title=f"Profile Validation — {prof.name}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("File", min_width=30)
    table.add_column("Sections", justify="right", min_width=8)
    table.add_column("Avg Score", justify="right", min_width=10)

    all_scores: list[float] = []

    for sample_path in sorted(sample_files):
        try:
            sections = _parse_document(sample_path)
        except Exception as e:
            table.add_row(sample_path.name, "ERR", f"[red]{e}[/red]")
            continue

        scores: list[float] = []
        for section in sections:
            if section.body.strip():
                rep = compute_style_report(section.body, prof)
                scores.append(rep.style_match_score)

        if not scores:
            table.add_row(sample_path.name, "0", "[dim]N/A[/dim]")
            continue

        avg = sum(scores) / len(scores)
        all_scores.extend(scores)
        color = _score_color(avg)
        table.add_row(
            sample_path.name,
            str(len(scores)),
            f"[{color}]{avg:.2f}[/{color}]",
        )

    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        overall_color = _score_color(overall)
        table.add_section()
        table.add_row(
            "[bold]Overall Average[/bold]",
            str(len(all_scores)),
            f"[bold {overall_color}]{overall:.2f}[/bold {overall_color}]",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--profiles", required=True, help="Comma-separated profile names (e.g. jinchi,other)")
def compare(file: Path, profiles: str):
    """Compare FILE against multiple style profiles."""
    from oxidizer.profiles.loader import load_profile
    from oxidizer.scoring.reporter import compute_style_report

    profile_names = [p.strip() for p in profiles.split(",") if p.strip()]
    if not profile_names:
        console.print("[red]Error:[/red] No profile names provided.")
        sys.exit(1)

    # Load all profiles
    loaded_profiles = []
    for pname in profile_names:
        try:
            prof = load_profile(pname)
            loaded_profiles.append(prof)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    # Parse document
    try:
        sections = _parse_document(file)
    except Exception as e:
        console.print(f"[red]Error parsing document:[/red] {e}")
        sys.exit(1)

    # Gather all text
    all_text = " ".join(s.body for s in sections if s.body.strip())
    if not all_text.strip():
        console.print("[yellow]No content found in document.[/yellow]")
        sys.exit(0)

    console.print(f"\n[bold]Document:[/bold] {file}")
    console.print(f"[bold]Comparing against {len(loaded_profiles)} profile(s)[/bold]\n")

    table = Table(
        title=f"Profile Comparison — {file.name}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Profile", min_width=20)
    table.add_column("Style Match", justify="right", min_width=12)
    table.add_column("Active Voice", justify="right", min_width=12)
    table.add_column("Banned Words", justify="right", min_width=12)
    table.add_column("Sentence Len", justify="right", min_width=12)

    for prof in loaded_profiles:
        rep = compute_style_report(all_text, prof)
        score_color = _score_color(rep.style_match_score)
        av_color = _score_color(rep.sub_scores.get("active_voice", 0.0))
        bw_color = _score_color(rep.sub_scores.get("banned_words", 0.0))
        sl_color = _score_color(rep.sub_scores.get("sentence_length_mean", 0.0))

        table.add_row(
            prof.name,
            f"[{score_color}]{rep.style_match_score:.2f}[/{score_color}]",
            f"[{av_color}]{rep.active_voice_ratio:.2f}[/{av_color}]",
            f"[{bw_color}]{len(rep.banned_words_found)}[/{bw_color}]",
            f"[{sl_color}]{rep.sentence_length_mean:.1f}[/{sl_color}]",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
