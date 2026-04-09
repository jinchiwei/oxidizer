"""HTML report generator for Oxidizer style reports.

Generates a self-contained HTML file with:
- Overall style match score (big number at top)
- CSS-only bar chart for each metric (no JS dependencies)
- Color coding: green (>0.85), yellow (0.6-0.85), red (<0.6)
- Banned words list if any
- Inline CSS, no external dependencies
"""
from __future__ import annotations

import html
from oxidizer.scoring.reporter import StyleReport


def _color_class(score: float) -> str:
    """Return a CSS class name based on score thresholds."""
    if score > 0.85:
        return "green"
    elif score >= 0.60:
        return "yellow"
    else:
        return "red"


def _color_for_score(score: float) -> str:
    """Return hex color based on score thresholds."""
    if score > 0.85:
        return "#27ae60"
    elif score >= 0.60:
        return "#f39c12"
    else:
        return "#e74c3c"


def _bar_html(label: str, score: float, actual: float | None = None, target: float | None = None) -> str:
    """Render a single metric bar row."""
    pct = round(score * 100, 1)
    color = _color_for_score(score)
    bar_width = max(0, min(100, pct))

    detail_parts: list[str] = []
    if actual is not None:
        detail_parts.append(f"actual: {actual:.2f}")
    if target is not None:
        detail_parts.append(f"target: {target:.2f}")
    detail_parts.append(f"score: {pct}%")
    detail_str = " | ".join(detail_parts)

    escaped_label = html.escape(label)
    escaped_detail = html.escape(detail_str)

    return f"""
    <div class="metric-row">
      <div class="metric-label">{escaped_label}</div>
      <div class="bar-container">
        <div class="bar" style="width: {bar_width}%; background-color: {color};"></div>
      </div>
      <div class="metric-detail">{escaped_detail}</div>
    </div>"""


def generate_html_report(report: StyleReport, title: str = "Style Report") -> str:
    """Generate a self-contained HTML style report.

    Args:
        report: A populated StyleReport from compute_style_report().
        title: Page title and heading for the report.

    Returns:
        A complete HTML string with inline CSS, no external dependencies.
    """
    overall_score = report.style_match_score
    overall_pct = round(overall_score * 100, 1)
    overall_color = _color_for_score(overall_score)
    escaped_title = html.escape(title)

    # Build metric bars
    sub = report.sub_scores
    bars_html = ""

    bars_html += _bar_html(
        "Sentence Length (Mean)",
        sub.get("sentence_length_mean", 0.0),
        actual=report.sentence_length_mean,
        target=report.sentence_length_target_mean,
    )
    bars_html += _bar_html(
        "Sentence Length (Variance)",
        sub.get("sentence_length_variance", 0.0),
        actual=report.sentence_length_std,
        target=report.sentence_length_target_std,
    )
    bars_html += _bar_html(
        "Active Voice",
        sub.get("active_voice", 0.0),
        actual=report.active_voice_ratio,
        target=report.active_voice_target,
    )
    bars_html += _bar_html(
        "Banned Words",
        sub.get("banned_words", 0.0),
    )
    bars_html += _bar_html(
        "Semicolons per 100 Sentences",
        sub.get("semicolons", 0.0),
        actual=report.semicolons_per_100,
        target=report.semicolons_target,
    )
    bars_html += _bar_html(
        "Parentheticals per 100 Sentences",
        sub.get("parentheticals", 0.0),
        actual=report.parentheticals_per_100,
        target=report.parentheticals_target,
    )
    bars_html += _bar_html(
        "Transitions",
        sub.get("transitions", 0.0),
        actual=report.transition_score,
    )
    bars_html += _bar_html(
        "Contractions",
        sub.get("contractions", 0.0),
        actual=float(report.contraction_count),
        target=0.0,
    )

    # Banned words section
    banned_section = ""
    if report.banned_words_found:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in report.banned_words_found)
        banned_section = f"""
    <div class="banned-section">
      <h2>Banned Words Found</h2>
      <ul class="banned-list">
        {items}
      </ul>
    </div>"""
    else:
        banned_section = """
    <div class="banned-section clean">
      <p>No banned words found.</p>
    </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escaped_title}</title>
  <style>
    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #f5f6fa;
      color: #2c3e50;
      padding: 2rem;
      max-width: 900px;
      margin: 0 auto;
    }}

    h1 {{
      font-size: 1.6rem;
      font-weight: 700;
      margin-bottom: 1.5rem;
      color: #2c3e50;
      border-bottom: 2px solid #dde1e7;
      padding-bottom: 0.5rem;
    }}

    h2 {{
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 0.75rem;
      color: #34495e;
    }}

    .score-hero {{
      background: white;
      border-radius: 12px;
      padding: 2rem;
      text-align: center;
      margin-bottom: 1.5rem;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}

    .score-hero .label {{
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #7f8c8d;
      margin-bottom: 0.5rem;
    }}

    .score-hero .big-number {{
      font-size: 4rem;
      font-weight: 800;
      color: {overall_color};
      line-height: 1;
    }}

    .score-hero .pct-label {{
      font-size: 1.2rem;
      color: #7f8c8d;
      margin-top: 0.25rem;
    }}

    .metrics-card {{
      background: white;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}

    .metric-row {{
      display: flex;
      align-items: center;
      margin-bottom: 0.85rem;
    }}

    .metric-label {{
      width: 240px;
      font-size: 0.85rem;
      font-weight: 500;
      flex-shrink: 0;
      color: #2c3e50;
    }}

    .bar-container {{
      flex: 1;
      background: #edf0f5;
      border-radius: 6px;
      height: 16px;
      overflow: hidden;
      margin: 0 1rem;
    }}

    .bar {{
      height: 100%;
      border-radius: 6px;
      transition: width 0.3s ease;
    }}

    .metric-detail {{
      width: 220px;
      font-size: 0.78rem;
      color: #7f8c8d;
      flex-shrink: 0;
      text-align: right;
    }}

    .banned-section {{
      background: white;
      border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}

    .banned-section.clean {{
      color: #27ae60;
      font-weight: 500;
    }}

    .banned-list {{
      list-style: none;
      padding: 0;
    }}

    .banned-list li {{
      background: #fef5f5;
      border-left: 3px solid #e74c3c;
      padding: 0.4rem 0.75rem;
      margin-bottom: 0.4rem;
      border-radius: 0 4px 4px 0;
      font-family: monospace;
      font-size: 0.9rem;
      color: #c0392b;
    }}

    .footer {{
      margin-top: 2rem;
      text-align: center;
      font-size: 0.75rem;
      color: #bdc3c7;
    }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>

  <div class="score-hero">
    <div class="label">Overall Style Match Score</div>
    <div class="big-number">{overall_pct}</div>
    <div class="pct-label">out of 100</div>
  </div>

  <div class="metrics-card">
    <h2>Metric Breakdown</h2>
    {bars_html}
  </div>

  {banned_section}

  <div class="footer">Generated by Oxidizer &mdash; Academic Writing Style Engine</div>
</body>
</html>"""

    return html_content
