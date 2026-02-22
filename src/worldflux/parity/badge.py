"""Parity badge SVG generator for embedding in README files.

Produces shields.io-style flat badges showing model family, parity status,
and confidence level. No external dependencies required.
"""

from __future__ import annotations

from pathlib import Path


def _estimate_text_width(text: str) -> float:
    """Estimate rendered text width in pixels using per-character heuristics.

    Uses approximations for DejaVu Sans / Verdana at 11px, which is the
    standard shields.io badge font size.  Widths are intentionally generous
    to avoid clipping.
    """
    narrow = set("fijlrt!|()[]{}:;',. 1")
    wide = set("mwMWGOQDHNBUAV@%")
    width = 0.0
    for ch in text:
        if ch in narrow:
            width += 5.8
        elif ch in wide:
            width += 9.0
        elif ch.isupper():
            width += 7.8
        else:
            width += 6.7
    return width


def generate_badge_svg(
    family: str,
    passed: bool,
    confidence: float,
    margin: float,
) -> str:
    """Generate a shields.io-style flat SVG badge string.

    Parameters
    ----------
    family:
        Model family name displayed on the left side (e.g. "DreamerV3").
    passed:
        Whether the parity proof passed.
    confidence:
        Confidence level as a float in [0, 1] (e.g. 0.95 for 95%).
    margin:
        Margin ratio as a float (e.g. 0.05 for 5%).

    Returns
    -------
    str
        Complete SVG markup ready to write to a file or embed inline.
    """
    status = "PASS" if passed else "FAIL"
    right_label = f"{status} {confidence:.0%} (m={margin:.0%})"
    left_label = f"parity | {family}"

    status_color = "#4c1" if passed else "#e05d44"
    left_color = "#555"

    h_pad = 10
    left_width = _estimate_text_width(left_label) + 2 * h_pad
    right_width = _estimate_text_width(right_label) + 2 * h_pad
    total_width = left_width + right_width
    height = 20

    left_text_x = left_width / 2
    right_text_x = left_width + right_width / 2

    return f"""\
<svg xmlns="http://www.w3.org/2000/svg" width="{total_width:.0f}" height="{height}">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_width:.0f}" height="{height}" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{left_width:.0f}" height="{height}" fill="{left_color}"/>
    <rect x="{left_width:.0f}" width="{right_width:.0f}" height="{height}" fill="{status_color}"/>
    <rect width="{total_width:.0f}" height="{height}" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle"
     font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{left_text_x:.1f}" y="15" fill="#010101" fill-opacity=".3">{left_label}</text>
    <text x="{left_text_x:.1f}" y="14" fill="#fff">{left_label}</text>
    <text x="{right_text_x:.1f}" y="15" fill="#010101" fill-opacity=".3">{right_label}</text>
    <text x="{right_text_x:.1f}" y="14" fill="#fff">{right_label}</text>
  </g>
</svg>
"""


def save_badge(
    path: Path,
    family: str,
    passed: bool,
    confidence: float,
    margin: float,
) -> None:
    """Generate a parity badge and write it to *path*.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    path:
        Destination file path for the SVG badge.
    family:
        Model family name (e.g. "DreamerV3").
    passed:
        Whether the parity proof passed.
    confidence:
        Confidence level as a float in [0, 1].
    margin:
        Margin ratio as a float.
    """
    svg = generate_badge_svg(family=family, passed=passed, confidence=confidence, margin=margin)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
