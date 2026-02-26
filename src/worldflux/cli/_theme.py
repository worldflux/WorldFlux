"""Centralized color palette, Rich Theme, and shared constants."""

from __future__ import annotations

from dataclasses import dataclass

from rich.theme import Theme


@dataclass(frozen=True)
class ColorPalette:
    """Immutable color palette for WorldFlux CLI.

    Designed for dark terminal backgrounds (~#1E1E2E).
    All text colors meet WCAG AA contrast ratio (>= 4.5:1).

    Contrast ratios against #1E1E2E:
      text        #CDD6F4  ~12.5:1   body text
      text_muted  #9399B2  ~ 5.2:1   secondary, still legible
      text_dim    #7A7F98  ~ 3.8:1   hints/shortcuts only (large text OK)
      primary     #7AA2F7  ~ 6.5:1   headers, brand
      accent      #89B4FA  ~ 7.2:1   selected items
    """

    # Brand
    primary: str = "#7AA2F7"
    primary_dim: str = "#5178B8"
    # Accents
    accent: str = "#89B4FA"
    # Semantic
    success: str = "#A6E3A1"
    warning: str = "#F9E2AF"
    error: str = "#F38BA8"
    info: str = "#89DCEB"
    # Text
    text: str = "#CDD6F4"
    text_muted: str = "#9399B2"
    text_dim: str = "#7A7F98"
    # Borders
    border: str = "#585B70"
    border_focus: str = "#7AA2F7"


PALETTE = ColorPalette()

# ---------------------------------------------------------------------------
# Rich Theme (semantic named styles)
# ---------------------------------------------------------------------------

WF_THEME = Theme(
    {
        # Structure
        "wf.header": f"bold {PALETTE.primary}",
        "wf.label": f"bold {PALETTE.text}",
        "wf.value": f"{PALETTE.text}",
        "wf.muted": f"{PALETTE.text_muted}",
        "wf.dim": f"{PALETTE.text_dim}",
        # Status indicators
        "wf.pass": f"bold {PALETTE.success}",
        "wf.fail": f"bold {PALETTE.error}",
        "wf.warn": f"bold {PALETTE.warning}",
        "wf.info": f"{PALETTE.info}",
        # Short indicators (icons)
        "wf.ok": f"{PALETTE.success}",
        "wf.err": f"{PALETTE.error}",
        "wf.caution": f"{PALETTE.warning}",
        # Brand
        "wf.brand": f"bold {PALETTE.primary}",
        "wf.accent": f"bold {PALETTE.accent}",
        # Borders
        "wf.border": f"{PALETTE.border}",
        "wf.border.success": f"{PALETTE.success}",
        "wf.border.error": f"{PALETTE.error}",
        "wf.border.info": f"{PALETTE.info}",
    }
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

STATUS_ICONS: dict[str, str] = {
    "pass": "\u2713",
    "fail": "\u2717",
    "warn": "!",
    "info": "\u2022",
}

PANEL_PADDING: tuple[int, int] = (1, 2)
