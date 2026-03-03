"""Theme-aware Rich rendering helpers shared across all CLI commands."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from typing import Any

from rich.box import ROUNDED
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from ._app import MAX_PANEL_WIDTH, console
from ._theme import PANEL_PADDING, STATUS_ICONS


def _bounded_width() -> int:
    """Return panel width capped at *MAX_PANEL_WIDTH*."""
    return min(MAX_PANEL_WIDTH, shutil.get_terminal_size().columns - 2)


def section_header(title: str) -> None:
    """Print a themed section header with a horizontal rule."""
    console.print(Rule(title, style="wf.header"), width=_bounded_width())


def step_progress(step: int, total: int, title: str) -> None:
    """Print a wizard step progress indicator.

    Renders a thin rule, step counter, and a progress bar using
    filled (``━``) and unfilled (``┄``) segments.
    """
    w = _bounded_width()
    console.print()
    console.print(Rule(style="wf.section"), width=w)
    console.print(f"[wf.step]Step {step}/{total}[/wf.step] [wf.muted]--[/wf.muted] {title}")
    filled = step
    remaining = total - step
    bar = "[wf.step]" + "\u2501" * (filled * 4) + "[/wf.step]"
    if remaining:
        bar += "[wf.muted]" + "\u2504" * (remaining * 4) + "[/wf.muted]"
    console.print(bar)


def grouped_summary_panel(
    groups: dict[str, dict[str, str]],
    *,
    title: str | None = None,
    border: str = "wf.border",
) -> Panel:
    """Render grouped key-value data inside a single panel.

    *groups* is ``{"Section Name": {"Key": "Value", ...}, ...}``.
    Each section is separated by a blank line with the section name
    rendered as a header.
    """
    parts: list[str] = []
    max_key_len = 0
    for section_data in groups.values():
        for k in section_data:
            max_key_len = max(max_key_len, len(str(k)))

    first = True
    for section_name, section_data in groups.items():
        if not first:
            parts.append("")
        first = False
        parts.append(f"  [wf.header]{section_name}[/wf.header]")
        for key, value in section_data.items():
            padded = str(key).ljust(max_key_len)
            parts.append(f"    [wf.label]{padded}[/wf.label]  {value}")

    return Panel(
        "\n".join(parts),
        title=title,
        border_style=border,
        box=ROUNDED,
        padding=PANEL_PADDING,
        width=_bounded_width(),
    )


def key_value_panel(
    data: dict[str, Any],
    *,
    title: str | None = None,
    border: str = "wf.border",
) -> Panel:
    """Render a dict as an aligned key-value panel.

    Replaces scattered ``Panel.fit()`` calls with a consistent style.
    """
    max_key_len = max((len(str(k)) for k in data), default=0)
    lines: list[str] = []
    for key, value in data.items():
        padded = str(key).ljust(max_key_len)
        lines.append(f"[wf.label]{padded}[/wf.label]  {value}")
    return Panel(
        "\n".join(lines),
        title=title,
        border_style=border,
        box=ROUNDED,
        padding=PANEL_PADDING,
        width=_bounded_width(),
    )


def status_table(
    rows: Sequence[tuple[str, str, str]],
    *,
    title: str | None = None,
    columns: tuple[str, str, str] = ("", "Component", "Value"),
) -> Table:
    """Render a status-icon table (used by ``doctor``, etc.).

    Each row is ``(status_key, label, value)`` where *status_key* is one
    of ``"pass"``, ``"fail"``, ``"warn"``, or ``"info"``.
    """
    table = Table(title=title, show_lines=False, padding=(0, 2), width=_bounded_width())
    table.add_column(columns[0], width=3, no_wrap=True)
    table.add_column(columns[1], style="wf.label", no_wrap=True)
    table.add_column(columns[2])

    style_map = {
        "pass": "wf.ok",
        "fail": "wf.err",
        "warn": "wf.caution",
        "info": "wf.info",
    }

    for status_key, label, value in rows:
        icon = STATUS_ICONS.get(status_key, STATUS_ICONS["info"])
        style = style_map.get(status_key, "")
        table.add_row(f"[{style}]{icon}[/{style}]", label, value)

    return table


def result_banner(
    *,
    passed: bool,
    title: str | None = None,
    lines: Sequence[str] = (),
) -> Panel:
    """Render a PASS / FAIL banner panel (used by ``verify``, ``eval``, ``train``)."""
    if passed:
        default_title = f"{STATUS_ICONS['pass']} PASS"
        border = "wf.border.success"
    else:
        default_title = f"{STATUS_ICONS['fail']} FAIL"
        border = "wf.border.error"

    return Panel(
        "\n".join(lines),
        title=title or default_title,
        border_style=border,
        box=ROUNDED,
        padding=PANEL_PADDING,
        width=_bounded_width(),
    )


def metric_table(
    rows: Sequence[tuple[str, str, str, bool | None]],
    *,
    title: str | None = None,
) -> Table:
    """Render a metrics table (used by ``eval``).

    Each row is ``(metric_name, value, threshold, passed)``.
    """
    table = Table(title=title, width=_bounded_width())
    table.add_column("Metric", style="wf.label")
    table.add_column("Value", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status")

    for metric_name, value, threshold, passed in rows:
        if passed is True:
            status = f"[wf.ok]{STATUS_ICONS['pass']}[/wf.ok] [wf.pass]PASS[/wf.pass]"
        elif passed is False:
            status = f"[wf.err]{STATUS_ICONS['fail']}[/wf.err] [wf.fail]FAIL[/wf.fail]"
        else:
            status = f"[wf.muted]{STATUS_ICONS['info']}[/wf.muted] [wf.muted]info[/wf.muted]"
        table.add_row(metric_name, value, threshold, status)

    return table
