"""Theme-aware Rich rendering helpers shared across all CLI commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.box import ROUNDED
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from ._app import console
from ._theme import PANEL_PADDING, STATUS_ICONS


def section_header(title: str) -> None:
    """Print a themed section header with a horizontal rule."""
    console.print(Rule(title, style="wf.header"))


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
    table = Table(title=title, show_lines=False, padding=(0, 2))
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
    )


def metric_table(
    rows: Sequence[tuple[str, str, str, bool | None]],
    *,
    title: str | None = None,
) -> Table:
    """Render a metrics table (used by ``eval``).

    Each row is ``(metric_name, value, threshold, passed)``.
    """
    table = Table(title=title)
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
