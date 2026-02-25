"""The ``models list`` and ``models info`` commands."""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from ._app import console, models_app


@models_app.command("list")
def models_list(
    maturity: str | None = typer.Option(
        None, "--maturity", "-m", help="Filter: reference, experimental, skeleton."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    format: str = typer.Option("table", "--format", "-f", help="table or json."),
) -> None:
    """List all available world model presets and aliases.

    [dim]Examples:[/dim]
      worldflux models list
      worldflux models list --maturity reference
      worldflux models list --format json
    """
    from worldflux.factory import list_models

    catalog = list_models(verbose=True, maturity=maturity)

    if not isinstance(catalog, dict):
        # list_models returns list[str] when verbose=False, but we forced True
        catalog = {k: {} for k in catalog}  # pragma: no cover

    if not catalog:
        console.print("[yellow]No models match the given filter.[/yellow]")
        raise typer.Exit(code=0)

    if format == "json":
        typer.echo(json.dumps(catalog, indent=2, default=str))
        return

    # Table output
    table = Table(title="WorldFlux Model Catalog", show_lines=False)
    table.add_column("Model ID", style="bold cyan", no_wrap=True)
    table.add_column("Description", max_width=42, no_wrap=True, overflow="ellipsis")
    table.add_column("Params", justify="right", style="dim", no_wrap=True)
    table.add_column("Maturity", no_wrap=True)

    for model_id, info in catalog.items():
        desc = _get(info, "description", "-")
        params = _get(info, "params", "-")
        mat = _get(info, "maturity", "-")
        mat_styled = _style_maturity(mat)
        table.add_row(model_id, desc, params, mat_styled)

    console.print(table)
    if not verbose:
        console.print("\n[dim]Tip: worldflux models info <id> for details.[/dim]")


@models_app.command("info")
def models_info(
    model: str = typer.Argument(..., help="Model ID or alias."),
    format: str = typer.Option("rich", "--format", "-f"),
) -> None:
    """Show detailed information about a specific model.

    [dim]Examples:[/dim]
      worldflux models info dreamer
      worldflux models info dreamerv3:size12m
      worldflux models info tdmpc2:5m --format json
    """
    from worldflux.factory import get_model_info

    try:
        info = get_model_info(model)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    if format == "json":
        typer.echo(json.dumps(info, indent=2, default=str))
        return

    model_id = info.get("model_id", model)
    lines = [f"[bold]Model ID:[/bold] {model_id}"]
    if "alias" in info:
        lines.append(f"[bold]Alias:[/bold] {info['alias']} -> {model_id}")
    for key in ("description", "params", "type", "maturity", "obs_shape", "action_dim"):
        if key in info:
            val = info[key]
            if key == "maturity":
                val = _style_maturity(str(val))
            lines.append(f"[bold]{_pretty_label(key)}:[/bold] {val}")

    # Show any remaining keys
    shown = {
        "model_id",
        "alias",
        "description",
        "params",
        "type",
        "maturity",
        "obs_shape",
        "action_dim",
    }
    for key, value in info.items():
        if key not in shown:
            lines.append(f"[bold]{_pretty_label(key)}:[/bold] {value}")

    console.print(
        Panel.fit(
            "\n".join(lines),
            title=f"Model: {model_id}",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(info: dict[str, Any], key: str, default: str = "-") -> str:
    val = info.get(key)
    return str(val) if val is not None else default


_FIELD_LABELS: dict[str, str] = {
    "description": "Description",
    "params": "Parameters",
    "type": "Type",
    "maturity": "Maturity",
    "obs_shape": "Observation Shape",
    "action_dim": "Action Dim",
    "default_obs": "Default Obs",
}


def _pretty_label(key: str) -> str:
    return _FIELD_LABELS.get(key, key.replace("_", " ").title())


def _style_maturity(maturity: str) -> str:
    if maturity == "reference":
        return "[bold green]reference[/bold green]"
    if maturity == "experimental":
        return "[yellow]experimental[/yellow]"
    if maturity == "skeleton":
        return "[dim]skeleton[/dim]"
    return maturity
