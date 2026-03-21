# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""The ``models list`` and ``models info`` commands."""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.table import Table

from ._app import console, models_app
from ._rich_output import key_value_panel


@models_app.command("list")
def models_list(
    maturity: str | None = typer.Option(
        None,
        "--maturity",
        "-m",
        help="Filter: reference, reference-family, experimental, skeleton.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    format: str = typer.Option("table", "--format", "-f", help="table or json."),
) -> None:
    """List all available world model presets and aliases.

    [dim]Examples:[/dim]
      worldflux models list
      worldflux models list --maturity reference
      worldflux models list --maturity reference-family
      worldflux models list --format json
    """
    from worldflux.factory import list_models

    catalog = list_models(verbose=True, maturity=maturity)

    if not isinstance(catalog, dict):
        # list_models returns list[str] when verbose=False, but we forced True
        catalog = {k: {} for k in catalog}  # pragma: no cover

    if not catalog:
        console.print("[wf.caution]No models match the given filter.[/wf.caution]")
        raise typer.Exit(code=0)

    if format == "json":
        typer.echo(json.dumps(catalog, indent=2, default=str))
        return

    # Table output
    table = Table(title="WorldFlux Model Catalog", show_lines=False)
    table.add_column("Model ID", style="wf.brand", no_wrap=True)
    table.add_column("Description", max_width=42, no_wrap=True, overflow="ellipsis")
    table.add_column("Params", justify="right", style="wf.muted", no_wrap=True)
    table.add_column("Support", no_wrap=True)
    table.add_column("Maturity", no_wrap=True)
    table.add_column("Parity Role", no_wrap=True)

    for model_id, info in catalog.items():
        desc = _get(info, "description", "-")
        params = _get(info, "params", "-")
        support_tier = _style_support_tier(_get(info, "support_tier", "-"))
        mat = _get(info, "maturity", "-")
        mat_styled = _style_maturity(mat)
        parity_role = _style_parity_role(_get(info, "parity_role", "-"))
        table.add_row(model_id, desc, params, support_tier, mat_styled, parity_role)

    console.print(table)
    if any(str(info.get("maturity")) == "reference" for info in catalog.values()):
        console.print(
            "\n[wf.muted]reference-family is an internal maturity tier; public proof "
            "claims require published evidence bundles.[/wf.muted]"
        )
    if not verbose:
        console.print("\n[wf.muted]Tip: worldflux models info <id> for details.[/wf.muted]")


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
        console.print(f"[wf.fail]Error:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    if format == "json":
        typer.echo(json.dumps(info, indent=2, default=str))
        return

    model_id = info.get("model_id", model)
    data: dict[str, Any] = {"Model ID": model_id}
    if "alias" in info:
        data["Alias"] = f"{info['alias']} -> {model_id}"
    for key in (
        "description",
        "params",
        "type",
        "support_tier",
        "maturity",
        "parity_role",
        "obs_shape",
        "action_dim",
    ):
        if key in info:
            val = info[key]
            if key == "support_tier":
                val = _style_support_tier(str(val))
            elif key == "maturity":
                val = _style_maturity(str(val))
            elif key == "parity_role":
                val = _style_parity_role(str(val))
            data[_pretty_label(key)] = val

    # Show any remaining keys
    shown = {
        "model_id",
        "alias",
        "description",
        "params",
        "type",
        "support_tier",
        "maturity",
        "parity_role",
        "obs_shape",
        "action_dim",
    }
    for key, value in info.items():
        if key not in shown:
            data[_pretty_label(key)] = value

    console.print(key_value_panel(data, title=f"Model: {model_id}", border="wf.border"))
    if str(info.get("maturity")) == "reference":
        console.print(
            "[wf.muted]reference-family is an internal maturity tier; it is not by "
            "itself a public proof claim.[/wf.muted]"
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
    "support_tier": "Support Tier",
    "maturity": "Maturity",
    "parity_role": "Parity Role",
    "obs_shape": "Observation Shape",
    "action_dim": "Action Dim",
    "default_obs": "Default Obs",
}


def _pretty_label(key: str) -> str:
    return _FIELD_LABELS.get(key, key.replace("_", " ").title())


def _style_maturity(maturity: str) -> str:
    if maturity == "reference":
        return "[wf.pass]reference-family[/wf.pass]"
    if maturity == "experimental":
        return "[wf.caution]experimental[/wf.caution]"
    if maturity == "skeleton":
        return "[wf.muted]skeleton[/wf.muted]"
    return maturity


def _style_support_tier(support_tier: str) -> str:
    if support_tier == "supported":
        return "[wf.pass]supported[/wf.pass]"
    if support_tier == "advanced":
        return "[wf.caution]advanced[/wf.caution]"
    if support_tier == "experimental":
        return "[wf.caution]experimental[/wf.caution]"
    if support_tier == "internal":
        return "[wf.muted]internal[/wf.muted]"
    return support_tier


def _style_parity_role(parity_role: str) -> str:
    if parity_role == "proof_canonical":
        return "[wf.pass]proof-canonical[/wf.pass]"
    if parity_role == "ci_only":
        return "[wf.caution]ci-only (not proof)[/wf.caution]"
    if parity_role == "reference_family":
        return "[wf.muted]reference-family[/wf.muted]"
    return parity_role
