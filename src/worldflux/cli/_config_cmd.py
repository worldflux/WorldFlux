# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""The ``config`` subcommand group: inspect, diff, validate, schema.

Provides introspection and validation tools for WorldFlux model
configurations (API-05/API-06).
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from ._app import config_app, console

# ---------------------------------------------------------------------------
# config inspect
# ---------------------------------------------------------------------------


@config_app.command("inspect")
def config_inspect(
    model: str = typer.Argument(..., help="Model preset ID (e.g. dreamerv3:size12m)."),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table or json."),
) -> None:
    """Show all configuration fields for a model preset.

    Displays field name, type, default value, and docstring-derived
    description for every parameter.

    [dim]Examples:[/dim]
      worldflux config inspect dreamerv3:size12m
      worldflux config inspect tdmpc2:5m --format json
    """
    from worldflux.factory import get_config

    try:
        cfg = get_config(model)
    except (ValueError, Exception) as exc:
        console.print(f"[wf.fail]Error:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    fields_info = _collect_fields(cfg)

    if format == "json":
        typer.echo(json.dumps(fields_info, indent=2, default=str))
        return

    title = f"{type(cfg).__name__} ({model})"
    console.print(f"\n[wf.header]{title}[/wf.header]")
    console.print("[wf.border]" + "=" * len(title) + "[/wf.border]\n")

    table = Table(show_header=True, show_lines=False, pad_edge=True)
    table.add_column("Field", style="wf.brand", no_wrap=True)
    table.add_column("Type", style="wf.muted", no_wrap=True)
    table.add_column("Value", style="wf.value", no_wrap=True)
    table.add_column("Default", style="wf.dim", no_wrap=True)

    for info in fields_info:
        current = str(info["value"])
        default = str(info["default"])
        # Highlight changed values
        val_style = "wf.pass" if current != default else ""
        table.add_row(
            info["name"],
            info["type"],
            f"[{val_style}]{current}[/{val_style}]" if val_style else current,
            default,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# config diff
# ---------------------------------------------------------------------------


@config_app.command("diff")
def config_diff(
    model_a: str = typer.Argument(..., help="First model preset."),
    model_b: str = typer.Argument(..., help="Second model preset."),
    format: str = typer.Option("table", "--format", "-f", help="table or json."),
) -> None:
    """Show differences between two model presets.

    Only fields whose values differ are displayed.

    [dim]Examples:[/dim]
      worldflux config diff dreamerv3:size12m dreamerv3:size200m
      worldflux config diff tdmpc2:5m tdmpc2:317m --format json
    """
    from worldflux.factory import get_config

    try:
        cfg_a = get_config(model_a)
        cfg_b = get_config(model_b)
    except (ValueError, Exception) as exc:
        console.print(f"[wf.fail]Error:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    fields_a = {f["name"]: f for f in _collect_fields(cfg_a)}
    fields_b = {f["name"]: f for f in _collect_fields(cfg_b)}

    all_keys = sorted(set(fields_a) | set(fields_b))
    diffs: list[dict[str, str]] = []
    for key in all_keys:
        va = fields_a.get(key, {}).get("value", "<absent>")
        vb = fields_b.get(key, {}).get("value", "<absent>")
        if str(va) != str(vb):
            diffs.append({"field": key, model_a: str(va), model_b: str(vb)})

    if format == "json":
        typer.echo(json.dumps(diffs, indent=2, default=str))
        return

    if not diffs:
        console.print("[wf.pass]No differences found.[/wf.pass]")
        return

    table = Table(
        title=f"Config diff: {model_a} vs {model_b}",
        show_lines=False,
    )
    table.add_column("Field", style="wf.brand", no_wrap=True)
    table.add_column(model_a, style="wf.caution", no_wrap=True)
    table.add_column(model_b, style="wf.info", no_wrap=True)

    for d in diffs:
        table.add_row(d["field"], d[model_a], d[model_b])

    console.print(table)


# ---------------------------------------------------------------------------
# config validate
# ---------------------------------------------------------------------------


@config_app.command("validate")
def config_validate(
    config_path: str = typer.Argument(..., help="Path to a config JSON/YAML file."),
) -> None:
    """Validate a configuration file against its schema.

    [dim]Examples:[/dim]
      worldflux config validate ./my_config.json
    """
    path = Path(config_path)
    if not path.exists():
        console.print(f"[wf.fail]File not found:[/wf.fail] {config_path}")
        raise typer.Exit(code=1)

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[wf.fail]Invalid JSON at line {exc.lineno}:[/wf.fail] {exc.msg}")
        raise typer.Exit(code=1) from None

    model_type = data.get("model_type", "base")

    from worldflux.core.registry import ConfigRegistry

    config_cls = ConfigRegistry._registry.get(model_type)
    if config_cls is None:
        from worldflux.core.config import WorldModelConfig

        config_cls = WorldModelConfig

    try:
        config_cls.from_dict(data)
    except Exception as exc:
        console.print(f"[wf.fail]Validation failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    console.print(f"[wf.pass]Valid[/wf.pass] {path.name} ({config_cls.__name__})")


# ---------------------------------------------------------------------------
# config schema
# ---------------------------------------------------------------------------


@config_app.command("schema")
def config_schema(
    model: str = typer.Argument(..., help="Model preset ID (e.g. dreamerv3:size12m)."),
    format: str = typer.Option(
        "json-schema", "--format", "-f", help="Output format (json-schema)."
    ),
) -> None:
    """Generate JSON Schema for a model configuration.

    [dim]Examples:[/dim]
      worldflux config schema dreamerv3:size12m
      worldflux config schema tdmpc2:5m --format json-schema
    """
    from worldflux.factory import get_config

    try:
        cfg = get_config(model)
    except (ValueError, Exception) as exc:
        console.print(f"[wf.fail]Error:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    schema = type(cfg).json_schema()
    typer.echo(json.dumps(schema, indent=2, default=str))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_fields(cfg: Any) -> list[dict[str, str]]:
    """Extract field metadata from a dataclass config instance."""
    result: list[dict[str, str]] = []
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if isinstance(value, Enum):
            value = value.value
        default = f.default
        if default is dataclasses.MISSING:
            if f.default_factory is not dataclasses.MISSING:  # type: ignore[arg-type]
                try:
                    default = f.default_factory()  # type: ignore[misc]
                except Exception:
                    default = "..."
            else:
                default = "<required>"
        elif isinstance(default, Enum):
            default = default.value

        type_name = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
        result.append(
            {
                "name": f.name,
                "type": str(type_name),
                "value": str(value),
                "default": str(default),
            }
        )
    return result
