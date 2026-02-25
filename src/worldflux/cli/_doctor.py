"""The ``doctor`` command — system environment check."""

from __future__ import annotations

import importlib.util
import platform

import typer
from rich.table import Table

from ._app import app, console


@app.command(rich_help_panel="Utilities")
def doctor() -> None:
    """Check system environment and dependencies for WorldFlux.

    [dim]Examples:[/dim]
      worldflux doctor
    """
    import torch

    from worldflux import __version__

    ok = "[green]✓[/green]"
    warn = "[yellow]![/yellow]"

    table = Table(title="WorldFlux Environment", show_lines=False, padding=(0, 2))
    table.add_column("", width=3, no_wrap=True)  # status indicator
    table.add_column("Component", style="bold", no_wrap=True)
    table.add_column("Value")

    # Core info
    table.add_row(ok, "WorldFlux", __version__)
    table.add_row(ok, "Python", platform.python_version())
    table.add_row(ok, "PyTorch", torch.__version__)

    # CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda or "n/a"
        table.add_row(ok, "CUDA", f"{cuda_version} ({gpu_name})")
    else:
        table.add_row(warn, "CUDA", "[yellow]not available[/yellow]")

    table.add_row(ok, "Platform", platform.platform())

    # Optional extras
    _optional = [
        ("gymnasium", "gymnasium"),
        ("ale_py", "ale-py"),
        ("matplotlib", "matplotlib"),
        ("wandb", "wandb"),
        ("InquirerPy", "InquirerPy"),
        ("tqdm", "tqdm"),
    ]
    installed: list[str] = []
    missing: list[str] = []
    for module_name, display_name in _optional:
        if importlib.util.find_spec(module_name) is not None:
            installed.append(display_name)
        else:
            missing.append(display_name)

    extras_text = ", ".join(installed) if installed else "[dim]none[/dim]"
    table.add_row(ok, "Installed extras", extras_text)
    if missing:
        table.add_row(warn, "Missing extras", "[dim]" + ", ".join(missing) + "[/dim]")

    # Model registry
    try:
        from worldflux.factory import list_models

        all_models = list_models(verbose=True)
        if isinstance(all_models, dict):
            total = len(all_models)
            by_maturity: dict[str, int] = {}
            for info in all_models.values():
                mat = info.get("maturity", "unknown")
                by_maturity[mat] = by_maturity.get(mat, 0) + 1
            parts = [f"{total} total"]
            for mat, count in sorted(by_maturity.items()):
                parts.append(f"{mat}: {count}")
            table.add_row(ok, "Model registry", ", ".join(parts))
        else:
            table.add_row(ok, "Model registry", f"{len(all_models)} models")
    except Exception:  # pragma: no cover
        table.add_row(warn, "Model registry", "[dim]unavailable[/dim]")

    console.print(table)
    if missing:
        console.print(
            f"\n[dim]Tip: Install missing extras with[/dim]  "
            f"uv pip install 'worldflux[{','.join(missing)}]'"
        )
    raise typer.Exit(code=0)
