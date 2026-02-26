"""The ``doctor`` command — system environment check."""

from __future__ import annotations

import importlib.util
import platform

import typer

from ._app import app, console
from ._rich_output import status_table


@app.command(rich_help_panel="Utilities")
def doctor() -> None:
    """Check system environment and dependencies for WorldFlux.

    [dim]Examples:[/dim]
      worldflux doctor
    """
    import torch

    from worldflux import __version__

    rows: list[tuple[str, str, str]] = [
        ("pass", "WorldFlux", __version__),
        ("pass", "Python", platform.python_version()),
        ("pass", "PyTorch", torch.__version__),
    ]

    # CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda or "n/a"
        rows.append(("pass", "CUDA", f"{cuda_version} ({gpu_name})"))
    else:
        rows.append(("warn", "CUDA", "[wf.caution]not available[/wf.caution]"))

    rows.append(("pass", "Platform", platform.platform()))

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

    extras_text = ", ".join(installed) if installed else "[wf.muted]none[/wf.muted]"
    rows.append(("pass", "Installed extras", extras_text))
    if missing:
        rows.append(("warn", "Missing extras", "[wf.muted]" + ", ".join(missing) + "[/wf.muted]"))

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
            rows.append(("pass", "Model registry", ", ".join(parts)))
        else:
            rows.append(("pass", "Model registry", f"{len(all_models)} models"))
    except Exception:  # pragma: no cover
        rows.append(("warn", "Model registry", "[wf.muted]unavailable[/wf.muted]"))

    console.print(status_table(rows, title="WorldFlux Environment"))
    if missing:
        console.print(
            f"\n[wf.muted]Tip: Install missing extras with[/wf.muted]  "
            f"uv pip install 'worldflux[{','.join(missing)}]'"
        )
    raise typer.Exit(code=0)
