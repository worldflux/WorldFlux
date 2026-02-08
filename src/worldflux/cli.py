"""Typer-based CLI for WorldFlux project scaffolding."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    missing = (exc.name or "cli dependency").split(".")[0]
    raise ModuleNotFoundError(
        f"Missing optional dependency '{missing}'. Install with: uv pip install -e '.[cli]'"
    ) from exc

import torch

from worldflux.scaffold import generate_project

app = typer.Typer(help="WorldFlux command-line interface.")
console = Console()

ENVIRONMENT_OPTIONS = {
    "atari": {
        "label": "🎮 Atari / Visual (Pixels)  -> recommended model: dreamer:ci",
        "obs_shape": "3,64,64",
        "action_dim": 6,
    },
    "mujoco": {
        "label": "🤖 MuJoCo / Robotics (States) -> recommended model: tdmpc2:ci",
        "obs_shape": "39",
        "action_dim": 6,
    },
    "custom": {
        "label": "🧪 Custom",
        "obs_shape": "39",
        "action_dim": 6,
    },
}

ASCII_LOGO = r"""
__        __         _     _ _____ _
\ \      / /__  _ __| | __| |  ___| |_   ___  __
 \ \ /\ / / _ \| '__| |/ _` | |_  | | | | \ \/ /
  \ V  V / (_) | |  | | (_| |  _| | | |_| |>  <
   \_/\_/ \___/|_|  |_|\__,_|_|   |_|\__,_/_/\_\
"""


@app.callback()
def main() -> None:
    """WorldFlux CLI commands."""


def _parse_obs_shape(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("Observation shape must contain at least one positive integer.")

    dims: list[int] = []
    for part in parts:
        try:
            dim = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid observation dimension: {part!r}") from exc
        if dim <= 0:
            raise ValueError(f"Observation dimensions must be positive. Got {dim}.")
        dims.append(dim)
    return dims


def _parse_action_dim(value: str) -> int:
    try:
        action_dim = int(value)
    except ValueError as exc:
        raise ValueError(f"Action dim must be an integer. Got {value!r}.") from exc
    if action_dim <= 0:
        raise ValueError(f"Action dim must be positive. Got {action_dim}.")
    return action_dim


def _resolve_model(environment: str, obs_shape: list[int]) -> tuple[str, str]:
    if environment == "atari":
        return "dreamer:ci", "dreamer"
    if environment == "mujoco":
        return "tdmpc2:ci", "tdmpc2"
    if len(obs_shape) >= 3:
        return "dreamer:ci", "dreamer"
    return "tdmpc2:ci", "tdmpc2"


def _prompt_with_inquirer() -> dict[str, Any] | None:
    try:
        from InquirerPy import inquirer
    except ModuleNotFoundError:
        return None

    project_name = (
        inquirer.text(
            message="Project Name:",
            default="my-world-model",
            validate=lambda value: bool(str(value).strip()),
            invalid_message="Project name cannot be empty.",
        )
        .execute()
        .strip()
    )

    environment = inquirer.select(
        message="Environment Type:",
        choices=[
            {"name": ENVIRONMENT_OPTIONS["atari"]["label"], "value": "atari"},
            {"name": ENVIRONMENT_OPTIONS["mujoco"]["label"], "value": "mujoco"},
            {"name": ENVIRONMENT_OPTIONS["custom"]["label"], "value": "custom"},
        ],
        default="atari",
    ).execute()

    obs_default = ENVIRONMENT_OPTIONS[environment]["obs_shape"]
    while True:
        obs_value = (
            inquirer.text(message="Observation Shape:", default=obs_default).execute().strip()
        )
        try:
            obs_shape = _parse_obs_shape(obs_value)
            break
        except ValueError as exc:
            console.print(f"[bold red]Invalid observation shape:[/bold red] {exc}")

    action_default = str(ENVIRONMENT_OPTIONS[environment]["action_dim"])
    while True:
        action_value = (
            inquirer.text(message="Action Dim:", default=action_default).execute().strip()
        )
        try:
            action_dim = _parse_action_dim(action_value)
            break
        except ValueError as exc:
            console.print(f"[bold red]Invalid action dim:[/bold red] {exc}")

    use_gpu = bool(
        inquirer.confirm(
            message="Use GPU?",
            default=torch.cuda.is_available(),
        ).execute()
    )

    model, model_type = _resolve_model(environment, obs_shape)
    device = "cuda" if use_gpu else "cpu"
    if use_gpu and not torch.cuda.is_available():
        console.print("[yellow]CUDA is not available. Falling back to CPU.[/yellow]")
        device = "cpu"

    return {
        "project_name": project_name,
        "environment": environment,
        "model": model,
        "model_type": model_type,
        "obs_shape": obs_shape,
        "action_dim": action_dim,
        "hidden_dim": 32,
        "device": device,
    }


def _prompt_with_rich() -> dict[str, Any]:
    project_name = Prompt.ask("Project Name", default="my-world-model").strip()
    while not project_name:
        console.print("[bold red]Project name cannot be empty.[/bold red]")
        project_name = Prompt.ask("Project Name", default="my-world-model").strip()

    console.print("Environment Type options: atari, mujoco, custom")
    environment = Prompt.ask(
        "Environment Type",
        choices=["atari", "mujoco", "custom"],
        default="atari",
    )

    obs_default = ENVIRONMENT_OPTIONS[environment]["obs_shape"]
    while True:
        obs_value = Prompt.ask("Observation Shape", default=obs_default).strip()
        try:
            obs_shape = _parse_obs_shape(obs_value)
            break
        except ValueError as exc:
            console.print(f"[bold red]Invalid observation shape:[/bold red] {exc}")

    action_default = str(ENVIRONMENT_OPTIONS[environment]["action_dim"])
    while True:
        action_value = Prompt.ask("Action Dim", default=action_default).strip()
        try:
            action_dim = _parse_action_dim(action_value)
            break
        except ValueError as exc:
            console.print(f"[bold red]Invalid action dim:[/bold red] {exc}")

    use_gpu = Confirm.ask("Use GPU?", default=torch.cuda.is_available())
    model, model_type = _resolve_model(environment, obs_shape)
    device = "cuda" if use_gpu else "cpu"
    if use_gpu and not torch.cuda.is_available():
        console.print("[yellow]CUDA is not available. Falling back to CPU.[/yellow]")
        device = "cpu"

    return {
        "project_name": project_name,
        "environment": environment,
        "model": model,
        "model_type": model_type,
        "obs_shape": obs_shape,
        "action_dim": action_dim,
        "hidden_dim": 32,
        "device": device,
    }


def _prompt_user_configuration() -> dict[str, Any]:
    config = _prompt_with_inquirer()
    if config is not None:
        return config
    return _prompt_with_rich()


def _print_logo() -> None:
    console.print(f"[bold cyan]{ASCII_LOGO}[/bold cyan]")
    console.print("[bold]WorldFlux CLI[/bold]  |  scaffold world-model projects")
    console.print()


def _resolve_python_launcher() -> str:
    """
    Resolve a runnable Python launcher command for the current environment.

    Preference order:
    1) uv run python (works even when `python` is not on PATH)
    2) python
    3) python3
    4) py (Windows launcher)
    """
    if shutil.which("uv"):
        return "uv run python"
    for candidate in ("python", "python3", "py"):
        if shutil.which(candidate):
            return candidate
    # Fallback for display only.
    return "python"


@app.command()
def init(
    path: Path | None = typer.Argument(
        None,
        help="Output directory. If omitted, ./<project-name> is created.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite generated files when target directory already exists.",
    ),
) -> None:
    """Initialize a new WorldFlux training project."""
    _print_logo()

    try:
        context = _prompt_user_configuration()

        if context["device"] == "cuda" and not torch.cuda.is_available():
            console.print("[yellow]CUDA is not available. Falling back to CPU.[/yellow]")
            context["device"] = "cpu"

        target_path = path if path is not None else Path.cwd() / str(context["project_name"])
        generate_project(target_path, context, force=force)
    except KeyboardInterrupt:
        console.print("\n[yellow]Initialization cancelled.[/yellow]")
        raise typer.Exit(code=130) from None
    except (ValueError, FileExistsError, IsADirectoryError, OSError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    resolved_target = target_path.resolve()
    console.print(f"\n[bold green]Project created:[/bold green] {resolved_target}")
    launcher = _resolve_python_launcher()
    next_steps = "\n".join(
        [
            f"cd {resolved_target}",
            f"{launcher} train.py",
            f"{launcher} inference.py",
        ]
    )
    console.print(
        Panel.fit(
            next_steps,
            title="Next Steps",
            border_style="green",
        )
    )
