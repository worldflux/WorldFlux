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
        f"Missing required CLI dependency '{missing}'. Reinstall with: uv pip install -U worldflux"
    ) from exc

import torch

from worldflux.scaffold import generate_project

app = typer.Typer(help="WorldFlux command-line interface.")
console = Console()

ENVIRONMENT_OPTIONS = {
    "atari": {
        "label": "Atari (pixel observations)",
        "description": "Image-based control tasks such as Atari-like environments.",
        "recommended_model": "dreamer:ci",
        "obs_shape": "3,64,64",
        "action_dim": 6,
    },
    "mujoco": {
        "label": "MuJoCo (state vectors)",
        "description": "Continuous-control robotics tasks with vector observations.",
        "recommended_model": "tdmpc2:ci",
        "obs_shape": "39",
        "action_dim": 6,
    },
    "custom": {
        "label": "Custom",
        "description": "Bring your own environment shape; model is inferred from shape.",
        "recommended_model": "auto",
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


def _format_environment_choice(name: str) -> str:
    option = ENVIRONMENT_OPTIONS[name]
    recommendation = option["recommended_model"]
    if recommendation == "auto":
        recommendation_text = "recommended model: auto"
    else:
        recommendation_text = f"recommended model: {recommendation}"
    return f"{option['label']} - {option['description']} ({recommendation_text})"


def _print_environment_options() -> None:
    console.print("[bold]Choose your environment type:[/bold]")
    for key in ("atari", "mujoco", "custom"):
        option = ENVIRONMENT_OPTIONS[key]
        console.print(
            f"- [cyan]{key}[/cyan]: {option['description']} "
            f"(default obs: {option['obs_shape']}, default action dim: {option['action_dim']})"
        )


def _print_model_recommendation(
    environment: str,
    obs_shape: list[int],
    model: str,
) -> None:
    if environment == "atari":
        reason = (
            "Atari tasks usually use image observations. "
            "dreamer:ci is tuned for latent world models on pixels."
        )
    elif environment == "mujoco":
        reason = (
            "MuJoCo tasks usually use compact state vectors. "
            "tdmpc2:ci is tuned for vector-based control."
        )
    elif len(obs_shape) >= 3:
        reason = (
            "Your custom observation has 3+ dimensions, which usually means "
            "spatial/image-like input. dreamer:ci is recommended."
        )
    else:
        reason = (
            "Your custom observation is a compact vector. tdmpc2:ci is recommended for this setup."
        )
    console.print(f"[bold green]Recommended model:[/bold green] {model}")
    console.print(
        Panel.fit(
            f"[bold]Recommended model:[/bold] {model}\n[bold]Why:[/bold] {reason}",
            title="Recommendation",
            border_style="magenta",
        )
    )


def _print_configuration_summary(context: dict[str, Any], target_path: Path, force: bool) -> None:
    summary = "\n".join(
        [
            f"[bold]Project name:[/bold] {context['project_name']}",
            f"[bold]Environment:[/bold] {context['environment']}",
            f"[bold]Observation shape:[/bold] {tuple(context['obs_shape'])}",
            f"[bold]Action dim:[/bold] {context['action_dim']}",
            f"[bold]Model:[/bold] {context['model']}",
            f"[bold]Device:[/bold] {context['device']}",
            f"[bold]Target path:[/bold] {target_path.resolve()}",
            f"[bold]Overwrite existing files:[/bold] {'yes' if force else 'no'}",
        ]
    )
    console.print(
        Panel.fit(
            summary,
            title="Configuration Summary",
            border_style="blue",
        )
    )


def _confirm_generation() -> bool:
    try:
        from InquirerPy import inquirer
    except ModuleNotFoundError:
        return bool(Confirm.ask("Proceed and generate files?", default=True))
    return bool(
        inquirer.confirm(
            message="Proceed and generate files?",
            default=True,
        ).execute()
    )


def _prompt_with_inquirer() -> dict[str, Any] | None:
    try:
        from InquirerPy import inquirer
    except ModuleNotFoundError:
        return None

    project_name = (
        inquirer.text(
            message="Project name (folder name for generated files):",
            default="my-world-model",
            validate=lambda value: bool(str(value).strip()),
            invalid_message="Project name cannot be empty. Use a short folder-style name.",
        )
        .execute()
        .strip()
    )

    environment = inquirer.select(
        message="Environment type (what kind of observations your task provides):",
        choices=[
            {"name": _format_environment_choice("atari"), "value": "atari"},
            {"name": _format_environment_choice("mujoco"), "value": "mujoco"},
            {"name": _format_environment_choice("custom"), "value": "custom"},
        ],
        default="atari",
    ).execute()

    obs_default = ENVIRONMENT_OPTIONS[environment]["obs_shape"]
    while True:
        obs_value = (
            inquirer.text(
                message="Observation shape (comma-separated integers, e.g. 3,64,64 or 39):",
                default=obs_default,
            )
            .execute()
            .strip()
        )
        try:
            obs_shape = _parse_obs_shape(obs_value)
            break
        except ValueError as exc:
            console.print(
                f"[bold red]Invalid observation shape:[/bold red] {exc} "
                "Expected format like 3,64,64 or 39."
            )

    action_default = str(ENVIRONMENT_OPTIONS[environment]["action_dim"])
    while True:
        action_value = (
            inquirer.text(
                message="Action dimension (positive integer, e.g. 6):",
                default=action_default,
            )
            .execute()
            .strip()
        )
        try:
            action_dim = _parse_action_dim(action_value)
            break
        except ValueError as exc:
            console.print(
                f"[bold red]Invalid action dim:[/bold red] {exc} "
                "Expected a positive integer like 6."
            )

    use_gpu = bool(
        inquirer.confirm(
            message="Prefer GPU for training? (recommended when CUDA is available)",
            default=torch.cuda.is_available(),
        ).execute()
    )

    model, model_type = _resolve_model(environment, obs_shape)
    device = "cuda" if use_gpu else "cpu"
    if use_gpu and not torch.cuda.is_available():
        console.print("[yellow]CUDA is not available. Falling back to CPU.[/yellow]")
        device = "cpu"

    _print_model_recommendation(environment, obs_shape, model)

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
    project_name = str(
        Prompt.ask(
            "Project name (folder name for generated files)",
            default="my-world-model",
        )
    ).strip()
    while not project_name:
        console.print("[bold red]Project name cannot be empty.[/bold red]")
        project_name = str(
            Prompt.ask(
                "Project name (folder name for generated files)",
                default="my-world-model",
            )
        ).strip()

    _print_environment_options()
    environment = str(
        Prompt.ask(
            "Environment type",
            choices=["atari", "mujoco", "custom"],
            default="atari",
        )
    )

    obs_default = ENVIRONMENT_OPTIONS[environment]["obs_shape"]
    while True:
        obs_value = str(
            Prompt.ask(
                "Observation shape (comma-separated integers, e.g. 3,64,64 or 39)",
                default=obs_default,
            )
        ).strip()
        try:
            obs_shape = _parse_obs_shape(obs_value)
            break
        except ValueError as exc:
            console.print(
                f"[bold red]Invalid observation shape:[/bold red] {exc} "
                "Expected format like 3,64,64 or 39."
            )

    action_default = str(ENVIRONMENT_OPTIONS[environment]["action_dim"])
    while True:
        action_value = str(
            Prompt.ask(
                "Action dimension (positive integer, e.g. 6)",
                default=action_default,
            )
        ).strip()
        try:
            action_dim = _parse_action_dim(action_value)
            break
        except ValueError as exc:
            console.print(
                f"[bold red]Invalid action dim:[/bold red] {exc} "
                "Expected a positive integer like 6."
            )

    use_gpu = Confirm.ask(
        "Prefer GPU for training? (recommended when CUDA is available)",
        default=torch.cuda.is_available(),
    )
    model, model_type = _resolve_model(environment, obs_shape)
    device = "cuda" if use_gpu else "cpu"
    if use_gpu and not torch.cuda.is_available():
        console.print("[yellow]CUDA is not available. Falling back to CPU.[/yellow]")
        device = "cpu"

    _print_model_recommendation(environment, obs_shape, model)

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
    console.print(
        Panel.fit(
            "Create a ready-to-run WorldFlux project with train.py, inference.py, and "
            "worldflux.toml.\nThis guided setup explains each question and applies safe defaults.",
            title="Guided Setup",
            border_style="cyan",
        )
    )
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
        _print_configuration_summary(context, target_path, force=force)
        if not _confirm_generation():
            console.print("\n[yellow]Initialization cancelled. No files were generated.[/yellow]")
            raise typer.Exit(code=130)
        generate_project(target_path, context, force=force)
    except KeyboardInterrupt:
        console.print("\n[yellow]Initialization cancelled. No files were generated.[/yellow]")
        raise typer.Exit(code=130) from None
    except (ValueError, FileExistsError, IsADirectoryError, OSError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    resolved_target = target_path.resolve()
    console.print(f"\n[bold green]Project created:[/bold green] {resolved_target}")
    launcher = _resolve_python_launcher()
    next_steps = "\n".join(
        [
            f"1. cd {resolved_target}",
            f"2. {launcher} train.py   # start training",
            f"3. {launcher} inference.py   # run inference",
            "4. Edit worldflux.toml to tune settings for your environment",
        ]
    )
    console.print(
        Panel.fit(
            next_steps,
            title="Next Steps",
            border_style="green",
        )
    )
