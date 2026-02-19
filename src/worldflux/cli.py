"""Typer-based CLI for WorldFlux project scaffolding."""

from __future__ import annotations

import glob
import importlib.util
import json
import shutil
import subprocess
import sys
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

from worldflux.bootstrap_env import ensure_init_dependencies
from worldflux.parity import (
    CampaignRunOptions,
    aggregate_runs,
    export_campaign_source,
    load_campaign_spec,
    parse_seed_csv,
    render_campaign_summary,
    render_markdown_report,
    run_campaign,
    run_suite,
)
from worldflux.parity.errors import ParityError
from worldflux.scaffold import generate_project

app = typer.Typer(help="WorldFlux command-line interface.")
parity_app = typer.Typer(
    help="Run parity tools (legacy harness + proof-grade official equivalence pipeline)."
)
parity_campaign_app = typer.Typer(help="Run reproducible parity campaigns.")
app.add_typer(parity_app, name="parity")
parity_app.add_typer(parity_campaign_app, name="campaign")
console = Console()
ATARI_OPTIONAL_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("gymnasium", "gymnasium"),
    ("ale_py", "ale-py"),
)

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
DEFAULT_TOTAL_STEPS = 100000
DEFAULT_BATCH_SIZE = 16
OBS_ACTION_GUIDE_URL = "https://worldflux.ai/reference/observation-action/"
MODEL_CHOICE_IDS: tuple[str, str] = ("dreamer:ci", "tdmpc2:ci")
MODEL_UI_CARDS: dict[str, dict[str, str]] = {
    "dreamer:ci": {
        "display_name": "Dreamer (CI preset)",
        "best_for": "pixel and spatial observations",
        "observation_fit": "image-like inputs such as 3,64,64",
        "compute_profile": "heavier",
        "tradeoff_note": "better when latent imagination quality matters more than speed",
    },
    "tdmpc2:ci": {
        "display_name": "TD-MPC2 (CI preset)",
        "best_for": "compact vector observations",
        "observation_fit": "state vectors such as 39",
        "compute_profile": "lighter",
        "tradeoff_note": "better when you want a faster vector-control baseline",
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


def _parse_positive_int(value: str, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer. Got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive. Got {parsed}.")
    return parsed


def _resolve_model(environment: str, obs_shape: list[int]) -> tuple[str, str]:
    if environment == "atari":
        return "dreamer:ci", "dreamer"
    if environment == "mujoco":
        return "tdmpc2:ci", "tdmpc2"
    if len(obs_shape) >= 3:
        return "dreamer:ci", "dreamer"
    return "tdmpc2:ci", "tdmpc2"


def _model_type_from_model_id(model_id: str) -> str:
    if model_id.startswith("dreamer:"):
        return "dreamer"
    if model_id.startswith("tdmpc2:"):
        return "tdmpc2"
    raise ValueError(f"Unsupported model id: {model_id!r}")


def _model_choice_order(recommended_model: str) -> list[str]:
    if recommended_model not in MODEL_CHOICE_IDS:
        return list(MODEL_CHOICE_IDS)
    return [
        recommended_model,
        *[model_id for model_id in MODEL_CHOICE_IDS if model_id != recommended_model],
    ]


def _format_model_choice_label(model_id: str, *, recommended: bool) -> str:
    card = MODEL_UI_CARDS[model_id]
    recommendation = " (recommended)" if recommended else ""
    return (
        f"{model_id}{recommendation} - {card['display_name']} "
        f"(best for: {card['best_for']}, compute: {card['compute_profile']})"
    )


def _print_model_choices(recommended_model: str) -> None:
    lines: list[str] = []
    for model_id in _model_choice_order(recommended_model):
        card = MODEL_UI_CARDS[model_id]
        suffix = " [recommended]" if model_id == recommended_model else ""
        lines.extend(
            [
                f"[bold]{model_id}{suffix}[/bold] - {card['display_name']}",
                f"Best for: {card['best_for']}",
                f"Observation fit: {card['observation_fit']}",
                f"Compute profile: {card['compute_profile']}",
                f"Tradeoff: {card['tradeoff_note']}",
                "",
            ]
        )
    console.print(
        Panel.fit(
            "\n".join(lines).strip(),
            title="Model Choices",
            border_style="cyan",
        )
    )


def _select_model_with_inquirer(inquirer: Any, recommended_model: str) -> str:
    try:
        return str(
            inquirer.select(
                message="Choose model (recommended is preselected):",
                choices=[
                    {
                        "name": _format_model_choice_label(
                            model_id, recommended=model_id == recommended_model
                        ),
                        "value": model_id,
                    }
                    for model_id in _model_choice_order(recommended_model)
                ],
                default=recommended_model,
            ).execute()
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        console.print(
            f"[yellow]Model selection prompt failed ({exc}). "
            f"Using recommended model: {recommended_model}[/yellow]"
        )
        return recommended_model


def _select_model_with_rich(recommended_model: str) -> str:
    try:
        return str(
            Prompt.ask(
                "Choose model (recommended is preselected)",
                choices=_model_choice_order(recommended_model),
                default=recommended_model,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        console.print(
            f"[yellow]Model selection prompt failed ({exc}). "
            f"Using recommended model: {recommended_model}[/yellow]"
        )
        return recommended_model


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
    other_model = next(candidate for candidate in MODEL_CHOICE_IDS if candidate != model)
    if environment == "atari":
        why_now = (
            "This environment is image-based, and dreamer:ci is usually the safer first choice "
            "for pixel observations."
        )
    elif environment == "mujoco":
        why_now = (
            "This environment is vector-based, and tdmpc2:ci is usually the safer first choice "
            "for compact state inputs."
        )
    elif len(obs_shape) >= 3:
        why_now = (
            "Your observation has 3+ dimensions, which is usually spatial or image-like, "
            "so dreamer:ci is recommended."
        )
    else:
        why_now = "Your observation is a compact vector, so tdmpc2:ci is recommended."
    if model == "dreamer:ci":
        when_other = (
            f"Choose {other_model} when your task is mostly vector observations "
            "and you want a lighter setup."
        )
    else:
        when_other = (
            f"Choose {other_model} when your task is image-heavy "
            "and latent imagination quality is the priority."
        )
    console.print(f"[bold green]Recommended model:[/bold green] {model}")
    console.print(
        Panel.fit(
            f"[bold]Recommended model:[/bold] {model}\n"
            f"[bold]Why recommended now:[/bold] {why_now}\n"
            f"[bold]When to choose the other model:[/bold] {when_other}",
            title="Recommendation",
            border_style="magenta",
        )
    )


def _print_configuration_summary(context: dict[str, Any], target_path: Path, force: bool) -> None:
    model_fit = MODEL_UI_CARDS.get(str(context["model"]), {}).get("best_for", "n/a")
    summary = "\n".join(
        [
            f"[bold]Project name:[/bold] {context['project_name']}",
            f"[bold]Environment:[/bold] {context['environment']}",
            f"[bold]Observation shape:[/bold] {tuple(context['obs_shape'])}",
            f"[bold]Action dim:[/bold] {context['action_dim']}",
            f"[bold]Shape/Action guide:[/bold] {OBS_ACTION_GUIDE_URL}",
            f"[bold]Total steps:[/bold] {context['training_total_steps']}",
            f"[bold]Batch size:[/bold] {context['training_batch_size']}",
            f"[bold]Model:[/bold] {context['model']}",
            f"[bold]Model fit:[/bold] {model_fit}",
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

    console.print(
        f"[dim]Need help with Observation shape or Action dim? {OBS_ACTION_GUIDE_URL}[/dim]"
    )

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

    recommended_model, _ = _resolve_model(environment, obs_shape)
    _print_model_recommendation(environment, obs_shape, recommended_model)
    _print_model_choices(recommended_model)
    model = _select_model_with_inquirer(inquirer, recommended_model)
    model_type = _model_type_from_model_id(model)

    while True:
        total_steps_value = (
            inquirer.text(
                message=(
                    f"Total training steps (recommended {DEFAULT_TOTAL_STEPS:,}, positive integer):"
                ),
                default=str(DEFAULT_TOTAL_STEPS),
            )
            .execute()
            .strip()
        )
        try:
            training_total_steps = _parse_positive_int(
                total_steps_value,
                field_name="Total training steps",
            )
            break
        except ValueError as exc:
            console.print(f"[bold red]Invalid total steps:[/bold red] {exc}")

    while True:
        batch_size_value = (
            inquirer.text(
                message=(f"Batch size (recommended {DEFAULT_BATCH_SIZE}, positive integer):"),
                default=str(DEFAULT_BATCH_SIZE),
            )
            .execute()
            .strip()
        )
        try:
            training_batch_size = _parse_positive_int(
                batch_size_value,
                field_name="Batch size",
            )
            break
        except ValueError as exc:
            console.print(f"[bold red]Invalid batch size:[/bold red] {exc}")

    use_gpu = bool(
        inquirer.confirm(
            message="Prefer GPU for training? (recommended when CUDA is available)",
            default=torch.cuda.is_available(),
        ).execute()
    )

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
        "training_total_steps": training_total_steps,
        "training_batch_size": training_batch_size,
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

    console.print(
        f"[dim]Need help with Observation shape or Action dim? {OBS_ACTION_GUIDE_URL}[/dim]"
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

    recommended_model, _ = _resolve_model(environment, obs_shape)
    _print_model_recommendation(environment, obs_shape, recommended_model)
    _print_model_choices(recommended_model)
    model = _select_model_with_rich(recommended_model)
    model_type = _model_type_from_model_id(model)

    while True:
        total_steps_value = str(
            Prompt.ask(
                f"Total training steps (recommended {DEFAULT_TOTAL_STEPS:,})",
                default=str(DEFAULT_TOTAL_STEPS),
            )
        ).strip()
        try:
            training_total_steps = _parse_positive_int(
                total_steps_value,
                field_name="Total training steps",
            )
            break
        except ValueError as exc:
            console.print(f"[bold red]Invalid total steps:[/bold red] {exc}")

    while True:
        batch_size_value = str(
            Prompt.ask(
                f"Batch size (recommended {DEFAULT_BATCH_SIZE})",
                default=str(DEFAULT_BATCH_SIZE),
            )
        ).strip()
        try:
            training_batch_size = _parse_positive_int(
                batch_size_value,
                field_name="Batch size",
            )
            break
        except ValueError as exc:
            console.print(f"[bold red]Invalid batch size:[/bold red] {exc}")

    use_gpu = Confirm.ask(
        "Prefer GPU for training? (recommended when CUDA is available)",
        default=torch.cuda.is_available(),
    )
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
        "training_total_steps": training_total_steps,
        "training_batch_size": training_batch_size,
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
    1) Current interpreter when running from `uv tool` (ensures `worldflux` import works)
    2) python
    3) python3
    4) py (Windows launcher)
    5) uv run python
    6) current interpreter path
    """
    normalized_executable = sys.executable.replace("\\", "/")
    if "/uv/tools/" in normalized_executable:
        return sys.executable

    for candidate in ("python", "python3", "py"):
        if shutil.which(candidate):
            return candidate
    if shutil.which("uv"):
        return "uv run python"
    return sys.executable


def _missing_atari_dependency_packages() -> list[str]:
    missing: list[str] = []
    for module_name, package_name in ATARI_OPTIONAL_DEPENDENCIES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def _is_interactive_terminal() -> bool:
    stdin = getattr(sys, "stdin", None)
    stdout = getattr(sys, "stdout", None)
    return bool(stdin and stdout and stdin.isatty() and stdout.isatty())


def _confirm_optional_dependency_install(packages: list[str]) -> bool:
    package_list = ", ".join(packages)
    message = f"Install optional Atari dependencies now? ({package_list})"
    try:
        from InquirerPy import inquirer
    except ModuleNotFoundError:
        return bool(Confirm.ask(message, default=True))
    return bool(inquirer.confirm(message=message, default=True).execute())


def _install_packages_with_pip(packages: list[str]) -> bool:
    if not packages:
        return True

    cmd = [sys.executable, "-m", "pip", "install", *packages]
    console.print(
        f"[cyan]Installing optional dependencies:[/cyan] [bold]{', '.join(packages)}[/bold]"
    )
    completed = subprocess.run(cmd, check=False)
    if completed.returncode == 0:
        return True

    if shutil.which("uv") is None:
        return False

    console.print("[yellow]pip is unavailable. Retrying with uv pip.[/yellow]")
    uv_python = sys.executable

    uv_install = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            uv_python,
            "--break-system-packages",
            *packages,
        ],
        check=False,
    )
    return uv_install.returncode == 0


def _handle_optional_atari_dependency_install(context: dict[str, Any]) -> None:
    if str(context.get("environment", "")).strip().lower() != "atari":
        return

    missing_packages = _missing_atari_dependency_packages()
    if not missing_packages:
        return

    install_command = f"{sys.executable} -m pip install {' '.join(missing_packages)}"
    uv_install_command = (
        f'uv pip install --python "{sys.executable}" '
        "--break-system-packages " + " ".join(missing_packages)
    )
    console.print(
        f"[yellow]Optional Atari dependencies are missing:[/yellow] {', '.join(missing_packages)}"
    )
    console.print("[dim]Without them, live gameplay falls back to random replay data.[/dim]")

    if not _is_interactive_terminal():
        console.print(f"[dim]Install later with: {install_command}[/dim]")
        if shutil.which("uv"):
            console.print(f"[dim]Or with uv: {uv_install_command}[/dim]")
        return

    if not _confirm_optional_dependency_install(missing_packages):
        console.print(f"[yellow]Skipped install.[/yellow] You can run: {install_command}")
        return

    if _install_packages_with_pip(missing_packages):
        console.print("[green]Optional Atari dependencies installed successfully.[/green]")
    else:
        console.print("[yellow]Install failed. Training can continue in fallback mode.[/yellow]")
        console.print(f"[dim]Retry with: {install_command}[/dim]")
        if shutil.which("uv"):
            console.print(f"[dim]Or with uv: {uv_install_command}[/dim]")


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
        console.print("[cyan]Preparing bootstrap dependencies before project generation...[/cyan]")

        def _bootstrap_progress(message: str) -> None:
            console.print(f"[dim][bootstrap][/dim] {message}")

        deps_result = ensure_init_dependencies(context, progress_callback=_bootstrap_progress)
        if deps_result.skipped:
            console.print(f"[yellow]{deps_result.message}[/yellow]")
        elif deps_result.success:
            console.print(f"[green]{deps_result.message}[/green]")
            if deps_result.launcher:
                context["preferred_python_launcher"] = deps_result.launcher
        else:
            console.print(
                f"[bold red]Dependency bootstrap failed:[/bold red] {deps_result.message}"
            )
            for line in deps_result.diagnostics:
                console.print(f"[dim]{line}[/dim]")
            if deps_result.retry_commands:
                console.print("[yellow]Retry with:[/yellow]")
                for command in deps_result.retry_commands:
                    console.print(f"[dim]{command}[/dim]")
            raise typer.Exit(code=1)

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
    launcher = str(context.get("preferred_python_launcher") or _resolve_python_launcher())
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


def _resolve_parity_script_path(script_name: str) -> Path:
    candidates = (
        Path(__file__).resolve().parents[2] / "scripts" / "parity" / script_name,
        Path.cwd().resolve() / "scripts" / "parity" / script_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ParityError(
        f"Unable to locate scripts/parity/{script_name}. "
        "Run this command from a WorldFlux source checkout."
    )


def _run_parity_proof_script(script_name: str, args: list[str]) -> str:
    script_path = _resolve_parity_script_path(script_name)
    completed = subprocess.run(
        [sys.executable, str(script_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        details = "\n".join(part for part in (stdout, stderr) if part)
        raise ParityError(
            f"Proof pipeline step failed ({script_name}, exit={completed.returncode}).\n{details}"
        )
    return stdout


def _fmt_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return "-"


@parity_app.command("run")
def parity_run(
    suite: Path = typer.Argument(..., help="Path to parity suite specification file."),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output path for normalized run artifact.",
    ),
    candidate: Path | None = typer.Option(
        None,
        "--candidate",
        help="Override worldflux source path defined in suite file.",
    ),
    candidate_format: str | None = typer.Option(
        None,
        "--candidate-format",
        help="Override worldflux source format defined in suite file.",
    ),
    oracle: Path | None = typer.Option(
        None,
        "--oracle",
        help="Override upstream/oracle source path defined in suite file.",
    ),
    oracle_format: str | None = typer.Option(
        None,
        "--oracle-format",
        help="Override upstream/oracle source format defined in suite file.",
    ),
    upstream_lock: Path | None = typer.Option(
        None,
        "--upstream-lock",
        help="Override upstream lock path used for suite_lock_ref metadata.",
    ),
    enforce: bool = typer.Option(
        False,
        "--enforce/--no-enforce",
        help="Exit non-zero when non-inferiority verdict fails.",
    ),
) -> None:
    """Run legacy non-inferiority parity harness and emit comparison artifact."""
    try:
        payload = run_suite(
            suite,
            output_path=output,
            upstream_path=oracle,
            upstream_format=oracle_format,
            worldflux_path=candidate,
            worldflux_format=candidate_format,
            upstream_lock_path=upstream_lock,
        )
    except (ParityError, ValueError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[bold red]Parity run failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    suite_meta = payload["suite"]
    stats = payload["stats"]
    passed = bool(stats["pass_non_inferiority"])
    verdict = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "Mode: legacy non-inferiority harness",
                    f"Suite: {suite_meta['suite_id']} ({suite_meta['family']})",
                    f"Samples: {stats['sample_size']}",
                    f"Mean drop ratio: {stats['mean_drop_ratio']:.6f}",
                    f"Upper CI (one-sided): {stats['ci_upper_ratio']:.6f}",
                    f"Margin: {stats['margin_ratio']:.6f}",
                    f"Verdict: {verdict}",
                ]
            ),
            title="Parity Run (Legacy)",
            border_style="cyan",
        )
    )
    if enforce and not passed:
        raise typer.Exit(code=1)


@parity_app.command("aggregate")
def parity_aggregate(
    run_paths: list[Path] = typer.Option(
        [],
        "--run",
        help="Run artifact path (repeat to pass multiple files).",
    ),
    runs_glob: str = typer.Option(
        "reports/parity/runs/*.json",
        "--runs-glob",
        help="Glob used when --run is omitted.",
    ),
    output: Path = typer.Option(
        Path("reports/parity/aggregate.json"),
        "--output",
        help="Output path for aggregate artifact.",
    ),
    enforce: bool = typer.Option(
        False,
        "--enforce/--no-enforce",
        help="Exit non-zero when any suite fails aggregate verdict.",
    ),
) -> None:
    """Aggregate legacy parity run artifacts."""
    paths = list(run_paths)
    if not paths:
        paths = [Path(path) for path in sorted(glob.glob(runs_glob))]
    if not paths:
        console.print("[bold red]No run artifacts found to aggregate.[/bold red]")
        raise typer.Exit(code=1)

    try:
        payload = aggregate_runs(paths, output_path=output)
    except (ParityError, ValueError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[bold red]Parity aggregate failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    passed = bool(payload["all_suites_pass"])
    verdict = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "Mode: legacy non-inferiority harness",
                    f"Runs: {payload['run_count']}",
                    f"Suite pass: {payload['suite_pass_count']}",
                    f"Suite fail: {payload['suite_fail_count']}",
                    f"Verdict: {verdict}",
                    f"Written: {output.resolve()}",
                ]
            ),
            title="Parity Aggregate (Legacy)",
            border_style="blue",
        )
    )
    if enforce and not passed:
        raise typer.Exit(code=1)


@parity_app.command("report")
def parity_report(
    aggregate: Path = typer.Option(
        Path("reports/parity/aggregate.json"),
        "--aggregate",
        help="Aggregate parity artifact path.",
    ),
    output: Path = typer.Option(
        Path("reports/parity/report.md"),
        "--output",
        help="Markdown report output path.",
    ),
) -> None:
    """Render legacy parity markdown report from aggregate artifact."""
    try:
        payload = json.loads(aggregate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ParityError("aggregate payload must be a JSON object")
        markdown = render_markdown_report(payload)
    except (ParityError, ValueError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[bold red]Parity report failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    console.print(f"[green]Wrote parity report:[/green] {output.resolve()}")


@parity_app.command("proof-run")
def parity_proof_run(
    manifest: Path = typer.Argument(
        ..., help="Proof manifest (parity.manifest.v1 or parity.suite.v2)."
    ),
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="Run identifier. If omitted, scripts/parity/run_parity_matrix.py generates one.",
    ),
    output_dir: Path = typer.Option(
        Path("reports/parity"),
        "--output-dir",
        help="Output directory root for proof artifacts.",
    ),
    device: str = typer.Option("cuda", "--device", help="Execution device."),
    seed_list: str = typer.Option("", "--seed-list", help="Optional seed override, e.g. 0,1,2."),
    max_retries: int = typer.Option(1, "--max-retries", help="Max retries per task/system pair."),
    task_filter: str = typer.Option(
        "",
        "--task-filter",
        help="Comma-separated task filters (supports fnmatch patterns).",
    ),
    shard_index: int = typer.Option(0, "--shard-index", help="Shard index (0-based)."),
    num_shards: int = typer.Option(1, "--num-shards", help="Total shard count."),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from existing parity_runs.jsonl."
    ),
) -> None:
    """Run proof-grade parity matrix execution (official path backed by scripts/parity)."""
    args = [
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--max-retries",
        str(max_retries),
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
    ]
    if run_id.strip():
        args.extend(["--run-id", run_id.strip()])
    if seed_list.strip():
        args.extend(["--seed-list", seed_list.strip()])
    if task_filter.strip():
        args.extend(["--task-filter", task_filter.strip()])
    if resume:
        args.append("--resume")

    try:
        stdout = _run_parity_proof_script("run_parity_matrix.py", args)
    except ParityError as exc:
        console.print(f"[bold red]Parity proof-run failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    if stdout:
        console.print(stdout)
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "Mode: proof-grade official equivalence path",
                    f"Manifest: {manifest.resolve()}",
                    f"Output dir: {output_dir.resolve()}",
                    "Next: run `worldflux parity proof-report --manifest ... --runs .../parity_runs.jsonl`",
                ]
            ),
            title="Parity Proof Run",
            border_style="green",
        )
    )


@parity_app.command("proof-report")
def parity_proof_report(
    manifest: Path = typer.Argument(..., help="Proof manifest used for the run."),
    runs: Path = typer.Option(
        ...,
        "--runs",
        help="Path to parity_runs.jsonl produced by proof-run or distributed orchestration.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory for coverage/equivalence reports (defaults to runs parent).",
    ),
) -> None:
    """Generate proof-grade completeness + equivalence + markdown reports."""
    resolved_runs = runs.resolve()
    run_root = resolved_runs.parent
    report_root = output_dir.resolve() if output_dir is not None else run_root
    report_root.mkdir(parents=True, exist_ok=True)

    coverage_report = report_root / "coverage_report.json"
    validity_report = report_root / "validity_report.json"
    equivalence_report = report_root / "equivalence_report.json"
    markdown_report = report_root / "equivalence_report.md"

    seed_plan = run_root / "seed_plan.json"
    run_context = run_root / "run_context.json"
    try:
        coverage_args = [
            "--manifest",
            str(manifest),
            "--runs",
            str(resolved_runs),
            "--output",
            str(coverage_report),
            "--max-missing-pairs",
            "0",
        ]
        if seed_plan.exists():
            coverage_args.extend(["--seed-plan", str(seed_plan)])
        if run_context.exists():
            coverage_args.extend(["--run-context", str(run_context)])
        _run_parity_proof_script("validate_matrix_completeness.py", coverage_args)

        _run_parity_proof_script(
            "stats_equivalence.py",
            [
                "--input",
                str(resolved_runs),
                "--output",
                str(equivalence_report),
                "--manifest",
                str(manifest),
                "--strict-completeness",
                "--strict-validity",
                "--proof-mode",
                "--validity-report",
                str(validity_report),
            ],
        )
        _run_parity_proof_script(
            "report_markdown.py",
            [
                "--input",
                str(equivalence_report),
                "--output",
                str(markdown_report),
            ],
        )
    except ParityError as exc:
        console.print(f"[bold red]Parity proof-report failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    report_payload = json.loads(equivalence_report.read_text(encoding="utf-8"))
    global_block = report_payload.get("global", {})
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "Mode: proof-grade official equivalence path",
                    f"Final verdict: {_fmt_bool(global_block.get('parity_pass_final'))}",
                    f"Validity pass: {_fmt_bool(global_block.get('validity_pass'))}",
                    f"Missing pairs: {global_block.get('missing_pairs', '-')}",
                    f"JSON: {equivalence_report}",
                    f"Markdown: {markdown_report}",
                ]
            ),
            title="Parity Proof Report",
            border_style="green",
        )
    )


def _resolve_campaign_seeds(
    spec_default: tuple[int, ...], seeds_option: str | None
) -> tuple[int, ...]:
    parsed = parse_seed_csv(seeds_option)
    if parsed:
        return parsed
    if spec_default:
        return spec_default
    raise ParityError("No seeds provided. Pass --seeds or define campaign.default_seeds.")


@parity_campaign_app.command("run")
def parity_campaign_run(
    campaign: Path = typer.Argument(..., help="Path to campaign specification file."),
    mode: str = typer.Option(
        "worldflux",
        "--mode",
        help="Execution mode: worldflux | oracle | both.",
    ),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seeds (e.g. 0,1,2). Uses campaign.default_seeds when omitted.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device string propagated to command template placeholders.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Override worldflux output path.",
    ),
    oracle_output: Path | None = typer.Option(
        None,
        "--oracle-output",
        help="Override oracle output path.",
    ),
    workdir: Path = typer.Option(
        Path.cwd(),
        "--workdir",
        help="Working directory for command template execution.",
    ),
    pair_output_root: Path | None = typer.Option(
        None,
        "--pair-output-root",
        help="Directory for per-task/seed temporary command outputs.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse existing canonical outputs and skip already available task/seed pairs.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Print command templates without executing them.",
    ),
) -> None:
    """Run parity campaign and emit canonical artifacts."""
    try:
        spec = load_campaign_spec(campaign)
        run_options = CampaignRunOptions(
            mode=mode,
            seeds=_resolve_campaign_seeds(spec.default_seeds, seeds),
            device=device,
            output=output.resolve() if output is not None else None,
            oracle_output=oracle_output.resolve() if oracle_output is not None else None,
            resume=resume,
            dry_run=dry_run,
            workdir=workdir.resolve(),
            pair_output_root=pair_output_root.resolve() if pair_output_root is not None else None,
        )
        summary = run_campaign(spec, run_options)
    except (ParityError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        console.print(f"[bold red]Parity campaign failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    console.print(
        Panel.fit(
            render_campaign_summary(summary),
            title="Parity Campaign",
            border_style="cyan",
        )
    )


@parity_campaign_app.command("resume")
def parity_campaign_resume(
    campaign: Path = typer.Argument(..., help="Path to campaign specification file."),
    mode: str = typer.Option(
        "worldflux",
        "--mode",
        help="Execution mode: worldflux | oracle | both.",
    ),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seeds (e.g. 0,1,2). Uses campaign.default_seeds when omitted.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device string propagated to command template placeholders.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Override worldflux output path.",
    ),
    oracle_output: Path | None = typer.Option(
        None,
        "--oracle-output",
        help="Override oracle output path.",
    ),
    workdir: Path = typer.Option(
        Path.cwd(),
        "--workdir",
        help="Working directory for command template execution.",
    ),
    pair_output_root: Path | None = typer.Option(
        None,
        "--pair-output-root",
        help="Directory for per-task/seed temporary command outputs.",
    ),
) -> None:
    """Resume parity campaign generation from existing outputs."""
    parity_campaign_run(
        campaign=campaign,
        mode=mode,
        seeds=seeds,
        device=device,
        output=output,
        oracle_output=oracle_output,
        workdir=workdir,
        pair_output_root=pair_output_root,
        resume=True,
        dry_run=False,
    )


@parity_campaign_app.command("export")
def parity_campaign_export(
    campaign: Path = typer.Argument(..., help="Path to campaign specification file."),
    source: str = typer.Option(
        "worldflux",
        "--source",
        help="Source to export: worldflux | oracle.",
    ),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seeds (e.g. 0,1,2). Uses campaign.default_seeds when omitted.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Override output path for exported canonical artifact.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse existing canonical output rows when output already exists.",
    ),
) -> None:
    """Export canonical artifact from campaign input source without command execution."""
    source_normalized = source.strip().lower()
    if source_normalized not in {"worldflux", "oracle"}:
        console.print("[bold red]--source must be one of: worldflux, oracle[/bold red]")
        raise typer.Exit(code=1)

    try:
        spec = load_campaign_spec(campaign)
        resolved_seeds = _resolve_campaign_seeds(spec.default_seeds, seeds)
        summary = export_campaign_source(
            spec,
            source_name=source_normalized,
            seeds=resolved_seeds,
            output_path=output.resolve() if output is not None else None,
            resume=resume,
        )
    except (ParityError, ValueError, OSError) as exc:
        console.print(f"[bold red]Parity campaign export failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    console.print(
        Panel.fit(
            render_campaign_summary(summary),
            title="Parity Campaign Export",
            border_style="blue",
        )
    )
