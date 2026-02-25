"""The ``init`` command and its helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import typer
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from ._app import (
    ASCII_LOGO,
    BATCH_SIZE_PRESETS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TOTAL_STEPS,
    ENVIRONMENT_OPTIONS,
    MODEL_CHOICE_IDS,
    MODEL_UI_CARDS,
    OBS_ACTION_GUIDE_URL,
    TOTAL_STEPS_PRESETS,
    app,
    console,
)
from ._utils import (
    _is_preset_environment,
    _model_choice_order,
    _model_type_from_model_id,
    _parse_action_dim,
    _parse_obs_shape,
    _parse_positive_int,
    _resolve_model,
)

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


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


def _numbered_select(
    label: str,
    options: Sequence[dict[str, Any]],
    *,
    default_index: int = 0,
) -> Any:
    """Rich fallback for radio-button style numbered selection."""
    console.print(f"[bold]{label}[/bold]")
    for i, opt in enumerate(options, 1):
        marker = "\u203a" if (i - 1) == default_index else " "
        console.print(f"  {marker} [cyan]\\[{i}][/cyan] {opt['name']}")
    choice = IntPrompt.ask(
        "Enter number",
        choices=[str(i) for i in range(1, len(options) + 1)],
        default=default_index + 1,
    )
    return options[int(choice) - 1]["value"]


def _select_model_with_rich(recommended_model: str) -> str:
    try:
        model_options = [
            {
                "name": _format_model_choice_label(
                    model_id, recommended=model_id == recommended_model
                ),
                "value": model_id,
            }
            for model_id in _model_choice_order(recommended_model)
        ]
        return str(_numbered_select("Choose model:", model_options, default_index=0))
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


# ---------------------------------------------------------------------------
# Prompt / confirmation helpers
# ---------------------------------------------------------------------------


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
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

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

    if _is_preset_environment(environment):
        obs_shape = _parse_obs_shape(str(ENVIRONMENT_OPTIONS[environment]["obs_shape"]))
        action_dim = ENVIRONMENT_OPTIONS[environment]["action_dim"]
        console.print(
            f"[dim]Using {environment} defaults: "
            f"obs_shape={tuple(obs_shape)}, action_dim={action_dim}[/dim]"
        )
    else:
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
    _cli._print_model_recommendation(environment, obs_shape, recommended_model)
    _cli._print_model_choices(recommended_model)
    model = _cli._select_model_with_inquirer(inquirer, recommended_model)
    model_type = _model_type_from_model_id(model)

    steps_choice = inquirer.select(
        message="Total training steps:",
        choices=[{"name": p["name"], "value": p["value"]} for p in TOTAL_STEPS_PRESETS],
        default=DEFAULT_TOTAL_STEPS,
    ).execute()
    if steps_choice == "custom":
        while True:
            total_steps_value = (
                inquirer.text(
                    message=(
                        f"Total training steps (recommended {DEFAULT_TOTAL_STEPS:,}, "
                        "positive integer):"
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
    else:
        training_total_steps = int(steps_choice)

    batch_choice = inquirer.select(
        message="Batch size:",
        choices=[{"name": p["name"], "value": p["value"]} for p in BATCH_SIZE_PRESETS],
        default=DEFAULT_BATCH_SIZE,
    ).execute()
    if batch_choice == "custom":
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
    else:
        training_batch_size = int(batch_choice)

    cuda_avail = torch.cuda.is_available()
    device_choices = [
        {
            "name": "GPU (CUDA)"
            + (" \u2014 detected" if cuda_avail else " \u2014 not available, falls back to CPU"),
            "value": "cuda",
        },
        {"name": "CPU", "value": "cpu"},
    ]
    device = inquirer.select(
        message="Training device:",
        choices=device_choices,
        default="cuda" if cuda_avail else "cpu",
    ).execute()
    if device == "cuda" and not cuda_avail:
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
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

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

    env_options = [
        {"name": _format_environment_choice(key), "value": key}
        for key in ("atari", "mujoco", "custom")
    ]
    environment = _cli._numbered_select(
        "Choose your environment type:", env_options, default_index=0
    )

    if _is_preset_environment(environment):
        obs_shape = _parse_obs_shape(str(ENVIRONMENT_OPTIONS[environment]["obs_shape"]))
        action_dim = ENVIRONMENT_OPTIONS[environment]["action_dim"]
        console.print(
            f"[dim]Using {environment} defaults: "
            f"obs_shape={tuple(obs_shape)}, action_dim={action_dim}[/dim]"
        )
    else:
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
    _cli._print_model_recommendation(environment, obs_shape, recommended_model)
    _cli._print_model_choices(recommended_model)
    model = _cli._select_model_with_rich(recommended_model)
    model_type = _model_type_from_model_id(model)

    steps_choice = _cli._numbered_select(
        "Total training steps:", list(TOTAL_STEPS_PRESETS), default_index=1
    )
    if steps_choice == "custom":
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
    else:
        training_total_steps = int(steps_choice)

    batch_choice = _cli._numbered_select("Batch size:", list(BATCH_SIZE_PRESETS), default_index=1)
    if batch_choice == "custom":
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
    else:
        training_batch_size = int(batch_choice)

    cuda_avail = torch.cuda.is_available()
    device_options = [
        {
            "name": "GPU (CUDA)"
            + (" \u2014 detected" if cuda_avail else " \u2014 not available, falls back to CPU"),
            "value": "cuda",
        },
        {"name": "CPU", "value": "cpu"},
    ]
    device = _cli._numbered_select(
        "Training device:",
        device_options,
        default_index=0 if cuda_avail else 1,
    )
    if device == "cuda" and not cuda_avail:
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
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    config = _cli._prompt_with_inquirer()
    if config is not None:
        return config
    return _cli._prompt_with_rich()


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


# ---------------------------------------------------------------------------
# Optional dependency installation
# ---------------------------------------------------------------------------


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
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    if str(context.get("environment", "")).strip().lower() != "atari":
        return

    missing_packages = _cli._missing_atari_dependency_packages()
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

    if not _cli._is_interactive_terminal():
        console.print(f"[dim]Install later with: {install_command}[/dim]")
        if shutil.which("uv"):
            console.print(f"[dim]Or with uv: {uv_install_command}[/dim]")
        return

    if not _cli._confirm_optional_dependency_install(missing_packages):
        console.print(f"[yellow]Skipped install.[/yellow] You can run: {install_command}")
        return

    if _cli._install_packages_with_pip(missing_packages):
        console.print("[green]Optional Atari dependencies installed successfully.[/green]")
    else:
        console.print("[yellow]Install failed. Training can continue in fallback mode.[/yellow]")
        console.print(f"[dim]Retry with: {install_command}[/dim]")
        if shutil.which("uv"):
            console.print(f"[dim]Or with uv: {uv_install_command}[/dim]")


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


@app.command(rich_help_panel="Getting Started")
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
    """Initialize a new WorldFlux training project.

    [dim]Examples:[/dim]
      worldflux init
      worldflux init ./my-project
      worldflux init ./my-project --force
    """
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    _cli._print_logo()

    try:
        context = _cli._prompt_user_configuration()
        console.print("[cyan]Preparing bootstrap dependencies before project generation...[/cyan]")

        def _bootstrap_progress(message: str) -> None:
            console.print(f"[dim][bootstrap][/dim] {message}")

        deps_result = _cli.ensure_init_dependencies(context, progress_callback=_bootstrap_progress)
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
        _cli._print_configuration_summary(context, target_path, force=force)
        if not _cli._confirm_generation():
            console.print("\n[yellow]Initialization cancelled. No files were generated.[/yellow]")
            raise typer.Exit(code=130)
        # Pop launcher before scaffold validation (templates use .get with default)
        preferred_launcher = context.pop("preferred_python_launcher", None)
        _cli.generate_project(target_path, context, force=force)
    except KeyboardInterrupt:
        console.print("\n[yellow]Initialization cancelled. No files were generated.[/yellow]")
        raise typer.Exit(code=130) from None
    except (ValueError, FileExistsError, IsADirectoryError, OSError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    resolved_target = target_path.resolve()
    console.print(f"\n[bold green]Project created:[/bold green] {resolved_target}")
    launcher = str(preferred_launcher or _cli._resolve_python_launcher())
    next_steps = "\n".join(
        [
            f"1. cd {resolved_target}",
            "2. worldflux train             # start training",
            f"3. {launcher} train.py   # start training (legacy path)",
            f"4. {launcher} inference.py   # run inference (legacy path)",
            "5. worldflux verify --target ./outputs  # verify your model",
            "6. Edit worldflux.toml to tune settings for your environment",
        ]
    )
    console.print(
        Panel.fit(
            next_steps,
            title="Next Steps",
            border_style="green",
        )
    )
