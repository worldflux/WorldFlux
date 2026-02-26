"""The ``init`` command and its helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import typer
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule

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
from ._rich_output import key_value_panel, result_banner
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

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _format_model_choice_label(model_id: str, *, recommended: bool) -> str:
    card = MODEL_UI_CARDS[model_id]
    recommendation = " (recommended)" if recommended else ""
    return (
        f"{model_id}{recommendation} - {card['display_name']} "
        f"(best for: {card['best_for']}, compute: {card['compute_profile']})"
    )


def _print_model_choices(recommended_model: str) -> None:
    console.print(Rule("Model Choices", style="wf.header"))
    for model_id in _model_choice_order(recommended_model):
        card = MODEL_UI_CARDS[model_id]
        badge = (
            " [wf.accent]\u2605 recommended[/wf.accent]" if model_id == recommended_model else ""
        )
        console.print(f"  [wf.brand]{model_id}[/wf.brand]{badge}")
        console.print(f"    [wf.label]Best for:[/wf.label]     {card['best_for']}")
        console.print(f"    [wf.label]Obs fit:[/wf.label]      {card['observation_fit']}")
        console.print(f"    [wf.label]Compute:[/wf.label]      {card['compute_profile']}")
        console.print(f"    [wf.label]Tradeoff:[/wf.label]     {card['tradeoff_note']}")
        console.print()


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
            f"[wf.caution]Model selection prompt failed ({exc}). "
            f"Using recommended model: {recommended_model}[/wf.caution]"
        )
        return recommended_model


def _arrow_select(
    label: str,
    options: Sequence[dict[str, Any]],
    *,
    default_index: int = 0,
) -> Any:
    """Interactive arrow-key selector using raw terminal input.

    Uses ↑/↓ to move, Enter to confirm, Ctrl-C to cancel.
    Handles terminal resize by recalculating screen rows from stored
    plain-text lines at the *current* terminal width before erasing.
    """
    import signal
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    selected = default_index
    n = len(options)
    # Plain-text content of each rendered line (ANSI stripped).
    # Used to recalculate screen rows at the current terminal width
    # so that erase is correct even after a terminal resize.
    prev_plain_lines: list[str] = []
    resized = False

    def _on_resize(_sig: int, _frame: object) -> None:
        nonlocal resized
        resized = True

    def _draw(idx: int, *, erase: bool = False) -> None:
        nonlocal prev_plain_lines
        if erase and prev_plain_lines:
            # Recalculate at CURRENT width — handles resize correctly.
            tw = shutil.get_terminal_size().columns
            erase_count = sum(max(1, -(-len(p) // tw)) for p in prev_plain_lines)
            for _ in range(erase_count):
                sys.stdout.write("\033[A\033[2K")
            sys.stdout.flush()
        new_plain: list[str] = []
        for i, opt in enumerate(options):
            if i == idx:
                markup = f"  [wf.accent]\u203a[/wf.accent] [bold]{opt['name']}[/bold]"
            else:
                markup = f"    [wf.muted]{opt['name']}[/wf.muted]"
            with console.capture() as capture:
                console.print(markup)
            output = capture.get()
            sys.stdout.write(output)
            # Store plain-text of each rendered line for resize-aware erase
            for seg in output.split("\n"):
                plain = _ANSI_RE.sub("", seg)
                if plain:
                    new_plain.append(plain)
        prev_plain_lines = new_plain
        sys.stdout.flush()

    old_handler = signal.getsignal(signal.SIGWINCH)
    signal.signal(signal.SIGWINCH, _on_resize)
    try:
        console.print(f"[wf.header]{label}[/wf.header]")
        console.print("[wf.dim]  \u2191/\u2193 select  Enter confirm[/wf.dim]")
        _draw(selected)
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            # On resize the next keypress triggers a full redraw first.
            if resized:
                resized = False
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                _draw(selected, erase=True)
                tty.setraw(fd)
            if ch in ("\r", "\n"):
                break
            if ch == "\x03":  # Ctrl-C
                raise KeyboardInterrupt
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":  # Up
                    selected = (selected - 1) % n
                elif seq == "[B":  # Down
                    selected = (selected + 1) % n
                else:
                    continue
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                _draw(selected, erase=True)
                tty.setraw(fd)
    finally:
        signal.signal(signal.SIGWINCH, old_handler)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return options[selected]["value"]


def _numbered_select(
    label: str,
    options: Sequence[dict[str, Any]],
    *,
    default_index: int = 0,
) -> Any:
    """Arrow-key selector with numbered fallback for non-interactive terminals."""
    from ._utils import _is_interactive_terminal

    if _is_interactive_terminal():
        try:
            return _arrow_select(label, options, default_index=default_index)
        except Exception:
            pass  # Fall through to numbered input

    console.print(f"[wf.header]{label}[/wf.header]")
    for i, opt in enumerate(options, 1):
        marker = "\u203a" if (i - 1) == default_index else " "
        console.print(f"  {marker} [wf.accent]\\[{i}][/wf.accent] {opt['name']}")
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
            f"[wf.caution]Model selection prompt failed ({exc}). "
            f"Using recommended model: {recommended_model}[/wf.caution]"
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
    console.print("[wf.header]Choose your environment type:[/wf.header]")
    for key in ("atari", "mujoco", "custom"):
        option = ENVIRONMENT_OPTIONS[key]
        console.print(
            f"- [wf.brand]{key}[/wf.brand]: {option['description']} "
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
    console.print()
    console.print(f"[wf.pass]Recommended model:[/wf.pass] {model}")
    console.print(
        key_value_panel(
            {
                "Recommended model": model,
                "Why recommended now": why_now,
                "When to choose other": when_other,
            },
            title="Recommendation",
            border="wf.accent",
        )
    )


def _print_configuration_summary(context: dict[str, Any], target_path: Path, force: bool) -> None:
    model_fit = MODEL_UI_CARDS.get(str(context["model"]), {}).get("best_for", "n/a")
    console.print(
        key_value_panel(
            {
                "Project name": context["project_name"],
                "Environment": context["environment"],
                "Observation shape": str(tuple(context["obs_shape"])),
                "Action dim": str(context["action_dim"]),
                "Shape/Action guide": OBS_ACTION_GUIDE_URL,
                "Total steps": str(context["training_total_steps"]),
                "Batch size": str(context["training_batch_size"]),
                "Model": context["model"],
                "Model fit": model_fit,
                "Device": context["device"],
                "Target path": str(target_path.resolve()),
                "Overwrite existing": "yes" if force else "no",
            },
            title="Configuration Summary",
            border="wf.border",
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
            f"[wf.muted]Using {environment} defaults: "
            f"obs_shape={tuple(obs_shape)}, action_dim={action_dim}[/wf.muted]"
        )
    else:
        console.print(
            f"[wf.muted]Need help with Observation shape or Action dim? {OBS_ACTION_GUIDE_URL}[/wf.muted]"
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
                    f"[wf.fail]Invalid observation shape:[/wf.fail] {exc} "
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
                    f"[wf.fail]Invalid action dim:[/wf.fail] {exc} "
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
                console.print(f"[wf.fail]Invalid total steps:[/wf.fail] {exc}")
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
                console.print(f"[wf.fail]Invalid batch size:[/wf.fail] {exc}")
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
        console.print("[wf.caution]CUDA is not available. Falling back to CPU.[/wf.caution]")
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
        console.print("[wf.fail]Project name cannot be empty.[/wf.fail]")
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
            f"[wf.muted]Using {environment} defaults: "
            f"obs_shape={tuple(obs_shape)}, action_dim={action_dim}[/wf.muted]"
        )
    else:
        console.print(
            f"[wf.muted]Need help with Observation shape or Action dim? {OBS_ACTION_GUIDE_URL}[/wf.muted]"
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
                    f"[wf.fail]Invalid observation shape:[/wf.fail] {exc} "
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
                    f"[wf.fail]Invalid action dim:[/wf.fail] {exc} "
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
                console.print(f"[wf.fail]Invalid total steps:[/wf.fail] {exc}")
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
                console.print(f"[wf.fail]Invalid batch size:[/wf.fail] {exc}")
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
        console.print("[wf.caution]CUDA is not available. Falling back to CPU.[/wf.caution]")
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
    term_width = shutil.get_terminal_size().columns
    if term_width >= 50:  # ASCII logo is 48 chars wide + 2 margin
        console.print(f"[wf.brand]{ASCII_LOGO}[/wf.brand]")
    console.print("[wf.header]WorldFlux CLI[/wf.header]  |  scaffold world-model projects")
    console.print()
    console.print(
        key_value_panel(
            {
                "What": "Create a ready-to-run WorldFlux project with train.py, "
                "inference.py, and worldflux.toml",
                "How": "This guided setup explains each question and applies safe defaults",
            },
            title="Guided Setup",
            border="wf.border",
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
        f"[wf.info]Installing optional dependencies:[/wf.info] "
        f"[wf.label]{', '.join(packages)}[/wf.label]"
    )
    completed = subprocess.run(cmd, check=False)
    if completed.returncode == 0:
        return True

    if shutil.which("uv") is None:
        return False

    console.print("[wf.caution]pip is unavailable. Retrying with uv pip.[/wf.caution]")
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
        f"[wf.caution]Optional Atari dependencies are missing:[/wf.caution] "
        f"{', '.join(missing_packages)}"
    )
    console.print(
        "[wf.muted]Without them, live gameplay falls back to random replay data.[/wf.muted]"
    )

    if not _cli._is_interactive_terminal():
        console.print(f"[wf.muted]Install later with: {install_command}[/wf.muted]")
        if shutil.which("uv"):
            console.print(f"[wf.muted]Or with uv: {uv_install_command}[/wf.muted]")
        return

    if not _cli._confirm_optional_dependency_install(missing_packages):
        console.print(f"[wf.caution]Skipped install.[/wf.caution] You can run: {install_command}")
        return

    if _cli._install_packages_with_pip(missing_packages):
        console.print("[wf.ok]Optional Atari dependencies installed successfully.[/wf.ok]")
    else:
        console.print(
            "[wf.caution]Install failed. Training can continue in fallback mode.[/wf.caution]"
        )
        console.print(f"[wf.muted]Retry with: {install_command}[/wf.muted]")
        if shutil.which("uv"):
            console.print(f"[wf.muted]Or with uv: {uv_install_command}[/wf.muted]")


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
        console.print(
            "[wf.info]Preparing bootstrap dependencies before project generation...[/wf.info]"
        )

        def _bootstrap_progress(message: str) -> None:
            console.print(f"[wf.dim][bootstrap][/wf.dim] {message}")

        deps_result = _cli.ensure_init_dependencies(context, progress_callback=_bootstrap_progress)
        if deps_result.skipped:
            console.print(f"[wf.caution]{deps_result.message}[/wf.caution]")
        elif deps_result.success:
            console.print(f"[wf.ok]{deps_result.message}[/wf.ok]")
            if deps_result.launcher:
                context["preferred_python_launcher"] = deps_result.launcher
        else:
            console.print(f"[wf.fail]Dependency bootstrap failed:[/wf.fail] {deps_result.message}")
            for line in deps_result.diagnostics:
                console.print(f"[wf.dim]{line}[/wf.dim]")
            if deps_result.retry_commands:
                console.print("[wf.caution]Retry with:[/wf.caution]")
                for command in deps_result.retry_commands:
                    console.print(f"[wf.dim]{command}[/wf.dim]")
            raise typer.Exit(code=1)

        if context["device"] == "cuda" and not torch.cuda.is_available():
            console.print("[wf.caution]CUDA is not available. Falling back to CPU.[/wf.caution]")
            context["device"] = "cpu"

        target_path = path if path is not None else Path.cwd() / str(context["project_name"])
        _cli._print_configuration_summary(context, target_path, force=force)
        if not _cli._confirm_generation():
            console.print(
                "\n[wf.caution]Initialization cancelled. No files were generated.[/wf.caution]"
            )
            raise typer.Exit(code=130)
        # Pop launcher before scaffold validation (templates use .get with default)
        preferred_launcher = context.pop("preferred_python_launcher", None)
        _cli.generate_project(target_path, context, force=force)
    except KeyboardInterrupt:
        console.print(
            "\n[wf.caution]Initialization cancelled. No files were generated.[/wf.caution]"
        )
        raise typer.Exit(code=130) from None
    except (ValueError, FileExistsError, IsADirectoryError, OSError) as exc:
        console.print(f"[wf.fail]Error:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    resolved_target = target_path.resolve()
    console.print(f"\n[wf.pass]Project created:[/wf.pass] {resolved_target}")
    launcher = str(preferred_launcher or _cli._resolve_python_launcher())
    console.print(
        result_banner(
            passed=True,
            title="Next Steps",
            lines=[
                f"[wf.label]1.[/wf.label] cd {resolved_target}",
                "[wf.label]2.[/wf.label] worldflux train             # start training",
                f"[wf.label]3.[/wf.label] {launcher} train.py   # start training (legacy path)",
                f"[wf.label]4.[/wf.label] {launcher} inference.py   # run inference (legacy path)",
                "[wf.label]5.[/wf.label] worldflux verify --target ./outputs  # verify your model",
                "[wf.label]6.[/wf.label] Edit worldflux.toml to tune settings for your environment",
            ],
        )
    )
