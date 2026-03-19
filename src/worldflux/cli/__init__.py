# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Lazy CLI exports and command bootstrap for WorldFlux."""

from __future__ import annotations

import importlib  # noqa: F401
import shutil  # noqa: F401
import subprocess  # noqa: F401
import sys  # noqa: F401
from importlib import import_module
from typing import Any

from rich.prompt import Confirm as Confirm
from rich.prompt import IntPrompt as IntPrompt
from rich.prompt import Prompt as Prompt

_COMMANDS_REGISTERED = False
_APP_EXPORTS = {
    "ASCII_LOGO",
    "BATCH_SIZE_PRESETS",
    "BRAND_NAME",
    "BRAND_TAGLINE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TOTAL_STEPS",
    "ENVIRONMENT_OPTIONS",
    "MAX_PANEL_WIDTH",
    "MODEL_CHOICE_IDS",
    "MODEL_UI_CARDS",
    "OBS_ACTION_GUIDE_URL",
    "TOTAL_STEPS_PRESETS",
    "WIZARD_STEPS",
    "WIZARD_TOTAL",
    "app",
    "console",
    "models_app",
    "parity_app",
    "parity_campaign_app",
}
_COMMAND_APP_EXPORTS = {"app", "models_app", "parity_app", "parity_campaign_app"}
_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "worldflux.bootstrap_env": ("ensure_init_dependencies",),
    "worldflux.parity": (
        "CampaignRunOptions",
        "aggregate_runs",
        "export_campaign_source",
        "load_campaign_spec",
        "parse_seed_csv",
        "render_campaign_summary",
        "render_markdown_report",
        "run_campaign",
        "run_suite",
        "save_badge",
    ),
    "worldflux.parity.errors": ("ParityError",),
    "worldflux.scaffold": ("generate_project",),
    "worldflux.verify": ("ParityVerifier", "VerifyResult"),
    "worldflux.cli._init_cmd": (
        "_confirm_generation",
        "_confirm_optional_dependency_install",
        "_format_environment_choice",
        "_format_model_choice_label",
        "_handle_optional_atari_dependency_install",
        "_install_packages_with_pip",
        "_numbered_select",
        "_print_configuration_summary",
        "_print_environment_options",
        "_print_logo",
        "_print_model_choices",
        "_print_model_recommendation",
        "_prompt_user_configuration",
        "_prompt_with_inquirer",
        "_prompt_with_rich",
        "_select_model_with_inquirer",
        "_select_model_with_rich",
    ),
    "worldflux.cli._parity": (
        "_resolve_campaign_seeds",
        "_run_parity_proof_script",
        "parity_campaign_run",
    ),
    "worldflux.cli._rich_output": (
        "_bounded_width",
        "grouped_summary_panel",
        "key_value_panel",
        "metric_table",
        "result_banner",
        "section_header",
        "status_table",
        "step_progress",
    ),
    "worldflux.cli._theme": ("PALETTE", "WF_THEME"),
    "worldflux.cli._utils": (
        "_hash_file",
        "_is_interactive_terminal",
        "_is_preset_environment",
        "_missing_atari_dependency_packages",
        "_model_choice_order",
        "_model_type_from_model_id",
        "_parse_action_dim",
        "_parse_obs_shape",
        "_parse_positive_int",
        "_resolve_model",
        "_resolve_python_launcher",
    ),
}
_EXPORTS: dict[str, str] = {
    name: module_path for module_path, names in _EXPORT_GROUPS.items() for name in names
}


def _load_app_module() -> Any:
    return import_module("worldflux.cli._app")


def _ensure_commands_registered() -> Any:
    global _COMMANDS_REGISTERED

    app_module = _load_app_module()
    if _COMMANDS_REGISTERED:
        return app_module

    import_module("worldflux.cli._init_cmd")
    import_module("worldflux.cli._train")
    app_module.app.add_typer(app_module.models_app, name="models", rich_help_panel="Model Catalog")
    import_module("worldflux.cli._models")
    import_module("worldflux.cli._eval")
    app_module.app.add_typer(
        app_module.parity_app,
        name="parity",
        rich_help_panel="Quality & Evaluation",
    )
    import_module("worldflux.cli._parity")
    import_module("worldflux.cli._report")
    import_module("worldflux.cli._verify")
    import_module("worldflux.cli._doctor")
    import_module("worldflux.cli._cloud")

    _COMMANDS_REGISTERED = True
    return app_module


def _load_export(name: str) -> Any:
    if name in _APP_EXPORTS:
        app_module = (
            _ensure_commands_registered() if name in _COMMAND_APP_EXPORTS else _load_app_module()
        )
        value = getattr(app_module, name)
    else:
        module = import_module(_EXPORTS[name])
        value = getattr(module, name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in _APP_EXPORTS or name in _EXPORTS:
        return _load_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        {
            "Confirm",
            "IntPrompt",
            "Prompt",
            "importlib",
            "shutil",
            "subprocess",
            "sys",
            *(_APP_EXPORTS | set(_EXPORTS)),
        }
    )


__all__ = sorted(set(_APP_EXPORTS) | set(_EXPORTS))
