"""WorldFlux CLI package."""

from __future__ import annotations

# Standard library re-exports (tests reference cli.sys, cli.shutil, etc.)
import importlib  # noqa: F401
import shutil  # noqa: F401
import subprocess  # noqa: F401
import sys  # noqa: F401

import torch  # noqa: F401 - tests reference cli.torch.cuda
from rich.prompt import Confirm as Confirm  # noqa: F401
from rich.prompt import IntPrompt as IntPrompt  # noqa: F401
from rich.prompt import Prompt as Prompt  # noqa: F401

from worldflux.bootstrap_env import (
    ensure_init_dependencies as ensure_init_dependencies,  # noqa: F401
)
from worldflux.parity import (
    CampaignRunOptions as CampaignRunOptions,  # noqa: F401
)
from worldflux.parity import (
    aggregate_runs as aggregate_runs,  # noqa: F401
)
from worldflux.parity import (
    export_campaign_source as export_campaign_source,  # noqa: F401
)
from worldflux.parity import (
    load_campaign_spec as load_campaign_spec,  # noqa: F401
)
from worldflux.parity import (
    parse_seed_csv as parse_seed_csv,  # noqa: F401
)
from worldflux.parity import (
    render_campaign_summary as render_campaign_summary,  # noqa: F401
)
from worldflux.parity import (
    render_markdown_report as render_markdown_report,  # noqa: F401
)
from worldflux.parity import (
    run_campaign as run_campaign,  # noqa: F401
)
from worldflux.parity import (
    run_suite as run_suite,  # noqa: F401
)
from worldflux.parity import (
    save_badge as save_badge,  # noqa: F401
)
from worldflux.parity.errors import ParityError as ParityError  # noqa: F401
from worldflux.scaffold import generate_project as generate_project  # noqa: F401
from worldflux.verify import ParityVerifier as ParityVerifier  # noqa: F401
from worldflux.verify import VerifyResult as VerifyResult  # noqa: F401

# Command modules are registered by _register_commands() below in desired
# help-panel order.  Individual modules are NOT imported at the top level
# because ruff isort would re-sort them alphabetically, destroying the
# ordering that controls Typer's --help output.
from ._app import (
    ASCII_LOGO as ASCII_LOGO,  # noqa: F401
)
from ._app import (
    BATCH_SIZE_PRESETS as BATCH_SIZE_PRESETS,  # noqa: F401
)
from ._app import (
    DEFAULT_BATCH_SIZE as DEFAULT_BATCH_SIZE,  # noqa: F401
)
from ._app import (
    DEFAULT_TOTAL_STEPS as DEFAULT_TOTAL_STEPS,  # noqa: F401
)
from ._app import (
    ENVIRONMENT_OPTIONS as ENVIRONMENT_OPTIONS,  # noqa: F401
)
from ._app import (
    MODEL_CHOICE_IDS as MODEL_CHOICE_IDS,  # noqa: F401
)
from ._app import (
    MODEL_UI_CARDS as MODEL_UI_CARDS,  # noqa: F401
)
from ._app import (
    OBS_ACTION_GUIDE_URL as OBS_ACTION_GUIDE_URL,  # noqa: F401
)
from ._app import (
    TOTAL_STEPS_PRESETS as TOTAL_STEPS_PRESETS,  # noqa: F401
)
from ._app import (
    app as app,  # noqa: F401
)
from ._app import (
    console as console,  # noqa: F401
)
from ._app import (
    models_app as models_app,  # noqa: F401
)
from ._app import (
    parity_app as parity_app,  # noqa: F401
)
from ._app import (
    parity_campaign_app as parity_campaign_app,  # noqa: F401
)
from ._init_cmd import (
    _confirm_generation as _confirm_generation,  # noqa: F401
)
from ._init_cmd import (
    _confirm_optional_dependency_install as _confirm_optional_dependency_install,  # noqa: F401
)
from ._init_cmd import (
    _format_environment_choice as _format_environment_choice,  # noqa: F401
)
from ._init_cmd import (
    _format_model_choice_label as _format_model_choice_label,  # noqa: F401
)
from ._init_cmd import (
    _handle_optional_atari_dependency_install as _handle_optional_atari_dependency_install,  # noqa: F401
)
from ._init_cmd import (
    _install_packages_with_pip as _install_packages_with_pip,  # noqa: F401
)
from ._init_cmd import (
    _numbered_select as _numbered_select,  # noqa: F401
)
from ._init_cmd import (
    _print_configuration_summary as _print_configuration_summary,  # noqa: F401
)
from ._init_cmd import (
    _print_environment_options as _print_environment_options,  # noqa: F401
)
from ._init_cmd import (
    _print_logo as _print_logo,  # noqa: F401
)
from ._init_cmd import (
    _print_model_choices as _print_model_choices,  # noqa: F401
)
from ._init_cmd import (
    _print_model_recommendation as _print_model_recommendation,  # noqa: F401
)
from ._init_cmd import (
    _prompt_user_configuration as _prompt_user_configuration,  # noqa: F401
)
from ._init_cmd import (
    _prompt_with_inquirer as _prompt_with_inquirer,  # noqa: F401
)
from ._init_cmd import (
    _prompt_with_rich as _prompt_with_rich,  # noqa: F401
)
from ._init_cmd import (
    _select_model_with_inquirer as _select_model_with_inquirer,  # noqa: F401
)
from ._init_cmd import (
    _select_model_with_rich as _select_model_with_rich,  # noqa: F401
)
from ._parity import (
    _resolve_campaign_seeds as _resolve_campaign_seeds,  # noqa: F401
)
from ._parity import (
    _run_parity_proof_script as _run_parity_proof_script,  # noqa: F401
)
from ._parity import (
    parity_campaign_run as parity_campaign_run,  # noqa: F401
)
from ._rich_output import (
    key_value_panel as key_value_panel,  # noqa: F401
)
from ._rich_output import (
    metric_table as metric_table,  # noqa: F401
)
from ._rich_output import (
    result_banner as result_banner,  # noqa: F401
)
from ._rich_output import (
    section_header as section_header,  # noqa: F401
)
from ._rich_output import (
    status_table as status_table,  # noqa: F401
)
from ._theme import (
    PALETTE as PALETTE,  # noqa: F401
)
from ._theme import (
    WF_THEME as WF_THEME,  # noqa: F401
)
from ._utils import (
    _hash_file as _hash_file,  # noqa: F401
)
from ._utils import (
    _is_interactive_terminal as _is_interactive_terminal,  # noqa: F401
)
from ._utils import (
    _is_preset_environment as _is_preset_environment,  # noqa: F401
)
from ._utils import (
    _missing_atari_dependency_packages as _missing_atari_dependency_packages,  # noqa: F401
)
from ._utils import (
    _model_choice_order as _model_choice_order,  # noqa: F401
)
from ._utils import (
    _model_type_from_model_id as _model_type_from_model_id,  # noqa: F401
)
from ._utils import (
    _parse_action_dim as _parse_action_dim,  # noqa: F401
)
from ._utils import (
    _parse_obs_shape as _parse_obs_shape,  # noqa: F401
)
from ._utils import (
    _parse_positive_int as _parse_positive_int,  # noqa: F401
)
from ._utils import (
    _resolve_model as _resolve_model,  # noqa: F401
)
from ._utils import (
    _resolve_python_launcher as _resolve_python_launcher,  # noqa: F401
)


def _register_commands() -> None:
    """Register command modules in desired help-panel order.

    Imports are intentionally inside this function and isort is disabled
    so that ruff cannot reorder them.  The import order determines the
    panel order shown by ``worldflux --help``.
    """
    # isort: off
    from . import _init_cmd  # noqa: F401  Getting Started
    from . import _train  # noqa: F401  Training

    app.add_typer(models_app, name="models", rich_help_panel="Model Catalog")
    from . import _models  # noqa: F401

    from . import _eval  # noqa: F401  Quality & Evaluation

    app.add_typer(parity_app, name="parity", rich_help_panel="Quality & Evaluation")
    from . import _parity  # noqa: F401
    from . import _report  # noqa: F401
    from . import _verify  # noqa: F401

    from . import _doctor  # noqa: F401  Utilities
    from . import _cloud  # noqa: F401  Cloud
    # isort: on


_register_commands()
