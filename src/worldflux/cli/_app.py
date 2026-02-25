"""App definition, constants, and root callback for WorldFlux CLI."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    help="Unified Python interface for world models in reinforcement learning.",
    epilog=(
        "[dim]Common workflows:\n"
        "  Get started      → worldflux init\n"
        "  Train a model    → worldflux train\n"
        "  List models      → worldflux models list\n"
        "  Evaluate         → worldflux eval ./outputs --suite quick\n"
        "  System info      → worldflux doctor\n"
        "  Docs             → https://worldflux.ai/[/dim]"
    ),
    rich_markup_mode="rich",
    no_args_is_help=True,
)
parity_app = typer.Typer(
    help="Verify model outputs match upstream reference implementations.",
    rich_markup_mode="rich",
)
parity_campaign_app = typer.Typer(help="Run reproducible parity campaigns.")
models_app = typer.Typer(help="Browse and inspect model presets.")

parity_app.add_typer(parity_campaign_app, name="campaign")

console = Console()

# ---------------------------------------------------------------------------
# Constants (moved from cli.py)
# ---------------------------------------------------------------------------

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
TOTAL_STEPS_PRESETS: tuple[dict[str, object], ...] = (
    {"name": "50,000 (quick experiment)", "value": 50_000},
    {"name": "100,000 (recommended)", "value": 100_000},
    {"name": "500,000 (thorough)", "value": 500_000},
    {"name": "Custom...", "value": "custom"},
)
BATCH_SIZE_PRESETS: tuple[dict[str, object], ...] = (
    {"name": "8 (low memory)", "value": 8},
    {"name": "16 (recommended)", "value": 16},
    {"name": "32 (faster, more memory)", "value": 32},
    {"name": "64 (high throughput)", "value": 64},
    {"name": "Custom...", "value": "custom"},
)
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

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        import platform

        import torch

        from worldflux import __version__

        if torch.cuda.is_available():
            cuda_info = "CUDA " + (torch.version.cuda or "n/a")
        else:
            cuda_info = "CPU"
        console.print(
            f"worldflux [bold]{__version__}[/bold]  "
            f"(Python {platform.python_version()}, "
            f"PyTorch {torch.__version__}, {cuda_info})"
        )
        raise typer.Exit()


def _debug_callback(debug: bool) -> None:
    """Enable debug logging when --debug is passed."""
    if debug:
        import logging

        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
) -> None:
    """WorldFlux command-line interface."""
    _debug_callback(debug)
