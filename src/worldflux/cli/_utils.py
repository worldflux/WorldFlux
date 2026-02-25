"""Shared utilities for WorldFlux CLI commands."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

from ._app import ATARI_OPTIONAL_DEPENDENCIES, ENVIRONMENT_OPTIONS, MODEL_CHOICE_IDS


def _is_preset_environment(environment: str) -> bool:
    """Return True if the environment has known default obs/action values."""
    return environment in ENVIRONMENT_OPTIONS and environment != "custom"


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


def _resolve_python_launcher() -> str:
    """Resolve a runnable Python launcher command for the current environment."""
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


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
