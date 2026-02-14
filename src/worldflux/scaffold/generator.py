"""Project generator for ``worldflux init``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from .templates import (
    render_dashboard_index_html,
    render_dataset_py,
    render_inference_py,
    render_local_dashboard_py,
    render_readme_md,
    render_train_py,
    render_worldflux_toml,
)

REQUIRED_CONTEXT_KEYS = (
    "project_name",
    "environment",
    "model",
    "model_type",
    "obs_shape",
    "action_dim",
    "hidden_dim",
    "device",
)
DEFAULT_TOTAL_STEPS = 100000
DEFAULT_BATCH_SIZE = 16


def _validate_context_with_pydantic(context: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from pydantic import BaseModel, StrictInt, StrictStr, ValidationError
    except ModuleNotFoundError:
        return None

    schema_model: type[BaseModel]
    if hasattr(BaseModel, "model_config"):
        # pydantic v2
        class _ScaffoldContextModelV2(BaseModel):
            project_name: StrictStr
            environment: StrictStr
            model: StrictStr
            model_type: StrictStr
            obs_shape: list[StrictInt]
            action_dim: StrictInt
            hidden_dim: StrictInt
            device: StrictStr
            training_total_steps: StrictInt = DEFAULT_TOTAL_STEPS
            training_batch_size: StrictInt = DEFAULT_BATCH_SIZE

            model_config = {"extra": "forbid"}

        schema_model = _ScaffoldContextModelV2

    else:
        # pydantic v1
        class _ScaffoldContextModelV1(BaseModel):
            project_name: StrictStr
            environment: StrictStr
            model: StrictStr
            model_type: StrictStr
            obs_shape: list[StrictInt]
            action_dim: StrictInt
            hidden_dim: StrictInt
            device: StrictStr
            training_total_steps: StrictInt = DEFAULT_TOTAL_STEPS
            training_batch_size: StrictInt = DEFAULT_BATCH_SIZE

            class Config:
                extra = "forbid"

        schema_model = _ScaffoldContextModelV1

    try:
        validated = schema_model(**context)
    except ValidationError as exc:
        messages = []
        for err in exc.errors():
            location = ".".join(str(part) for part in err.get("loc", ()))
            message = str(err.get("msg", "invalid value"))
            messages.append(f"{location}: {message}")
        joined = "; ".join(messages) if messages else str(exc)
        raise ValueError(f"Invalid scaffold context schema: {joined}") from exc

    validated_any = cast(Any, validated)
    if hasattr(validated_any, "model_dump"):
        return dict(validated_any.model_dump())
    return dict(validated_any.dict())


def _validate_context(context: dict[str, Any]) -> None:
    schema_validated = _validate_context_with_pydantic(context)
    if schema_validated is not None:
        context.clear()
        context.update(schema_validated)

    missing = [key for key in REQUIRED_CONTEXT_KEYS if key not in context]
    if missing:
        raise ValueError(f"Missing scaffold context keys: {', '.join(missing)}")

    project_name = str(context["project_name"]).strip()
    if not project_name:
        raise ValueError("project_name must not be empty")
    context["project_name"] = project_name

    environment = str(context["environment"]).strip().lower()
    if environment not in {"atari", "mujoco", "custom"}:
        raise ValueError(f"Unsupported environment: {environment!r}")
    context["environment"] = environment

    model = str(context["model"]).strip()
    if ":" not in model:
        raise ValueError("model must be a preset id like 'dreamer:ci' or 'tdmpc2:ci'")
    context["model"] = model

    model_type = str(context["model_type"]).strip().lower()
    if model_type not in {"dreamer", "tdmpc2"}:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    context["model_type"] = model_type

    obs_shape_raw = context["obs_shape"]
    if not isinstance(obs_shape_raw, list | tuple) or len(obs_shape_raw) == 0:
        raise ValueError("obs_shape must be a non-empty list of positive integers")
    obs_shape = [int(dim) for dim in obs_shape_raw]
    if any(dim <= 0 for dim in obs_shape):
        raise ValueError(f"obs_shape dimensions must be positive, got {obs_shape}")
    context["obs_shape"] = obs_shape

    action_dim = int(context["action_dim"])
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    context["action_dim"] = action_dim

    hidden_dim = int(context["hidden_dim"])
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    context["hidden_dim"] = hidden_dim

    device = str(context["device"]).strip().lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    context["device"] = device

    training_total_steps = int(context.get("training_total_steps", DEFAULT_TOTAL_STEPS))
    if training_total_steps <= 0:
        raise ValueError(f"training_total_steps must be positive, got {training_total_steps}")
    context["training_total_steps"] = training_total_steps

    training_batch_size = int(context.get("training_batch_size", DEFAULT_BATCH_SIZE))
    if training_batch_size <= 0:
        raise ValueError(f"training_batch_size must be positive, got {training_batch_size}")
    context["training_batch_size"] = training_batch_size


def _validate_target_directory(target: Path, force: bool) -> None:
    if target.exists() and target.is_file():
        raise FileExistsError(f"Target path is a file: {target}")

    if target.exists() and target.is_dir() and not force:
        if any(target.iterdir()):
            raise FileExistsError(
                f"Target directory is not empty: {target}. Use --force to overwrite."
            )


def generate_project(path: str | Path, context: dict[str, Any], force: bool = False) -> list[Path]:
    """
    Generate a new WorldFlux project at ``path``.

    Args:
        path: Output directory.
        context: User-provided scaffold context.
        force: Overwrite files when target directory already contains files.

    Returns:
        List of created file paths.
    """
    context_copy = dict(context)
    _validate_context(context_copy)

    target = Path(path).expanduser()
    _validate_target_directory(target, force=force)
    target.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {
        "worldflux.toml": render_worldflux_toml(context_copy),
        "train.py": render_train_py(context_copy),
        "inference.py": render_inference_py(context_copy),
        "dataset.py": render_dataset_py(context_copy),
        "local_dashboard.py": render_local_dashboard_py(context_copy),
        "dashboard/index.html": render_dashboard_index_html(context_copy),
        "README.md": render_readme_md(context_copy),
    }

    written_files: list[Path] = []
    for relative_path, content in files.items():
        destination = target / relative_path
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(f"Cannot overwrite directory with file: {destination}")
        if destination.exists() and not force:
            raise FileExistsError(f"File already exists: {destination}. Use --force to overwrite.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written_files.append(destination)

    return written_files
