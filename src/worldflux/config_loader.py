# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Unified worldflux.toml configuration loader.

Parses ``worldflux.toml`` into structured types consumed by
``worldflux train``, ``worldflux verify``, and the scaffold system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redefine]


@dataclass(frozen=True)
class ArchitectureConfig:
    """Model architecture parameters from ``[architecture]``."""

    obs_shape: tuple[int, ...]
    action_dim: int
    hidden_dim: int = 32


@dataclass(frozen=True)
class TrainingSectionConfig:
    """Training parameters from ``[training]``."""

    total_steps: int = 100_000
    batch_size: int = 16
    sequence_length: int = 50
    learning_rate: float = 3e-4
    device: str = "cpu"
    backend: str = "native_torch"
    backend_profile: str = ""
    output_dir: str = "./outputs"


@dataclass(frozen=True)
class DataSectionConfig:
    """Training data parameters from ``[data]``."""

    source: str = "random"
    num_episodes: int = 100
    episode_length: int = 100
    buffer_capacity: int = 10_000
    gym_env: str = ""


@dataclass(frozen=True)
class GameplaySectionConfig:
    """Gameplay stream parameters from ``[gameplay]``."""

    enabled: bool = False
    fps: int = 8
    max_frames: int = 512


@dataclass(frozen=True)
class OnlineCollectionSectionConfig:
    """Online collection parameters from ``[online_collection]``."""

    enabled: bool = False
    warmup_transitions: int = 512
    collect_steps_per_update: int = 64
    max_episode_steps: int = 100


@dataclass(frozen=True)
class InferenceSectionConfig:
    """Inference helper parameters from ``[inference]``."""

    horizon: int = 15
    checkpoint: str = "./outputs/checkpoint_best.pt"


@dataclass(frozen=True)
class VisualizationSectionConfig:
    """Visualization parameters from ``[visualization]``."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    refresh_ms: int = 1000
    history_max_points: int = 2000
    open_browser: bool = False


@dataclass(frozen=True)
class VerifySectionConfig:
    """Verification parameters from ``[verify]``."""

    baseline: str = "official/dreamerv3"
    env: str = "atari/pong"
    backend: str = "native_torch"
    backend_profile: str = ""
    mode: str = "auto"
    proof_claim: str = "compare"
    allow_official_only: bool = False


@dataclass(frozen=True)
class CloudSectionConfig:
    """Cloud training parameters from ``[cloud]``."""

    gpu_type: str = "a100"
    spot: bool = True
    region: str = "us-east-1"
    timeout_hours: int = 24


@dataclass(frozen=True)
class FlywheelSectionConfig:
    """Data flywheel privacy controls from ``[flywheel]``."""

    opt_in: bool = False
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level parsed representation of ``worldflux.toml``."""

    project_name: str
    environment: str
    model: str
    model_type: str
    architecture: ArchitectureConfig
    training: TrainingSectionConfig
    data: DataSectionConfig
    gameplay: GameplaySectionConfig
    online_collection: OnlineCollectionSectionConfig
    inference: InferenceSectionConfig
    visualization: VisualizationSectionConfig
    verify: VerifySectionConfig
    cloud: CloudSectionConfig
    flywheel: FlywheelSectionConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "project_name",
        "environment",
        "model",
        "model_type",
        "architecture",
        "training",
        "data",
        "gameplay",
        "online_collection",
        "inference",
        "visualization",
        "verify",
        "cloud",
        "flywheel",
    }
)


def _allowed_section_keys(config_cls: type) -> frozenset[str]:
    return frozenset(field_def.name for field_def in fields(config_cls))


def _validate_top_level_keys(raw: dict[str, Any]) -> None:
    for key in raw:
        if key not in _TOP_LEVEL_KEYS:
            raise ValueError(f"Unknown top-level key: {key}")


def _validate_section_keys(
    section_name: str,
    raw: dict[str, Any],
    config_cls: type,
) -> None:
    allowed = _allowed_section_keys(config_cls)
    for key in raw:
        if key not in allowed:
            raise ValueError(f"Unknown key '{section_name}.{key}' in worldflux.toml")


def _coerce_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    section = raw.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Section '{key}' must be a TOML table.")
    return section


def _parse_obs_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, list | tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, int):
        return (value,)
    raise ValueError(f"Invalid obs_shape: {value!r}")


def _infer_model_type(model: str) -> str:
    if model.startswith("dreamer"):
        return "dreamer"
    if model.startswith("tdmpc2"):
        return "tdmpc2"
    return model.split(":")[0] if ":" in model else model


def load_config(path: str | Path = "worldflux.toml") -> ProjectConfig:
    """Load and parse a ``worldflux.toml`` file.

    Parameters
    ----------
    path:
        Path to the TOML configuration file.  Defaults to
        ``worldflux.toml`` in the current directory.

    Returns
    -------
    ProjectConfig
        Structured configuration object.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If required fields are missing or malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    _validate_top_level_keys(raw)

    project_name = str(raw.get("project_name", config_path.parent.name))
    environment = str(raw.get("environment", "custom"))
    model = str(raw.get("model", ""))
    model_type = str(raw.get("model_type", ""))

    if not model:
        model = "dreamer:ci" if environment == "atari" else "tdmpc2:ci"
    if not model_type:
        model_type = _infer_model_type(model)
    if model_type not in {"dreamer", "tdmpc2", "tdmpc"}:
        raise ValueError(
            "worldflux.toml supported newcomer schema only accepts DreamerV3 / TD-MPC2 families."
        )

    arch_raw = _coerce_section(raw, "architecture")
    _validate_section_keys("architecture", arch_raw, ArchitectureConfig)
    architecture = ArchitectureConfig(
        obs_shape=_parse_obs_shape(arch_raw.get("obs_shape", [3, 64, 64])),
        action_dim=int(arch_raw.get("action_dim", 6)),
        hidden_dim=int(arch_raw.get("hidden_dim", 32)),
    )

    train_raw = _coerce_section(raw, "training")
    _validate_section_keys("training", train_raw, TrainingSectionConfig)
    training = TrainingSectionConfig(
        total_steps=int(train_raw.get("total_steps", 100_000)),
        batch_size=int(train_raw.get("batch_size", 16)),
        sequence_length=int(train_raw.get("sequence_length", 50)),
        learning_rate=float(train_raw.get("learning_rate", 3e-4)),
        device=str(train_raw.get("device", "cpu")),
        backend=str(train_raw.get("backend", "native_torch")),
        backend_profile=str(train_raw.get("backend_profile", "")),
        output_dir=str(train_raw.get("output_dir", "./outputs")),
    )

    data_raw = _coerce_section(raw, "data")
    _validate_section_keys("data", data_raw, DataSectionConfig)
    data = DataSectionConfig(
        source=str(data_raw.get("source", "random")).strip().lower() or "random",
        num_episodes=int(data_raw.get("num_episodes", 100)),
        episode_length=int(data_raw.get("episode_length", 100)),
        buffer_capacity=int(data_raw.get("buffer_capacity", 10_000)),
        gym_env=str(data_raw.get("gym_env", "")).strip(),
    )

    gameplay_raw = _coerce_section(raw, "gameplay")
    _validate_section_keys("gameplay", gameplay_raw, GameplaySectionConfig)
    gameplay = GameplaySectionConfig(
        enabled=bool(gameplay_raw.get("enabled", False)),
        fps=int(gameplay_raw.get("fps", 8)),
        max_frames=int(gameplay_raw.get("max_frames", 512)),
    )

    online_raw = _coerce_section(raw, "online_collection")
    _validate_section_keys("online_collection", online_raw, OnlineCollectionSectionConfig)
    online_collection = OnlineCollectionSectionConfig(
        enabled=bool(online_raw.get("enabled", False)),
        warmup_transitions=int(online_raw.get("warmup_transitions", 512)),
        collect_steps_per_update=int(online_raw.get("collect_steps_per_update", 64)),
        max_episode_steps=int(online_raw.get("max_episode_steps", 100)),
    )

    inference_raw = _coerce_section(raw, "inference")
    _validate_section_keys("inference", inference_raw, InferenceSectionConfig)
    inference = InferenceSectionConfig(
        horizon=int(inference_raw.get("horizon", 15)),
        checkpoint=str(inference_raw.get("checkpoint", "./outputs/checkpoint_best.pt")).strip()
        or "./outputs/checkpoint_best.pt",
    )

    visualization_raw = _coerce_section(raw, "visualization")
    _validate_section_keys("visualization", visualization_raw, VisualizationSectionConfig)
    visualization = VisualizationSectionConfig(
        enabled=bool(visualization_raw.get("enabled", False)),
        host=str(visualization_raw.get("host", "127.0.0.1")).strip() or "127.0.0.1",
        port=int(visualization_raw.get("port", 8765)),
        refresh_ms=int(visualization_raw.get("refresh_ms", 1000)),
        history_max_points=int(visualization_raw.get("history_max_points", 2000)),
        open_browser=bool(visualization_raw.get("open_browser", False)),
    )

    verify_raw = _coerce_section(raw, "verify")
    _validate_section_keys("verify", verify_raw, VerifySectionConfig)
    verify = VerifySectionConfig(
        baseline=str(verify_raw.get("baseline", "official/dreamerv3")),
        env=str(verify_raw.get("env", "atari/pong")),
        backend=str(verify_raw.get("backend", "native_torch")),
        backend_profile=str(verify_raw.get("backend_profile", "")),
        mode=str(verify_raw.get("mode", "auto")),
        proof_claim=str(verify_raw.get("proof_claim", "compare")),
        allow_official_only=bool(verify_raw.get("allow_official_only", False)),
    )

    cloud_raw = _coerce_section(raw, "cloud")
    _validate_section_keys("cloud", cloud_raw, CloudSectionConfig)
    cloud = CloudSectionConfig(
        gpu_type=str(cloud_raw.get("gpu_type", "a100")),
        spot=bool(cloud_raw.get("spot", True)),
        region=str(cloud_raw.get("region", "us-east-1")),
        timeout_hours=int(cloud_raw.get("timeout_hours", 24)),
    )

    flywheel_raw = _coerce_section(raw, "flywheel")
    _validate_section_keys("flywheel", flywheel_raw, FlywheelSectionConfig)
    flywheel = FlywheelSectionConfig(
        opt_in=bool(flywheel_raw.get("opt_in", False)),
        privacy_epsilon=float(flywheel_raw.get("privacy_epsilon", 1.0)),
        privacy_delta=float(flywheel_raw.get("privacy_delta", 1e-5)),
    )

    return ProjectConfig(
        project_name=project_name,
        environment=environment,
        model=model,
        model_type=model_type,
        architecture=architecture,
        training=training,
        data=data,
        gameplay=gameplay,
        online_collection=online_collection,
        inference=inference,
        visualization=visualization,
        verify=verify,
        cloud=cloud,
        flywheel=flywheel,
        raw=raw,
    )
