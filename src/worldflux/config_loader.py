"""Unified worldflux.toml configuration loader.

Parses ``worldflux.toml`` into structured types consumed by
``worldflux train``, ``worldflux verify``, and the scaffold system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    output_dir: str = "./outputs"


@dataclass(frozen=True)
class VerifySectionConfig:
    """Verification parameters from ``[verify]``."""

    baseline: str = "official/dreamerv3"
    env: str = "atari/pong"


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
    verify: VerifySectionConfig
    cloud: CloudSectionConfig
    flywheel: FlywheelSectionConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


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

    project_name = str(raw.get("project_name", config_path.parent.name))
    environment = str(raw.get("environment", "custom"))
    model = str(raw.get("model", ""))
    model_type = str(raw.get("model_type", ""))

    if not model:
        model = "dreamer:ci" if environment == "atari" else "tdmpc2:ci"
    if not model_type:
        model_type = _infer_model_type(model)

    arch_raw = raw.get("architecture", {})
    if not isinstance(arch_raw, dict):
        arch_raw = {}
    architecture = ArchitectureConfig(
        obs_shape=_parse_obs_shape(arch_raw.get("obs_shape", [3, 64, 64])),
        action_dim=int(arch_raw.get("action_dim", 6)),
        hidden_dim=int(arch_raw.get("hidden_dim", 32)),
    )

    train_raw = raw.get("training", {})
    if not isinstance(train_raw, dict):
        train_raw = {}
    training = TrainingSectionConfig(
        total_steps=int(train_raw.get("total_steps", 100_000)),
        batch_size=int(train_raw.get("batch_size", 16)),
        sequence_length=int(train_raw.get("sequence_length", 50)),
        learning_rate=float(train_raw.get("learning_rate", 3e-4)),
        device=str(train_raw.get("device", "cpu")),
        output_dir=str(train_raw.get("output_dir", "./outputs")),
    )

    verify_raw = raw.get("verify", {})
    if not isinstance(verify_raw, dict):
        verify_raw = {}
    verify = VerifySectionConfig(
        baseline=str(verify_raw.get("baseline", "official/dreamerv3")),
        env=str(verify_raw.get("env", "atari/pong")),
    )

    cloud_raw = raw.get("cloud", {})
    if not isinstance(cloud_raw, dict):
        cloud_raw = {}
    cloud = CloudSectionConfig(
        gpu_type=str(cloud_raw.get("gpu_type", "a100")),
        spot=bool(cloud_raw.get("spot", True)),
        region=str(cloud_raw.get("region", "us-east-1")),
        timeout_hours=int(cloud_raw.get("timeout_hours", 24)),
    )

    flywheel_raw = raw.get("flywheel", {})
    if not isinstance(flywheel_raw, dict):
        flywheel_raw = {}
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
        verify=verify,
        cloud=cloud,
        flywheel=flywheel,
        raw=raw,
    )
