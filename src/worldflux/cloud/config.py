"""Cloud configuration models for WorldFlux SaaS integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CloudConfig:
    """Cloud execution defaults consumed by CLI and backends."""

    gpu_type: str = "a100"
    spot: bool = True
    region: str = "us-east-1"
    timeout_hours: int = 24


@dataclass(frozen=True)
class FlywheelConfig:
    """Data flywheel privacy defaults consumed by cloud upload paths."""

    opt_in: bool = False
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
