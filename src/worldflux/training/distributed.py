# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Minimal distributed training launch helpers."""

from __future__ import annotations

from typing import Any

from .config import TrainingConfig


def build_launch_config(config: TrainingConfig) -> dict[str, Any]:
    """Return a normalized launch description for distributed execution."""
    return {
        "enabled": config.distributed_mode != "none",
        "mode": config.distributed_mode,
        "world_size": int(config.distributed_world_size),
        "device": config.resolve_device(),
    }
