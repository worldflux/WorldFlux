# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for distributed training configuration helpers."""

from __future__ import annotations

import pytest

from worldflux.training.config import TrainingConfig
from worldflux.training.distributed import build_launch_config


def test_training_config_supports_distributed_mode() -> None:
    config = TrainingConfig(distributed_mode="ddp")
    assert config.distributed_mode == "ddp"


def test_build_launch_config_returns_single_process_defaults() -> None:
    config = TrainingConfig(distributed_mode="ddp", distributed_world_size=2)
    launch = build_launch_config(config)

    assert launch["mode"] == "ddp"
    assert launch["world_size"] == 2
    assert launch["enabled"] is True


def test_training_config_rejects_unknown_distributed_mode() -> None:
    with pytest.raises(Exception):
        TrainingConfig(distributed_mode="unknown")
