# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for stable training API exports."""

from __future__ import annotations

import pytest

import worldflux.training as training


def test_training_public_api_excludes_placeholder_fsdp_trainer() -> None:
    assert "FSDPTrainer" not in training.__all__
    assert not hasattr(training, "FSDPTrainer")

    with pytest.raises(ImportError):
        exec("from worldflux.training import FSDPTrainer")
