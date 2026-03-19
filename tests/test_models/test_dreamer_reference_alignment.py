# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Reference-alignment tests for DreamerV3 profiles."""

from __future__ import annotations

from worldflux.core.config import DreamerV3Config
from worldflux.models.dreamer.world_model import DreamerV3WorldModel


def test_official_xl_profile_exposes_alignment_metadata() -> None:
    config = DreamerV3Config.from_size("official_xl")

    assert config.learning_rate == 1e-4
    assert config.grad_clip == 1000.0
    assert config.reference_tier == "proof"
    assert config.parity_profile == "official_xl"


def test_reference_model_reports_alignment_summary() -> None:
    config = DreamerV3Config.from_size("size12m")
    model = DreamerV3WorldModel(config)

    summary = model.reference_profile()
    assert summary["family"] == "dreamerv3"
    assert summary["preset"] == "size12m"
    assert summary["reference_tier"] == "reference"
    assert summary["parity_profile"] == ""
    assert summary["learning_rate"] == 1e-4
    assert summary["grad_clip"] == 1000.0
    assert summary["loss_scales"]["kl_dynamics"] == 0.5
    assert summary["loss_scales"]["kl_representation"] == 0.1
