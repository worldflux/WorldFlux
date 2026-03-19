# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for tiered quick verification."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldflux.verify.quick import quick_verify


class _MiniModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"obs_shape": (4,), "action_dim": 2})()

    def encode(self, obs):
        from worldflux.core.state import State

        return State(tensors={"latent": obs.float()})

    def rollout(self, state, actions):
        from worldflux.core.trajectory import Trajectory

        rewards = actions.sum(dim=-1)
        return Trajectory(
            states=[state] * (actions.shape[0] + 1),
            actions=actions,
            rewards=rewards,
            continues=None,
        )


def test_quick_verify_supports_offline_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "model"
    target.mkdir()

    monkeypatch.setattr(
        "worldflux.verify.quick._load_model_from_target",
        lambda target_path, device: _MiniModel(),
    )

    result = quick_verify(str(target), env="atari/pong", tier="offline", episodes=3, horizon=4)
    assert result.protocol_version
    assert result.stats["verification_tier"] == "offline"


def test_quick_verify_rejects_unknown_tier(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.mkdir()

    with pytest.raises(ValueError, match="Unknown verification tier"):
        quick_verify(str(target), tier="unknown")
