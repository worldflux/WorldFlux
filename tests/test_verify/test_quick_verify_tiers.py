# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for tiered quick verification."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldflux import create_world_model
from worldflux.training import Trainer, TrainingConfig
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


@pytest.mark.parametrize(
    ("model_id", "kwargs"),
    (
        ("tdmpc2:5m", {"obs_shape": (39,), "action_dim": 6}),
        (
            "dreamerv3:size12m",
            {
                "obs_shape": (8,),
                "action_dim": 2,
                "encoder_type": "mlp",
                "decoder_type": "mlp",
            },
        ),
    ),
)
def test_load_model_from_target_restores_non_ci_save_pretrained(
    tmp_path: Path,
    model_id: str,
    kwargs: dict[str, object],
) -> None:
    from worldflux.verify.quick import _load_model_from_target

    model = create_world_model(model_id, device="cpu", **kwargs)
    target = tmp_path / "saved-model"
    model.save_pretrained(str(target))

    restored = _load_model_from_target(target, device="cpu")

    assert restored.config.model_type == model.config.model_type
    assert restored.config.model_name == model.config.model_name


@pytest.mark.parametrize(
    ("model_id", "kwargs"),
    (
        ("tdmpc2:5m", {"obs_shape": (39,), "action_dim": 6}),
        (
            "dreamerv3:size12m",
            {
                "obs_shape": (8,),
                "action_dim": 2,
                "encoder_type": "mlp",
                "decoder_type": "mlp",
            },
        ),
    ),
)
def test_load_model_from_target_restores_trainer_checkpoint_with_exact_config(
    tmp_path: Path,
    model_id: str,
    kwargs: dict[str, object],
) -> None:
    from worldflux.verify.quick import _load_model_from_target

    model = create_world_model(model_id, device="cpu", **kwargs)
    trainer = Trainer(
        model,
        TrainingConfig(
            total_steps=1,
            batch_size=2,
            sequence_length=2,
            output_dir=str(tmp_path / "outputs"),
            device="cpu",
            auto_quality_check=False,
        ),
        callbacks=[],
    )

    checkpoint = tmp_path / "outputs" / "checkpoint_final.pt"
    trainer.save_checkpoint(str(checkpoint))
    restored = _load_model_from_target(checkpoint, device="cpu")

    assert restored.config.model_type == model.config.model_type
    assert restored.config.model_name == model.config.model_name


def test_load_model_from_target_rejects_legacy_checkpoint_without_model_name(
    tmp_path: Path,
) -> None:
    from worldflux.verify.quick import _load_model_from_target

    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": {},
            "model_config": {"model_type": "tdmpc2"},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="model_name"):
        _load_model_from_target(checkpoint, device="cpu")
