"""Smoke tests for universal-architecture model families."""

from __future__ import annotations

import pytest
import torch

from worldflux import create_world_model
from worldflux.core.batch import Batch
from worldflux.training import Trainer, TrainingConfig


class _SmokeProvider:
    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def sample(
        self, batch_size: int, seq_len: int | None = None, device: str | torch.device = "cpu"
    ) -> Batch:
        t = seq_len or 3
        obs = torch.randn(batch_size, t, self.obs_dim, device=device)
        next_obs = torch.randn(batch_size, t, self.obs_dim, device=device)
        actions = torch.randn(batch_size, t, self.action_dim, device=device)
        rewards = torch.randn(batch_size, t, device=device)
        terminations = torch.zeros(batch_size, t, device=device)
        return Batch(
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            context=obs,
            target=next_obs,
            layouts={
                "obs": "BT...",
                "next_obs": "BT...",
                "actions": "BT...",
                "rewards": "BT",
                "terminations": "BT",
                "context": "BT...",
                "target": "BT...",
            },
            strict_layout=True,
        )


@pytest.mark.parametrize(
    "model_id",
    [
        "vjepa2:base",
        "token:base",
        "diffusion:base",
        "dit:base",
        "ssm:base",
        "renderer3d:base",
        "physics:base",
        "gan:base",
    ],
)
def test_model_family_smoke_train_one_step(model_id: str, tmp_path):
    with pytest.warns(DeprecationWarning):
        model = create_world_model(model_id, obs_shape=(4,), action_dim=2, api_version="v0.2")

    provider = _SmokeProvider(obs_dim=4, action_dim=2)
    cfg = TrainingConfig(
        total_steps=1,
        batch_size=3,
        sequence_length=3,
        output_dir=str(tmp_path / model_id.replace(":", "-")),
        device="cpu",
    )
    trainer = Trainer(model, cfg)
    trained = trainer.train(provider, num_steps=1)
    assert trained is model
