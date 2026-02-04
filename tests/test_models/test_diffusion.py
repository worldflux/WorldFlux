"""Tests for diffusion world model."""

import torch

from worldflux.core.batch import Batch
from worldflux.core.config import DiffusionWorldModelConfig
from worldflux.models.diffusion import DiffusionWorldModel


def test_diffusion_encode_decode():
    config = DiffusionWorldModelConfig(
        obs_shape=(4,),
        action_dim=2,
        hidden_dim=16,
        diffusion_steps=2,
    )
    model = DiffusionWorldModel(config)
    obs = torch.randn(2, 4)
    state = model.encode(obs)
    next_state = model.transition(state, torch.randn(2, 2))
    output = model.decode(next_state)
    assert output.preds["obs"].shape == (2, 4)


def test_diffusion_loss():
    config = DiffusionWorldModelConfig(
        obs_shape=(4,),
        action_dim=2,
        hidden_dim=16,
        diffusion_steps=2,
    )
    model = DiffusionWorldModel(config)
    obs = torch.randn(2, 4)
    target = torch.randn(2, 4)
    batch = Batch(obs=obs, actions=torch.randn(2, 2), target=target)
    loss_out = model.loss(batch)
    assert "diffusion_mse" in loss_out.components
    assert torch.isfinite(loss_out.loss)
