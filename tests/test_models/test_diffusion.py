"""Tests for diffusion world model."""

import torch

from worldflux import AutoWorldModel
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


def test_diffusion_io_contract():
    config = DiffusionWorldModelConfig(
        obs_shape=(4,),
        action_dim=2,
        hidden_dim=16,
        diffusion_steps=2,
    )
    model = DiffusionWorldModel(config)
    contract = model.io_contract()
    assert contract.required_state_keys == ("obs",)
    assert contract.prediction_spec.tensors["obs"].shape == (4,)


def test_diffusion_accepts_prediction_target_config():
    config = DiffusionWorldModelConfig(
        obs_shape=(4,),
        action_dim=2,
        hidden_dim=16,
        diffusion_steps=2,
        prediction_target="x0",
    )
    model = DiffusionWorldModel(config)
    assert model.scheduler.prediction_target == "x0"


def test_diffusion_transition_populates_timestep_meta():
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
    assert "timestep" in next_state.meta
    assert next_state.meta["timestep"].shape == (2,)


def test_diffusion_save_pretrained_and_load(tmp_path):
    config = DiffusionWorldModelConfig(
        obs_shape=(4,),
        action_dim=2,
        hidden_dim=16,
        diffusion_steps=2,
    )
    model = DiffusionWorldModel(config)
    save_path = str(tmp_path / "diffusion_model")
    model.save_pretrained(save_path)
    loaded = AutoWorldModel.from_pretrained(save_path)
    assert loaded.config.model_type == "diffusion"
