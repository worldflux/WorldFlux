"""Regression guard to ensure public API remains compatible."""

from __future__ import annotations

import inspect

import numpy as np

from worldflux import create_world_model
from worldflux.training import ReplayBuffer, Trainer, TrainingConfig


def test_factory_and_training_public_symbols_are_unchanged() -> None:
    create_sig = inspect.signature(create_world_model)
    assert "model" in create_sig.parameters
    assert "obs_shape" in create_sig.parameters
    assert "action_dim" in create_sig.parameters

    config_sig = inspect.signature(TrainingConfig)
    assert "total_steps" in config_sig.parameters
    assert "batch_size" in config_sig.parameters
    assert "sequence_length" in config_sig.parameters


def test_public_training_path_executes_without_parity_side_effects(tmp_path) -> None:
    obs_shape = (8,)
    action_dim = 3

    buffer = ReplayBuffer(capacity=256, obs_shape=obs_shape, action_dim=action_dim)
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(size=(64, *obs_shape), dtype=np.float32)
    actions = rng.uniform(low=-1.0, high=1.0, size=(64, action_dim)).astype(np.float32)
    rewards = rng.normal(size=64).astype(np.float32)
    dones = np.zeros(64, dtype=np.float32)
    dones[-1] = 1.0
    buffer.add_episode(obs=obs, actions=actions, rewards=rewards, dones=dones)

    model = create_world_model("tdmpc2:ci", obs_shape=obs_shape, action_dim=action_dim)
    trainer = Trainer(
        model,
        TrainingConfig(
            total_steps=1,
            batch_size=4,
            sequence_length=8,
            output_dir=str(tmp_path / "train"),
            device="cpu",
            seed=0,
            log_interval=1,
            save_interval=10,
        ),
    )
    trainer.train(buffer)

    dreamer = create_world_model("dreamerv3:ci", obs_shape=(3, 32, 32), action_dim=4)
    assert dreamer is not None
