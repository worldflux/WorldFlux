"""Tests for trainer instant profile mode and runtime metrics."""

from __future__ import annotations

from pathlib import Path

import torch

from worldflux.core.batch import Batch
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput
from worldflux.core.state import State
from worldflux.training import Trainer, TrainingConfig


class _MiniModel(WorldModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def encode(self, obs, deterministic: bool = False) -> State:
        del deterministic
        if isinstance(obs, dict):
            obs = obs["obs"]
        return State(tensors={"latent": obs})

    def transition(
        self, state: State, action, conditions=None, deterministic: bool = False
    ) -> State:
        del action, conditions, deterministic
        return state

    def update(self, state: State, action, obs, conditions=None) -> State:
        del action, conditions
        return self.encode(obs)

    def decode(self, state: State, conditions=None):
        del conditions
        return state

    def loss(self, batch: Batch) -> LossOutput:
        obs = batch.obs
        if isinstance(obs, dict):
            obs = obs["obs"]
        assert isinstance(obs, torch.Tensor)
        pred = self.linear(obs)
        loss = pred.mean()
        return LossOutput(loss=loss, components={"mini": loss})


class _Provider:
    def sample(self, *, batch_size: int, seq_len: int, device: torch.device) -> Batch:
        obs = torch.randn(batch_size, seq_len, 4, device=device)
        actions = torch.randn(batch_size, seq_len, 2, device=device)
        rewards = torch.randn(batch_size, seq_len, device=device)
        terminations = torch.zeros(batch_size, seq_len, device=device)
        next_obs = torch.randn(batch_size, seq_len, 4, device=device)
        return Batch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            next_obs=next_obs,
        )


def test_training_config_effective_values_when_instant_mode_enabled() -> None:
    config = TrainingConfig(
        total_steps=100,
        batch_size=64,
        sequence_length=50,
        instant_mode=True,
        instant_total_steps=5,
        instant_batch_size=3,
        instant_sequence_length=7,
    )
    assert config.effective_total_steps() == 5
    assert config.effective_batch_size() == 3
    assert config.effective_sequence_length() == 7


def test_trainer_instant_mode_sets_ttfi_and_runtime_profile(tmp_path: Path) -> None:
    config = TrainingConfig(
        total_steps=100,
        batch_size=64,
        sequence_length=50,
        instant_mode=True,
        instant_total_steps=3,
        instant_batch_size=2,
        instant_sequence_length=4,
        output_dir=str(tmp_path / "outputs"),
        save_interval=10_000,
        log_interval=10_000,
        device="cpu",
    )
    trainer = Trainer(_MiniModel(), config, callbacks=[])
    trainer.train(_Provider())

    assert trainer.state.global_step == 3
    assert trainer.state.ttfi_sec is not None
    profile = trainer.runtime_profile()
    assert profile["ttfi_sec"] is not None
    assert profile["elapsed_sec"] is not None
    assert profile["throughput_steps_per_sec"] is not None
