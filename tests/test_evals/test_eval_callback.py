"""Tests for EvalCallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput, ModelOutput
from worldflux.core.spec import ActionSpec, Capability, ModelIOContract
from worldflux.core.state import State
from worldflux.training.callbacks import EvalCallback


class _MockWorldModel(WorldModel):
    """Minimal mock for callback tests."""

    def __init__(self):
        super().__init__()
        self._encoder = nn.Linear(8, 32)
        self._decoder = nn.Linear(32, 8)
        self._reward_head = nn.Linear(32, 1)
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.OBS_DECODER,
            Capability.REWARD_PRED,
        }
        self.config = type("Config", (), {"obs_shape": (8,), "action_dim": 4})()

    def io_contract(self) -> ModelIOContract:
        return ModelIOContract(
            action_spec=ActionSpec(kind="continuous", dim=4),
        )

    def encode(self, obs, deterministic: bool = False) -> State:
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)
        return State(tensors={"latent": self._encoder(obs)})

    def decode(self, state, conditions=None) -> ModelOutput:
        latent = state.tensors["latent"]
        return ModelOutput(
            predictions={
                "obs": self._decoder(latent),
                "reward": self._reward_head(latent),
            },
            state=state,
        )

    def transition(self, state, action, conditions=None, deterministic=False) -> State:
        latent = state.tensors["latent"]
        if isinstance(action, torch.Tensor):
            new_latent = latent + 0.01 * action.sum(dim=-1, keepdim=True).expand_as(latent)
        else:
            new_latent = latent + 0.01
        return State(tensors={"latent": new_latent})

    def loss(self, batch) -> LossOutput:
        return LossOutput(loss=torch.tensor(0.0))


def _make_trainer_mock(model, step: int = 0):
    """Create a mock trainer with the given model and step."""
    trainer = MagicMock()
    trainer.model = model
    trainer.state.global_step = step
    trainer.state.epoch = 0
    trainer.state.ttfi_sec = 0.0
    trainer.state.train_start_time = None
    return trainer


class TestEvalCallback:
    def test_init_validation(self):
        with pytest.raises(ValueError, match="eval_interval must be positive"):
            EvalCallback(eval_interval=0)

    def test_skips_non_interval_steps(self):
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        trainer = _make_trainer_mock(model, step=50)

        with patch("worldflux.training.callbacks.write_event") as mock_write:
            cb.on_step_end(trainer)
            mock_write.assert_not_called()

    def test_runs_at_interval(self):
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        trainer = _make_trainer_mock(model, step=100)
        cb.on_train_begin(trainer)

        with patch("worldflux.training.callbacks.write_event") as mock_write:
            cb.on_step_end(trainer)
            mock_write.assert_called_once()
            call_kwargs = mock_write.call_args[1]
            assert call_kwargs["event"] == "eval.quick"
            assert call_kwargs["step"] == 100

    def test_skips_step_zero(self):
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        trainer = _make_trainer_mock(model, step=0)

        with patch("worldflux.training.callbacks.write_event") as mock_write:
            cb.on_step_end(trainer)
            mock_write.assert_not_called()
