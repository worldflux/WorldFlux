"""Tests for eval metric functions."""

from __future__ import annotations

import torch
import torch.nn as nn

from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput, ModelOutput
from worldflux.core.spec import ActionSpec, Capability, ModelIOContract
from worldflux.core.state import State
from worldflux.evals.metrics import (
    imagination_coherence,
    latent_consistency,
    latent_utilization,
    reconstruction_fidelity,
    reward_prediction_accuracy,
)
from worldflux.evals.result import EvalResult


class _MockWorldModel(WorldModel):
    """Minimal mock world model for testing eval metrics."""

    def __init__(self, latent_dim: int = 32, obs_dim: int = 8, action_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._encoder = nn.Linear(obs_dim, latent_dim)
        self._decoder = nn.Linear(latent_dim, obs_dim)
        self._reward_head = nn.Linear(latent_dim, 1)
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.OBS_DECODER,
            Capability.REWARD_PRED,
        }

    def io_contract(self) -> ModelIOContract:
        return ModelIOContract(
            action_spec=ActionSpec(kind="continuous", dim=self.action_dim),
        )

    def encode(self, obs, deterministic: bool = False) -> State:
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)
        latent = self._encoder(obs)
        return State(tensors={"latent": latent})

    def decode(self, state, conditions=None) -> ModelOutput:
        latent = state.tensors["latent"]
        reconstructed = self._decoder(latent)
        reward = self._reward_head(latent)
        return ModelOutput(
            predictions={"obs": reconstructed, "reward": reward},
            state=state,
        )

    def transition(self, state, action, conditions=None, deterministic=False) -> State:
        latent = state.tensors["latent"]
        if isinstance(action, torch.Tensor):
            # Simple linear combination for mock dynamics
            new_latent = latent + 0.01 * action.sum(dim=-1, keepdim=True).expand_as(latent)
        else:
            new_latent = latent + 0.01
        return State(tensors={"latent": new_latent})

    def loss(self, batch) -> LossOutput:
        return LossOutput(loss=torch.tensor(0.0))


class _NoDecoderModel(WorldModel):
    """Mock model that does not support decode."""

    def __init__(self):
        super().__init__()
        self.capabilities = {Capability.LATENT_DYNAMICS}
        self._linear = nn.Linear(8, 16)

    def io_contract(self) -> ModelIOContract:
        return ModelIOContract(
            action_spec=ActionSpec(kind="continuous", dim=4),
        )

    def encode(self, obs, deterministic: bool = False) -> State:
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)
        return State(tensors={"latent": self._linear(obs)})

    def loss(self, batch) -> LossOutput:
        return LossOutput(loss=torch.tensor(0.0))


class TestReconstructionFidelity:
    def test_returns_eval_result(self):
        model = _MockWorldModel()
        obs = torch.randn(4, 8)
        actions = torch.randn(10, 4, 4)
        result = reconstruction_fidelity(model, obs, actions)

        assert isinstance(result, EvalResult)
        assert result.metric == "reconstruction_fidelity"
        assert result.value >= 0.0
        assert result.passed is None  # informational metric

    def test_no_decoder_graceful(self):
        model = _NoDecoderModel()
        obs = torch.randn(4, 8)
        actions = torch.randn(10, 4, 4)
        result = reconstruction_fidelity(model, obs, actions)

        assert isinstance(result, EvalResult)
        assert result.passed is None
        assert "note" in result.metadata


class TestLatentConsistency:
    def test_deterministic_encoding(self):
        model = _MockWorldModel()
        model.eval()
        obs = torch.randn(4, 8)
        actions = torch.randn(10, 4, 4)
        result = latent_consistency(model, obs, actions)

        assert isinstance(result, EvalResult)
        assert result.metric == "latent_consistency"
        assert result.threshold is not None
        # Deterministic linear encoder should produce consistent results
        assert result.passed is True


class TestImaginationCoherence:
    def test_finite_rollout(self):
        model = _MockWorldModel()
        model.eval()
        obs = torch.randn(4, 8)
        actions = torch.randn(10, 4, 4)
        result = imagination_coherence(model, obs, actions, horizon=5)

        assert isinstance(result, EvalResult)
        assert result.metric == "imagination_coherence"
        assert result.passed is True

    def test_custom_horizon(self):
        model = _MockWorldModel()
        model.eval()
        obs = torch.randn(4, 8)
        actions = torch.randn(20, 4, 4)
        result = imagination_coherence(model, obs, actions, horizon=15)

        assert isinstance(result, EvalResult)
        assert result.passed is True


class TestRewardPredictionAccuracy:
    def test_returns_mse(self):
        model = _MockWorldModel()
        model.eval()
        obs = torch.randn(4, 8)
        actions = torch.randn(5, 4, 4)
        rewards = torch.randn(5, 4)
        result = reward_prediction_accuracy(model, obs, actions, rewards)

        assert isinstance(result, EvalResult)
        assert result.metric == "reward_prediction_accuracy"
        assert result.value >= 0.0
        assert result.passed is None  # informational

    def test_no_reward_model_graceful(self):
        model = _NoDecoderModel()
        obs = torch.randn(4, 8)
        actions = torch.randn(5, 4, 4)
        rewards = torch.randn(5, 4)
        result = reward_prediction_accuracy(model, obs, actions, rewards)

        assert isinstance(result, EvalResult)
        assert result.passed is None


class TestLatentUtilization:
    def test_returns_ratio(self):
        model = _MockWorldModel()
        obs = torch.randn(16, 8)  # more samples for variance
        actions = torch.randn(10, 16, 4)
        result = latent_utilization(model, obs, actions)

        assert isinstance(result, EvalResult)
        assert result.metric == "latent_utilization"
        assert 0.0 <= result.value <= 1.0
        assert "active_dims" in result.metadata
        assert "total_dims" in result.metadata
