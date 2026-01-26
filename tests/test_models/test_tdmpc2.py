"""Tests for TD-MPC2 world model."""

import pytest
import torch

from worldmodels import AutoWorldModel, TDMPC2Config
from worldmodels.models.tdmpc2 import TDMPC2WorldModel


class TestTDMPC2Config:
    """TD-MPC2 configuration tests."""

    def test_from_size_presets(self):
        for size in ["5m", "19m", "48m", "317m"]:
            config = TDMPC2Config.from_size(size)
            assert config.model_name == size
            assert config.model_type == "tdmpc2"

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            TDMPC2Config.from_size("invalid")


class TestTDMPC2WorldModel:
    """TD-MPC2 world model tests."""

    @pytest.fixture
    def model(self):
        config = TDMPC2Config.from_size("5m")
        config.obs_shape = (39,)  # Vector observation
        return TDMPC2WorldModel(config)

    @pytest.fixture
    def small_model(self):
        """Smaller model for faster tests."""
        config = TDMPC2Config(
            latent_dim=128,
            hidden_dim=128,
            obs_shape=(39,),
            num_q_networks=2,
        )
        return TDMPC2WorldModel(config)

    def test_simnorm_constraint(self, small_model):
        """SimNorm constraint should be satisfied."""
        obs = torch.randn(4, 39)
        state = small_model.encode(obs)

        # Check SimNorm (each simplex sums to 1)
        reshaped = state.deterministic.view(4, -1, 8)
        sums = reshaped.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_encode(self, small_model):
        obs = torch.randn(4, 39)
        state = small_model.encode(obs)

        assert state.deterministic is not None
        assert state.stochastic is None  # TD-MPC2 has no stochastic
        assert state.latent_type == "simnorm"

    def test_predict(self, small_model):
        obs = torch.randn(4, 39)
        action = torch.randn(4, 6)

        state = small_model.encode(obs)
        next_state = small_model.predict(state, action)

        assert next_state.deterministic.shape == state.deterministic.shape

    def test_observe(self, small_model):
        """TD-MPC2 observe just encodes (no posterior)."""
        obs1 = torch.randn(4, 39)
        obs2 = torch.randn(4, 39)
        action = torch.randn(4, 6)

        state = small_model.encode(obs1)
        next_state = small_model.observe(state, action, obs2)

        # Should just be encoded obs2
        direct_encode = small_model.encode(obs2)
        assert torch.allclose(next_state.deterministic, direct_encode.deterministic)

    def test_decode_no_obs(self, small_model):
        """TD-MPC2 has no decoder for observations."""
        obs = torch.randn(4, 39)
        state = small_model.encode(obs)
        decoded = small_model.decode(state)

        assert "obs" not in decoded or decoded.get("obs") is None
        assert "reward" in decoded
        assert "q_values" in decoded
        assert "action" in decoded

    def test_imagine(self, small_model):
        obs = torch.randn(4, 39)
        actions = torch.randn(5, 4, 6)

        initial = small_model.encode(obs)
        trajectory = small_model.imagine(initial, actions)

        assert len(trajectory) == 6  # initial + 5 steps
        assert trajectory.rewards.shape == (5, 4)

    def test_predict_q(self, small_model):
        obs = torch.randn(4, 39)
        action = torch.randn(4, 6)

        state = small_model.encode(obs)
        q_values = small_model.predict_q(state, action)

        assert q_values.shape == (2, 4)  # num_q_networks=2

    def test_predict_reward(self, small_model):
        obs = torch.randn(4, 39)
        action = torch.randn(4, 6)

        state = small_model.encode(obs)
        reward = small_model.predict_reward(state, action)

        assert reward.shape == (4,)

    def test_initial_state(self, small_model):
        state = small_model.initial_state(batch_size=4)

        assert state.deterministic is not None
        assert state.batch_size == 4

        # Should be uniform SimNorm
        reshaped = state.deterministic.view(4, -1, 8)
        sums = reshaped.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_compute_loss(self, small_model):
        batch = {
            "obs": torch.randn(4, 8, 39),
            "actions": torch.randn(4, 8, 6),
            "rewards": torch.randn(4, 8),
        }

        losses = small_model.compute_loss(batch)

        assert "loss" in losses
        assert "consistency" in losses
        assert "reward" in losses
        assert "td" in losses

        for name, loss in losses.items():
            assert loss.dim() == 0, f"{name} should be scalar"

    def test_from_pretrained(self):
        model = AutoWorldModel.from_pretrained("tdmpc2:5m", obs_shape=(39,))
        assert isinstance(model, TDMPC2WorldModel)
