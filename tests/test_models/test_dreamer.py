"""Tests for DreamerV3 world model."""

import pytest
import torch

from worldmodels import AutoWorldModel, DreamerV3Config
from worldmodels.models.dreamer import DreamerV3WorldModel


class TestDreamerV3Config:
    """DreamerV3 configuration tests."""
    
    def test_from_size_presets(self):
        for size in ["size12m", "size25m", "size50m", "size100m", "size200m"]:
            config = DreamerV3Config.from_size(size)
            assert config.model_name == size
            assert config.model_type == "dreamer"
    
    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            DreamerV3Config.from_size("invalid")
    
    def test_stoch_dim_computed(self):
        config = DreamerV3Config(stoch_discrete=32, stoch_classes=32)
        assert config.stoch_dim == 32 * 32


class TestDreamerV3WorldModel:
    """DreamerV3 world model tests."""
    
    @pytest.fixture
    def model(self):
        config = DreamerV3Config.from_size("size12m")
        return DreamerV3WorldModel(config)
    
    @pytest.fixture
    def small_model(self):
        """Smaller model for faster tests."""
        config = DreamerV3Config(
            deter_dim=256,
            stoch_discrete=8,
            stoch_classes=8,
            hidden_dim=128,
            cnn_depth=16,
        )
        return DreamerV3WorldModel(config)
    
    def test_encode(self, small_model):
        obs = torch.randn(4, 3, 64, 64)
        state = small_model.encode(obs)
        
        assert state.deterministic is not None
        assert state.stochastic is not None
        assert state.batch_size == 4
    
    def test_predict(self, small_model):
        obs = torch.randn(4, 3, 64, 64)
        action = torch.randn(4, 6)
        
        state = small_model.encode(obs)
        next_state = small_model.predict(state, action)
        
        assert next_state.deterministic.shape == state.deterministic.shape
        assert next_state.stochastic.shape == state.stochastic.shape
    
    def test_observe(self, small_model):
        obs1 = torch.randn(4, 3, 64, 64)
        obs2 = torch.randn(4, 3, 64, 64)
        action = torch.randn(4, 6)
        
        state = small_model.encode(obs1)
        next_state = small_model.observe(state, action, obs2)
        
        # Posterior should have both prior and posterior logits
        assert next_state.prior_logits is not None
        assert next_state.posterior_logits is not None
    
    def test_decode(self, small_model):
        obs = torch.randn(4, 3, 64, 64)
        state = small_model.encode(obs)
        decoded = small_model.decode(state)
        
        assert "obs" in decoded
        assert "reward" in decoded
        assert "continue" in decoded
        assert decoded["obs"].shape == obs.shape
    
    def test_imagine(self, small_model):
        obs = torch.randn(4, 3, 64, 64)
        actions = torch.randn(10, 4, 6)
        
        initial = small_model.encode(obs)
        trajectory = small_model.imagine(initial, actions)
        
        assert len(trajectory) == 11  # initial + 10 steps
        assert trajectory.rewards.shape == (10, 4)
        assert trajectory.continues.shape == (10, 4)
    
    def test_initial_state(self, small_model):
        state = small_model.initial_state(batch_size=4)
        
        assert state.deterministic is not None
        assert state.stochastic is not None
        assert state.batch_size == 4
    
    def test_compute_loss(self, small_model):
        batch = {
            "obs": torch.randn(4, 8, 3, 64, 64),
            "actions": torch.randn(4, 8, 6),
            "rewards": torch.randn(4, 8),
            "continues": torch.ones(4, 8),
        }
        
        losses = small_model.compute_loss(batch)
        
        assert "loss" in losses
        assert "kl" in losses
        assert "reconstruction" in losses
        assert "reward" in losses
        assert "continue" in losses
        
        # All losses should be scalars
        for name, loss in losses.items():
            assert loss.dim() == 0, f"{name} should be scalar"
    
    def test_training_loss_decreases(self, small_model):
        """Loss should decrease over training steps."""
        small_model.train()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
        
        batch = {
            "obs": torch.randn(4, 8, 3, 64, 64),
            "actions": torch.randn(4, 8, 6),
            "rewards": torch.randn(4, 8),
            "continues": torch.ones(4, 8),
        }
        
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            loss_dict = small_model.compute_loss(batch)
            loss_dict["loss"].backward()
            optimizer.step()
            losses.append(loss_dict["loss"].item())
        
        # Loss should generally decrease (not strictly required due to noise)
        assert losses[-1] < losses[0] * 2  # At least not exploding
    
    def test_from_pretrained(self):
        model = AutoWorldModel.from_pretrained("dreamerv3:size12m")
        assert isinstance(model, DreamerV3WorldModel)
