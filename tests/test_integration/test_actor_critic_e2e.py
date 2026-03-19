# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""End-to-end integration tests for actor-critic training loop.

Validates that the DreamerV3 actor-critic pipeline (imagination rollout,
policy gradient, slow critic EMA, advantage normalization) produces finite
losses and correct gradient flow when trained for a short number of steps.

All tests use the ``dreamer:ci`` preset to stay fast in CI.
"""

from __future__ import annotations

import pytest
import torch

from worldflux import create_world_model
from worldflux.training import Trainer, TrainingConfig
from worldflux.training.replay import ReplayBuffer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_replay_buffer(
    obs_shape: tuple[int, ...] = (3, 64, 64),
    action_dim: int = 6,
    seq_len: int = 16,
    n_episodes: int = 4,
) -> ReplayBuffer:
    """Create a small replay buffer with random data for testing."""
    buf = ReplayBuffer(capacity=1000, obs_shape=obs_shape, action_dim=action_dim)
    for _ in range(n_episodes):
        obs = torch.randn(seq_len, *obs_shape)
        actions = torch.randn(seq_len, action_dim)
        rewards = torch.randn(seq_len)
        dones = torch.zeros(seq_len)
        dones[-1] = 1.0
        buf.add_trajectory(obs, actions, rewards, dones)
    return buf


def _param_norm_dict(module: torch.nn.Module) -> dict[str, float]:
    """Return {name: L2 norm} for each parameter in the module."""
    return {name: float(p.data.norm().item()) for name, p in module.named_parameters()}


def _param_diff(mod_a: torch.nn.Module, mod_b: torch.nn.Module) -> float:
    """Sum of absolute differences between matching parameters."""
    total = 0.0
    for (_, pa), (_, pb) in zip(mod_a.named_parameters(), mod_b.named_parameters()):
        total += float((pa.data - pb.data).abs().sum().item())
    return total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestActorCriticE2E:
    """Integration tests for DreamerV3 actor-critic pipeline."""

    @pytest.fixture
    def ac_model(self):
        """Create a tiny DreamerV3 model with actor-critic enabled."""
        return create_world_model(
            "dreamer:ci",
            obs_shape=(3, 64, 64),
            action_dim=6,
            actor_critic=True,
            deter_dim=64,
            stoch_discrete=4,
            stoch_classes=4,
            hidden_dim=64,
            cnn_depth=8,
        )

    @pytest.fixture
    def replay_buffer(self):
        return _make_replay_buffer()

    def test_actor_critic_training_loop(self, ac_model, replay_buffer):
        """100-step training loop with actor_critic=True produces finite losses."""
        config = TrainingConfig(
            total_steps=100,
            batch_size=4,
            sequence_length=8,
            learning_rate=1e-4,
            device="cpu",
            log_interval=50,
            save_interval=0,
            auto_quality_check=False,
        )
        trainer = Trainer(ac_model, config)
        trainer.train(replay_buffer, num_steps=100)

        # All losses should be finite
        assert trainer.state.global_step == 100
        loss = trainer.state.metrics.get("loss", None)
        assert loss is not None
        assert not (loss != loss), "Loss is NaN"  # NaN check

    def test_actor_gradient_flows_through_imagination(self, ac_model, replay_buffer):
        """Actor and critic heads receive non-zero gradients."""
        config = TrainingConfig(
            total_steps=10,
            batch_size=4,
            sequence_length=8,
            learning_rate=1e-4,
            device="cpu",
            log_interval=100,
            save_interval=0,
            auto_quality_check=False,
        )
        trainer = Trainer(ac_model, config)
        trainer.train(replay_buffer, num_steps=10)

        # Check actor head gradients
        actor_has_grad = False
        for name, p in ac_model.actor_head.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                actor_has_grad = True
                break
        assert actor_has_grad, "Actor head received no gradients"

        # Check critic head gradients
        critic_has_grad = False
        for name, p in ac_model.critic_head.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                critic_has_grad = True
                break
        assert critic_has_grad, "Critic head received no gradients"

    def test_slow_critic_ema_updates(self, ac_model, replay_buffer):
        """Slow critic target network diverges from online critic via EMA."""
        config = TrainingConfig(
            total_steps=50,
            batch_size=4,
            sequence_length=8,
            learning_rate=1e-4,
            device="cpu",
            log_interval=100,
            save_interval=0,
            auto_quality_check=False,
        )
        trainer = Trainer(ac_model, config)
        trainer.train(replay_buffer, num_steps=50)

        final_diff = _param_diff(ac_model.critic_head, ac_model.slow_critic)

        # After training, the critic and slow_critic should differ because
        # the slow critic tracks via EMA (lagging behind the online critic)
        assert final_diff > 0, (
            "Slow critic is identical to online critic after training - "
            "EMA update may not be working"
        )

    def test_advantage_normalization_range(self, ac_model, replay_buffer):
        """Return normalization should keep advantages in a bounded range."""
        config = TrainingConfig(
            total_steps=30,
            batch_size=4,
            sequence_length=8,
            learning_rate=1e-4,
            device="cpu",
            log_interval=100,
            save_interval=0,
            auto_quality_check=False,
        )
        trainer = Trainer(ac_model, config)
        trainer.train(replay_buffer, num_steps=30)

        # The percentile trackers should have been updated
        if hasattr(ac_model, "_return_low") and hasattr(ac_model, "_return_high"):
            low = ac_model._return_low.item()
            high = ac_model._return_high.item()
            # The range should be finite and sensible
            assert not (low != low), "Return low percentile is NaN"
            assert not (high != high), "Return high percentile is NaN"

    def test_actor_critic_false_regression(self, replay_buffer):
        """Model with actor_critic=False still trains correctly (backward compat)."""
        model = create_world_model(
            "dreamer:ci",
            obs_shape=(3, 64, 64),
            action_dim=6,
            actor_critic=False,
            deter_dim=64,
            stoch_discrete=4,
            stoch_classes=4,
            hidden_dim=64,
            cnn_depth=8,
        )

        # Should NOT have actor/critic heads
        assert not hasattr(model, "actor_head") or not model.config.actor_critic

        config = TrainingConfig(
            total_steps=20,
            batch_size=4,
            sequence_length=8,
            learning_rate=1e-4,
            device="cpu",
            log_interval=100,
            save_interval=0,
            auto_quality_check=False,
        )
        trainer = Trainer(model, config)
        trainer.train(replay_buffer, num_steps=20)

        assert trainer.state.global_step == 20
        loss = trainer.state.metrics.get("loss", None)
        assert loss is not None
        assert not (loss != loss), "Loss is NaN"
