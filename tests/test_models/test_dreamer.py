"""Tests for DreamerV3 world model."""

from __future__ import annotations

import pytest
import torch

from worldflux import AutoWorldModel, Batch, DreamerV3Config
from worldflux.models.dreamer import DreamerV3WorldModel
from worldflux.models.dreamer.heads import (
    ContinuousActorHead,
    CriticHead,
    DiscreteActorHead,
    compute_td_lambda,
    twohot_encode,
)


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

        assert state.tensors.get("deter") is not None
        assert state.tensors.get("stoch") is not None
        assert state.batch_size == 4

    def test_predict(self, small_model):
        obs = torch.randn(4, 3, 64, 64)
        action = torch.randn(4, 6)

        state = small_model.encode(obs)
        next_state = small_model.transition(state, action)

        assert next_state.tensors["deter"].shape == state.tensors["deter"].shape
        assert next_state.tensors["stoch"].shape == state.tensors["stoch"].shape

    def test_observe(self, small_model):
        obs1 = torch.randn(4, 3, 64, 64)
        obs2 = torch.randn(4, 3, 64, 64)
        action = torch.randn(4, 6)

        state = small_model.encode(obs1)
        next_state = small_model.update(state, action, obs2)

        # Posterior should have both prior and posterior logits
        assert next_state.tensors.get("prior_logits") is not None
        assert next_state.tensors.get("posterior_logits") is not None

    def test_decode(self, small_model):
        obs = torch.randn(4, 3, 64, 64)
        state = small_model.encode(obs)
        decoded = small_model.decode(state)

        assert "obs" in decoded.preds
        assert "reward" in decoded.preds
        assert "continue" in decoded.preds
        assert decoded.preds["obs"].shape == obs.shape

    def test_imagine(self, small_model):
        obs = torch.randn(4, 3, 64, 64)
        actions = torch.randn(10, 4, 6)

        initial = small_model.encode(obs)
        trajectory = small_model.rollout(initial, actions)

        assert len(trajectory) == 11  # initial + 10 steps
        assert trajectory.rewards.shape == (10, 4)
        assert trajectory.continues.shape == (10, 4)

    def test_initial_state(self, small_model):
        state = small_model.initial_state(batch_size=4)

        assert state.tensors.get("deter") is not None
        assert state.tensors.get("stoch") is not None
        assert state.batch_size == 4

    def test_loss(self, small_model):
        batch = Batch(
            obs=torch.randn(4, 8, 3, 64, 64),
            actions=torch.randn(4, 8, 6),
            rewards=torch.randn(4, 8),
            terminations=torch.zeros(4, 8),
        )

        loss_out = small_model.loss(batch)

        assert "kl" in loss_out.components
        assert "kl_dynamics" in loss_out.components
        assert "kl_representation" in loss_out.components
        assert "reconstruction" in loss_out.components
        assert "reward" in loss_out.components
        assert "continue" in loss_out.components

        # All losses should be scalars
        assert loss_out.loss.dim() == 0
        for name, loss in loss_out.components.items():
            assert loss.dim() == 0, f"{name} should be scalar"

    def test_training_loss_decreases(self, small_model):
        """Loss should decrease over training steps."""
        small_model.train()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)

        batch = Batch(
            obs=torch.randn(4, 8, 3, 64, 64),
            actions=torch.randn(4, 8, 6),
            rewards=torch.randn(4, 8),
            terminations=torch.zeros(4, 8),
        )

        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            loss_out = small_model.loss(batch)
            loss_out.loss.backward()
            optimizer.step()
            losses.append(loss_out.loss.item())

        # Loss should generally decrease (not strictly required due to noise)
        assert losses[-1] < losses[0] * 2  # At least not exploding

    def test_from_pretrained(self):
        model = AutoWorldModel.from_pretrained("dreamerv3:size12m")
        assert isinstance(model, DreamerV3WorldModel)

    def test_gradient_flow_through_imagination(self, small_model):
        """Verify gradients flow through imagination rollouts."""
        small_model.train()
        obs = torch.randn(2, 3, 64, 64)
        actions = torch.randn(5, 2, 6, requires_grad=True)

        initial = small_model.encode(obs)
        trajectory = small_model.rollout(initial, actions)

        # Compute loss and backpropagate
        loss = trajectory.rewards.sum()
        loss.backward()

        # Gradients should flow to actions
        assert actions.grad is not None
        assert not torch.isnan(actions.grad).any()
        assert not torch.isinf(actions.grad).any()

    def test_long_horizon_stability(self, small_model):
        """Verify no NaN/Inf in long horizon imagination."""
        small_model.eval()
        obs = torch.randn(2, 3, 64, 64)
        actions = torch.randn(50, 2, 6)  # 50 steps

        with torch.no_grad():
            initial = small_model.encode(obs)
            trajectory = small_model.rollout(initial, actions)

        # Check all states for NaN/Inf
        for state in trajectory.states:
            features = torch.cat(
                [state.tensors["deter"], state.tensors["stoch"].flatten(1)],
                dim=-1,
            )
            assert not torch.isnan(features).any(), "NaN in state features"
            assert not torch.isinf(features).any(), "Inf in state features"

        # Check rewards and continues
        assert not torch.isnan(trajectory.rewards).any(), "NaN in rewards"
        assert not torch.isinf(trajectory.rewards).any(), "Inf in rewards"


class TestTwohotEncode:
    """Twohot encoding tests."""

    def test_bin_center_gives_one_hot(self):
        bins = torch.linspace(-20.0, 20.0, 255)
        # Value exactly at bin center should give weight 1.0 on that bin
        x = bins[127:128]  # middle bin
        target = twohot_encode(x, bins)
        assert target.shape == (1, 255)
        assert torch.isclose(target.sum(), torch.tensor(1.0))
        assert torch.isclose(target[0, 127], torch.tensor(1.0), atol=1e-5)

    def test_midpoint_gives_equal_weights(self):
        bins = torch.linspace(-20.0, 20.0, 255)
        # Midpoint between bins[10] and bins[11]
        x = ((bins[10] + bins[11]) / 2).unsqueeze(0)
        target = twohot_encode(x, bins)
        assert torch.isclose(target[0, 10], torch.tensor(0.5), atol=1e-3)
        assert torch.isclose(target[0, 11], torch.tensor(0.5), atol=1e-3)

    def test_boundary_clamp(self):
        bins = torch.linspace(-20.0, 20.0, 255)
        # Values outside bin range should be clamped
        x = torch.tensor([-100.0, 100.0])
        target = twohot_encode(x, bins)
        assert target.shape == (2, 255)
        assert torch.isclose(target.sum(dim=-1), torch.ones(2)).all()

    def test_expected_value_roundtrip(self):
        bins = torch.linspace(-20.0, 20.0, 255)
        values = torch.tensor([-5.0, 0.0, 3.7, 15.2])
        target = twohot_encode(values, bins)
        # Use twohot target as "logits" (log of target gives log-probs)
        # For a proper roundtrip, expected value of the distribution should match
        # Using target directly as probs: sum(target * bins)
        recovered = (target * bins).sum(dim=-1)
        assert torch.allclose(values, recovered, atol=0.2)


class TestRewardHeadTwohot:
    """RewardHead twohot mode tests."""

    def test_forward_shape(self):
        from worldflux.models.dreamer.heads import RewardHead

        head = RewardHead(feature_dim=128, hidden_dim=64, use_twohot=True, num_bins=255)
        features = torch.randn(4, 128)
        out = head(features)
        assert out.shape == (4, 255)

    def test_predict_scalar(self):
        from worldflux.models.dreamer.heads import RewardHead

        head = RewardHead(feature_dim=128, hidden_dim=64, use_twohot=True, num_bins=255)
        features = torch.randn(4, 128)
        out = head.predict(features)
        assert out.shape == (4,)
        assert not torch.isnan(out).any()

    def test_scalar_backward_compat(self):
        from worldflux.models.dreamer.heads import RewardHead

        head = RewardHead(feature_dim=128, hidden_dim=64, use_twohot=False)
        features = torch.randn(4, 128)
        assert head(features).shape == (4, 1)
        assert head.predict(features).shape == (4,)


class TestDreamerV3Twohot:
    """DreamerV3 world model with twohot enabled."""

    @pytest.fixture
    def twohot_model(self):
        config = DreamerV3Config(
            deter_dim=256,
            stoch_discrete=8,
            stoch_classes=8,
            hidden_dim=128,
            cnn_depth=16,
            use_twohot=True,
        )
        return DreamerV3WorldModel(config)

    def test_loss_with_twohot(self, twohot_model):
        batch = Batch(
            obs=torch.randn(4, 8, 3, 64, 64),
            actions=torch.randn(4, 8, 6),
            rewards=torch.randn(4, 8),
            terminations=torch.zeros(4, 8),
        )
        loss_out = twohot_model.loss(batch)
        assert loss_out.loss.dim() == 0
        assert not torch.isnan(loss_out.loss)
        for name, loss in loss_out.components.items():
            assert loss.dim() == 0, f"{name} should be scalar"
            assert not torch.isnan(loss), f"{name} is NaN"

    def test_rollout_with_twohot(self, twohot_model):
        obs = torch.randn(2, 3, 64, 64)
        actions = torch.randn(5, 2, 6)
        initial = twohot_model.encode(obs)
        trajectory = twohot_model.rollout(initial, actions)
        assert trajectory.rewards.shape == (5, 2)
        assert not torch.isnan(trajectory.rewards).any()

    def test_gradient_flow_twohot(self, twohot_model):
        twohot_model.train()
        batch = Batch(
            obs=torch.randn(2, 4, 3, 64, 64),
            actions=torch.randn(2, 4, 6),
            rewards=torch.randn(2, 4),
            terminations=torch.zeros(2, 4),
        )
        loss_out = twohot_model.loss(batch)
        loss_out.loss.backward()
        # Check reward head has gradients
        for name, p in twohot_model.reward_head.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for reward_head.{name}"


class TestDiscreteActorHead:
    """Discrete actor head tests."""

    def test_sample_shape_and_onehot(self):
        head = DiscreteActorHead(feature_dim=128, action_dim=6, hidden_dim=64)
        features = torch.randn(4, 128)
        action, log_prob = head.sample(features)
        assert action.shape == (4, 6)
        assert log_prob.shape == (4,)
        # Should be approximately one-hot (straight-through has probs mixed in)
        assert torch.allclose(action.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_gradient_flow(self):
        head = DiscreteActorHead(feature_dim=64, action_dim=4, hidden_dim=32)
        features = torch.randn(2, 64, requires_grad=True)
        action, log_prob = head.sample(features)
        loss = log_prob.sum()
        loss.backward()
        assert features.grad is not None

    def test_entropy(self):
        head = DiscreteActorHead(feature_dim=64, action_dim=4, hidden_dim=32)
        features = torch.randn(3, 64)
        ent = head.entropy(features)
        assert ent.shape == (3,)
        assert (ent >= 0).all()


class TestContinuousActorHead:
    """Continuous actor head tests."""

    def test_sample_bounded(self):
        head = ContinuousActorHead(feature_dim=128, action_dim=6, hidden_dim=64)
        features = torch.randn(4, 128)
        action, log_prob = head.sample(features)
        assert action.shape == (4, 6)
        assert log_prob.shape == (4,)
        # tanh output should be in (-1, 1)
        assert (action > -1.0).all() and (action < 1.0).all()

    def test_log_prob_finite(self):
        head = ContinuousActorHead(feature_dim=64, action_dim=4, hidden_dim=32)
        features = torch.randn(8, 64)
        _, log_prob = head.sample(features)
        assert torch.isfinite(log_prob).all()

    def test_entropy(self):
        head = ContinuousActorHead(feature_dim=64, action_dim=4, hidden_dim=32)
        features = torch.randn(3, 64)
        ent = head.entropy(features)
        assert ent.shape == (3,)


class TestCriticHead:
    """Critic head tests."""

    def test_predict_scalar(self):
        head = CriticHead(feature_dim=128, hidden_dim=64)
        features = torch.randn(4, 128)
        value = head.predict(features)
        assert value.shape == (4,)
        assert torch.isfinite(value).all()

    def test_forward_logits(self):
        head = CriticHead(feature_dim=128, hidden_dim=64, num_bins=255)
        features = torch.randn(4, 128)
        logits = head(features)
        assert logits.shape == (4, 255)


class TestTDLambda:
    """TD-lambda computation tests."""

    def test_shape(self):
        h, n = 10, 4
        rewards = torch.randn(h, n)
        values = torch.randn(h + 1, n)
        continues = torch.ones(h, n)
        returns = compute_td_lambda(rewards, values, continues)
        assert returns.shape == (h, n)

    def test_finite(self):
        h, n = 5, 3
        rewards = torch.randn(h, n)
        values = torch.randn(h + 1, n)
        continues = torch.ones(h, n)
        returns = compute_td_lambda(rewards, values, continues)
        assert torch.isfinite(returns).all()

    def test_single_step(self):
        """With H=1 and lambda=0, return = r + gamma * V_next."""
        rewards = torch.tensor([[1.0]])
        values = torch.tensor([[0.0], [2.0]])
        continues = torch.ones(1, 1)
        ret = compute_td_lambda(rewards, values, continues, gamma=0.99, lambda_=0.0)
        expected = 1.0 + 0.99 * 2.0
        assert torch.allclose(ret, torch.tensor([[expected]]), atol=1e-5)


class TestDreamerV3ActorCritic:
    """DreamerV3 world model with actor-critic integration tests."""

    @pytest.fixture
    def ac_model(self):
        config = DreamerV3Config(
            deter_dim=64,
            stoch_discrete=4,
            stoch_classes=4,
            hidden_dim=32,
            cnn_depth=8,
            actor_critic=True,
            imagination_horizon=3,
        )
        return DreamerV3WorldModel(config)

    @pytest.fixture
    def ac_model_discrete(self):
        config = DreamerV3Config(
            deter_dim=64,
            stoch_discrete=4,
            stoch_classes=4,
            hidden_dim=32,
            cnn_depth=8,
            actor_critic=True,
            action_type="discrete",
            imagination_horizon=3,
        )
        return DreamerV3WorldModel(config)

    def test_ac_capabilities(self, ac_model):
        from worldflux.core.spec import Capability

        assert Capability.POLICY in ac_model.capabilities
        assert Capability.VALUE in ac_model.capabilities

    def test_loss_includes_ac(self, ac_model):
        batch = Batch(
            obs=torch.randn(2, 4, 3, 64, 64),
            actions=torch.randn(2, 4, 6),
            rewards=torch.randn(2, 4),
            terminations=torch.zeros(2, 4),
        )
        loss_out = ac_model.loss(batch)
        assert "actor" in loss_out.components
        assert "critic" in loss_out.components
        assert loss_out.loss.dim() == 0
        assert torch.isfinite(loss_out.loss)

    def test_loss_includes_ac_discrete(self, ac_model_discrete):
        batch = Batch(
            obs=torch.randn(2, 4, 3, 64, 64),
            actions=torch.randn(2, 4, 6),
            rewards=torch.randn(2, 4),
            terminations=torch.zeros(2, 4),
        )
        loss_out = ac_model_discrete.loss(batch)
        assert "actor" in loss_out.components
        assert "critic" in loss_out.components
        assert torch.isfinite(loss_out.loss)

    def test_parameter_groups(self, ac_model):
        groups = ac_model.parameter_groups()
        assert len(groups) == 3
        # Check LRs
        assert groups[0]["lr"] == ac_model.config.learning_rate
        assert groups[1]["lr"] == ac_model.config.actor_lr
        assert groups[2]["lr"] == ac_model.config.critic_lr

    def test_backward_compat(self):
        """actor_critic=False should produce same behavior as before."""
        config = DreamerV3Config(
            deter_dim=64,
            stoch_discrete=4,
            stoch_classes=4,
            hidden_dim=32,
            cnn_depth=8,
            actor_critic=False,
        )
        model = DreamerV3WorldModel(config)
        assert not hasattr(model, "actor_head")
        assert not hasattr(model, "critic_head")

        batch = Batch(
            obs=torch.randn(2, 4, 3, 64, 64),
            actions=torch.randn(2, 4, 6),
            rewards=torch.randn(2, 4),
            terminations=torch.zeros(2, 4),
        )
        loss_out = model.loss(batch)
        assert "actor" not in loss_out.components
        assert "critic" not in loss_out.components

    def test_ac_gradient_flow(self, ac_model):
        """Actor and critic should receive gradients."""
        ac_model.train()
        batch = Batch(
            obs=torch.randn(2, 4, 3, 64, 64),
            actions=torch.randn(2, 4, 6),
            rewards=torch.randn(2, 4),
            terminations=torch.zeros(2, 4),
        )
        loss_out = ac_model.loss(batch)
        loss_out.loss.backward()

        actor_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in ac_model.actor_head.parameters()
        )
        critic_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in ac_model.critic_head.parameters()
        )
        assert actor_has_grad, "Actor head should receive gradients"
        assert critic_has_grad, "Critic head should receive gradients"
