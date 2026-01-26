"""Numerical stability tests for WorldLoom.

These tests verify that mathematical operations remain stable
across extreme input ranges to prevent NaN/Inf propagation.
"""

import torch

from worldloom.models.dreamer.heads import symexp, symlog


class TestSymlogSymexp:
    """Test symlog/symexp numerical stability."""

    def test_symexp_overflow_prevention(self):
        """symexp should not overflow for large positive inputs."""
        # exp(89) > float32 max, so inputs > ~88 would overflow without protection
        large_values = torch.tensor([88.0, 89.0, 100.0, 500.0, 1000.0])
        result = symexp(large_values)

        assert not torch.isnan(result).any(), f"symexp produced NaN: {result}"
        assert not torch.isinf(result).any(), f"symexp produced Inf: {result}"
        # All results should be positive (sign of input is positive)
        assert (result > 0).all(), f"symexp should preserve sign: {result}"

    def test_symexp_negative_large_values(self):
        """symexp should not overflow for large negative inputs."""
        large_neg_values = torch.tensor([-88.0, -89.0, -100.0, -500.0, -1000.0])
        result = symexp(large_neg_values)

        assert not torch.isnan(result).any(), f"symexp produced NaN: {result}"
        assert not torch.isinf(result).any(), f"symexp produced Inf: {result}"
        # All results should be negative (sign of input is negative)
        assert (result < 0).all(), f"symexp should preserve sign: {result}"

    def test_symexp_zero(self):
        """symexp(0) should equal 0."""
        result = symexp(torch.tensor([0.0]))
        torch.testing.assert_close(result, torch.tensor([0.0]))

    def test_symexp_small_values(self):
        """symexp should be accurate for small values."""
        x = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
        result = symexp(x)

        # For small positive x: symexp(x) = exp(x) - 1
        expected = torch.exp(x) - 1
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_symlog_symexp_inverse_normal_range(self):
        """symlog and symexp should be inverses for normal values."""
        x = torch.linspace(-50, 50, 101)
        roundtrip = symexp(symlog(x))
        torch.testing.assert_close(roundtrip, x, rtol=1e-4, atol=1e-4)

    def test_symlog_symexp_inverse_positive(self):
        """Forward: symlog then symexp should recover original for positive values."""
        x = torch.tensor([0.001, 0.1, 1.0, 10.0, 100.0, 1000.0])
        roundtrip = symexp(symlog(x))
        torch.testing.assert_close(roundtrip, x, rtol=1e-4, atol=1e-4)

    def test_symlog_symexp_inverse_negative(self):
        """Forward: symlog then symexp should recover original for negative values."""
        x = torch.tensor([-0.001, -0.1, -1.0, -10.0, -100.0, -1000.0])
        roundtrip = symexp(symlog(x))
        torch.testing.assert_close(roundtrip, x, rtol=1e-4, atol=1e-4)

    def test_symlog_bounded_output(self):
        """symlog should produce bounded output for any finite input."""
        x = torch.tensor([1e10, 1e20, 1e30, -1e10, -1e20, -1e30])
        result = symlog(x)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_symexp_gradient_stability(self):
        """symexp should have stable gradients for normal inputs."""
        x = torch.tensor([0.0, 1.0, 5.0, 10.0, 50.0], requires_grad=True)
        result = symexp(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any(), f"Gradient contains NaN: {x.grad}"
        assert not torch.isinf(x.grad).any(), f"Gradient contains Inf: {x.grad}"

    def test_symexp_gradient_large_inputs(self):
        """symexp should have stable gradients even for large inputs (clamped region)."""
        x = torch.tensor([100.0, 200.0, 500.0], requires_grad=True)
        result = symexp(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        # In clamped region, gradient should be zero (due to clamp)
        # Actually, clamp has zero gradient outside the range
        assert not torch.isnan(x.grad).any()

    def test_batch_numerical_stability(self):
        """Test numerical stability with batched inputs."""
        # Mix of normal and extreme values
        batch = torch.tensor(
            [
                [0.0, 1.0, 10.0],
                [50.0, 100.0, 200.0],
                [-50.0, -100.0, -200.0],
            ]
        )

        result = symexp(batch)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestGradientFlow:
    """Test gradient flow through world model components."""

    def test_dreamer_gradient_flow(self):
        """Verify gradients flow through DreamerV3 encode-decode."""
        from worldloom.core.config import DreamerV3Config
        from worldloom.models.dreamer.world_model import DreamerV3WorldModel

        config = DreamerV3Config(
            obs_shape=(3, 64, 64),
            action_dim=4,
            deter_dim=128,
            hidden_dim=64,
            cnn_depth=16,
        )
        model = DreamerV3WorldModel(config)

        obs = torch.randn(2, 3, 64, 64, requires_grad=True)
        state = model.encode(obs)
        decoded = model.decode(state)

        # Backprop through reconstruction
        loss = decoded["obs"].mean()
        loss.backward()

        assert obs.grad is not None
        assert not torch.isnan(obs.grad).any()

    def test_tdmpc2_gradient_flow(self):
        """Verify gradients flow through TD-MPC2 encode-predict."""
        from worldloom.core.config import TDMPC2Config
        from worldloom.models.tdmpc2.world_model import TDMPC2WorldModel

        config = TDMPC2Config(
            obs_shape=(39,),
            action_dim=6,
            latent_dim=64,
            hidden_dim=64,
        )
        model = TDMPC2WorldModel(config)

        obs = torch.randn(2, 39, requires_grad=True)
        action = torch.randn(2, 6)

        state = model.encode(obs)
        next_state = model.predict(state, action)
        reward = model.predict_reward(next_state, action)

        loss = reward.mean()
        loss.backward()

        assert obs.grad is not None
        assert not torch.isnan(obs.grad).any()
