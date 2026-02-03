"""Edge case tests for WorldFlux OSS robustness."""

import pytest
import torch

from worldflux import Batch
from worldflux.core.config import DreamerV3Config, TDMPC2Config
from worldflux.core.state import State
from worldflux.models.dreamer.heads import symexp, symlog


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_symexp_large_values_no_overflow(self):
        """symexp should handle large values without NaN/Inf."""
        x = torch.tensor([100.0, 200.0, 500.0, -100.0, -200.0])
        result = symexp(x)

        assert not torch.isnan(result).any(), "symexp produced NaN for large values"
        assert not torch.isinf(result).any(), "symexp produced Inf for large values"

    def test_symexp_normal_range(self):
        """symexp should work correctly for normal values."""
        x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = symexp(x)

        # Verify inverse relationship with symlog for normal values
        roundtrip = symlog(result)
        torch.testing.assert_close(roundtrip, x, atol=1e-5, rtol=1e-5)

    def test_symlog_symexp_roundtrip(self):
        """symlog and symexp should be inverses for reasonable values."""
        x = torch.randn(100) * 10  # Random values in [-30, 30] range
        result = symexp(symlog(x))
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_nan_input_handling(self):
        """Models should handle NaN input gracefully or raise clear errors."""
        from worldflux.models.dreamer.world_model import DreamerV3WorldModel

        config = DreamerV3Config(
            obs_shape=(3, 64, 64),
            action_dim=4,
            deter_dim=128,
            hidden_dim=64,
            cnn_depth=16,
        )
        model = DreamerV3WorldModel(config)

        # NaN observation - model may produce NaN output but shouldn't crash
        nan_obs = torch.full((2, 3, 64, 64), float("nan"))
        try:
            state = model.encode(nan_obs)
            # If it doesn't crash, check that output exists
            assert state is not None
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for NaN input
            pass

    def test_inf_input_handling(self):
        """Models should handle Inf input gracefully or raise clear errors."""
        from worldflux.models.tdmpc2.world_model import TDMPC2WorldModel

        config = TDMPC2Config(
            obs_shape=(39,),
            action_dim=6,
            latent_dim=64,
            hidden_dim=64,
        )
        model = TDMPC2WorldModel(config)

        # Inf observation
        inf_obs = torch.full((2, 39), float("inf"))
        try:
            state = model.encode(inf_obs)
            assert state is not None
        except (ValueError, RuntimeError):
            pass


class TestEmptyBatchHandling:
    """Test handling of empty or edge-case batch sizes."""

    def test_dreamer_batch_size_one(self):
        """DreamerV3 should handle batch_size=1."""
        from worldflux.models.dreamer.world_model import DreamerV3WorldModel

        config = DreamerV3Config(
            obs_shape=(3, 64, 64),
            action_dim=4,
            deter_dim=128,
            hidden_dim=64,
            cnn_depth=16,
        )
        model = DreamerV3WorldModel(config)

        obs = torch.randn(1, 3, 64, 64)
        state = model.encode(obs)
        assert state.tensors.get("deter") is not None
        assert state.tensors["deter"].shape[0] == 1

    def test_tdmpc2_batch_size_one(self):
        """TD-MPC2 should handle batch_size=1."""
        from worldflux.models.tdmpc2.world_model import TDMPC2WorldModel

        config = TDMPC2Config(
            obs_shape=(39,),
            action_dim=6,
            latent_dim=64,
            hidden_dim=64,
        )
        model = TDMPC2WorldModel(config)

        obs = torch.randn(1, 39)
        state = model.encode(obs)
        assert state.tensors.get("latent") is not None
        assert state.tensors["latent"].shape[0] == 1

    def test_sequence_length_one(self):
        """Models should handle sequence_length=1 in loss."""
        from worldflux.models.tdmpc2.world_model import TDMPC2WorldModel

        config = TDMPC2Config(
            obs_shape=(39,),
            action_dim=6,
            latent_dim=64,
            hidden_dim=64,
        )
        model = TDMPC2WorldModel(config)

        batch = Batch(
            obs=torch.randn(4, 1, 39),
            actions=torch.randn(4, 1, 6),
            rewards=torch.randn(4, 1),
        )

        # Should not crash, though losses may be zero
        loss_out = model.loss(batch)
        assert loss_out.loss is not None


class TestLongHorizonStability:
    """Test stability over long imagination horizons."""

    def test_dreamer_long_imagination(self):
        """DreamerV3 imagination should remain stable over long horizons."""
        from worldflux.models.dreamer.world_model import DreamerV3WorldModel

        config = DreamerV3Config(
            obs_shape=(3, 64, 64),
            action_dim=4,
            deter_dim=128,
            hidden_dim=64,
            cnn_depth=16,
        )
        model = DreamerV3WorldModel(config)

        obs = torch.randn(2, 3, 64, 64)
        initial_state = model.encode(obs)

        # Long horizon imagination
        horizon = 50
        actions = torch.randn(horizon, 2, 4)
        trajectory = model.rollout(initial_state, actions)

        # Check no NaN/Inf in trajectory
        assert not torch.isnan(trajectory.rewards).any(), "Rewards contain NaN"
        assert not torch.isinf(trajectory.rewards).any(), "Rewards contain Inf"

        for state in trajectory.states:
            for tensor in state.tensors.values():
                assert not torch.isnan(tensor).any()

    def test_tdmpc2_long_imagination(self):
        """TD-MPC2 imagination should remain stable over long horizons."""
        from worldflux.models.tdmpc2.world_model import TDMPC2WorldModel

        config = TDMPC2Config(
            obs_shape=(39,),
            action_dim=6,
            latent_dim=64,
            hidden_dim=64,
        )
        model = TDMPC2WorldModel(config)

        obs = torch.randn(2, 39)
        initial_state = model.encode(obs)

        # Long horizon imagination
        horizon = 50
        actions = torch.randn(horizon, 2, 6)
        trajectory = model.rollout(initial_state, actions)

        # Check no NaN/Inf
        assert not torch.isnan(trajectory.rewards).any()
        assert not torch.isinf(trajectory.rewards).any()


class TestConfigValidation:
    """Test configuration validation edge cases."""

    def test_tdmpc2_custom_gamma(self):
        """TD-MPC2 should accept custom gamma values."""
        config = TDMPC2Config(
            obs_shape=(39,),
            action_dim=6,
            gamma=0.95,
        )
        assert config.gamma == 0.95

    def test_tdmpc2_gamma_validation(self):
        """TD-MPC2 should reject invalid gamma values."""
        with pytest.raises(Exception):  # ConfigurationError
            TDMPC2Config(
                obs_shape=(39,),
                action_dim=6,
                gamma=0.0,  # Invalid: must be > 0
            )

        with pytest.raises(Exception):
            TDMPC2Config(
                obs_shape=(39,),
                action_dim=6,
                gamma=1.5,  # Invalid: must be <= 1
            )

    def test_dreamer_invalid_stoch_discrete(self):
        """DreamerV3 should reject non-positive stoch_discrete."""
        with pytest.raises(Exception):
            DreamerV3Config(
                obs_shape=(3, 64, 64),
                action_dim=4,
                stoch_discrete=0,
            )

    def test_tdmpc2_simnorm_divisibility(self):
        """TD-MPC2 should validate latent_dim divisibility by simnorm_dim."""
        with pytest.raises(Exception):
            TDMPC2Config(
                obs_shape=(39,),
                action_dim=6,
                latent_dim=100,  # Not divisible by 8
                simnorm_dim=8,
            )


class TestStateEdgeCases:
    """Test State edge cases."""

    def test_latent_state_none_components(self):
        """State should handle missing keys correctly."""
        state = State(tensors={"latent": torch.randn(4, 256)})
        assert state.batch_size == 4

    def test_latent_state_features_concatenation(self):
        """State supports multiple tensors."""
        deter = torch.randn(4, 512)
        stoch = torch.randn(4, 256)
        state = State(tensors={"deter": deter, "stoch": stoch})
        assert state.tensors["deter"].shape == (4, 512)
        assert state.tensors["stoch"].shape == (4, 256)
