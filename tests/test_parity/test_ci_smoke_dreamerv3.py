"""CI smoke tests for DreamerV3 training loop validation."""

from __future__ import annotations

import math

import pytest
import torch

from worldflux import DreamerV3Config
from worldflux.models.dreamer import DreamerV3WorldModel
from worldflux.parity.ci_smoke import run_smoke_test


@pytest.fixture
def small_model():
    """Small DreamerV3 model for smoke testing."""
    config = DreamerV3Config(
        deter_dim=128,
        stoch_discrete=4,
        stoch_classes=4,
        hidden_dim=64,
        cnn_depth=8,
    )
    return DreamerV3WorldModel(config)


class TestDreamerV3SmokeLossRange:
    """Verify final loss is within acceptable range after training."""

    def test_loss_in_range(self, small_model):
        result = run_smoke_test(small_model, family="dreamerv3", steps=10, seed=42)
        assert not math.isnan(result.final_loss), "final loss is NaN"
        assert not math.isinf(result.final_loss), "final loss is Inf"
        assert result.final_loss > 0.0, "final loss should be positive"

    def test_component_losses_present(self, small_model):
        result = run_smoke_test(small_model, family="dreamerv3", steps=10, seed=42)
        for key in ("kl", "reconstruction", "reward", "continue"):
            assert key in result.component_losses, f"missing component: {key}"
            val = result.component_losses[key]
            assert not math.isnan(val), f"component {key} is NaN"

    def test_component_losses_in_range(self, small_model):
        result = run_smoke_test(small_model, family="dreamerv3", steps=10, seed=42)
        for key, val in result.component_losses.items():
            assert val >= 0.0, f"component {key}={val} is negative"


class TestDreamerV3SmokeNaN:
    """Verify no NaN values during training."""

    def test_no_nan_loss(self, small_model):
        result = run_smoke_test(small_model, family="dreamerv3", steps=10, seed=42)
        assert not math.isnan(result.final_loss)

    def test_no_nan_gradients(self, small_model):
        result = run_smoke_test(small_model, family="dreamerv3", steps=10, seed=42)
        for i, g in enumerate(result.gradient_norms):
            assert not math.isnan(g), f"gradient norm at step {i} is NaN"

    def test_no_nan_parameters(self, small_model):
        """After training, parameters should contain no NaN."""
        run_smoke_test(small_model, family="dreamerv3", steps=10, seed=42)
        for name, p in small_model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN in parameter {name}"


class TestDreamerV3SmokeGradientFlow:
    """Verify gradients flow through the model."""

    def test_nonzero_gradients(self, small_model):
        result = run_smoke_test(small_model, family="dreamerv3", steps=10, seed=42)
        assert any(g > 0.0 for g in result.gradient_norms), "all gradient norms are zero"

    def test_gradient_norm_bounded(self, small_model):
        result = run_smoke_test(small_model, family="dreamerv3", steps=10, seed=42)
        for i, g in enumerate(result.gradient_norms):
            assert g < 1e6, f"gradient norm at step {i} is {g:.1e}, exceeds 1e6"


class TestDreamerV3SmokeParameterUpdate:
    """Verify parameters are updated during training."""

    def test_param_delta_nonzero(self, small_model):
        result = run_smoke_test(small_model, family="dreamerv3", steps=10, seed=42)
        assert result.param_delta_norm > 0.0, "parameters did not change after training"

    def test_full_smoke_passes(self, small_model):
        """Full smoke checkpoint validation."""
        result = run_smoke_test(small_model, family="dreamerv3", steps=10, seed=42)
        assert result.passed, f"smoke test failed: {result.violations}"
