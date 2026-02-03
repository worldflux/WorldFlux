"""Tests for LatentState."""

import pytest
import torch

from worldflux.core.exceptions import StateError
from worldflux.core.state import LatentState


class TestLatentState:
    """LatentState unit tests."""

    def test_deterministic_only(self):
        """TD-MPC2 style: deterministic only."""
        state = LatentState(deterministic=torch.randn(4, 512))
        assert state.batch_size == 4
        assert state.features.shape == (4, 512)
        assert state.stochastic is None

    def test_both_components(self):
        """DreamerV3 style: deterministic + stochastic."""
        state = LatentState(
            deterministic=torch.randn(4, 4096),
            stochastic=torch.randn(4, 32, 32),
        )
        assert state.features.shape == (4, 4096 + 32 * 32)

    def test_stochastic_2d(self):
        """Stochastic with 2D shape."""
        state = LatentState(
            deterministic=torch.randn(4, 256),
            stochastic=torch.randn(4, 128),
        )
        assert state.features.shape == (4, 256 + 128)

    def test_to_device(self):
        """Device transfer."""
        state = LatentState(
            deterministic=torch.randn(4, 512),
            stochastic=torch.randn(4, 32, 32),
        )
        if torch.cuda.is_available():
            state_cuda = state.to(torch.device("cuda"))
            assert state_cuda.deterministic.device.type == "cuda"
            assert state_cuda.stochastic.device.type == "cuda"

    def test_detach(self):
        """Detach from computation graph."""
        x = torch.randn(4, 512, requires_grad=True)
        state = LatentState(deterministic=x)
        detached = state.detach()
        assert not detached.deterministic.requires_grad

    def test_clone(self):
        """Deep copy."""
        state = LatentState(
            deterministic=torch.randn(4, 512),
            stochastic=torch.randn(4, 32, 32),
        )
        cloned = state.clone()

        # Modify original
        state.deterministic.fill_(0)

        # Clone should be unaffected
        assert not torch.allclose(cloned.deterministic, state.deterministic)

    def test_no_components_raises(self):
        """Empty state raises error during initialization."""
        with pytest.raises(StateError):
            LatentState()

    def test_device_property(self):
        """Device property."""
        state = LatentState(deterministic=torch.randn(4, 512))
        assert state.device == torch.device("cpu")
