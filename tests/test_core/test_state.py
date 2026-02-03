"""Tests for State."""

import pytest
import torch

from worldflux.core.state import State


class TestState:
    """State unit tests."""

    def test_basic_state(self):
        state = State(tensors={"latent": torch.randn(4, 512)})
        assert state.batch_size == 4

    def test_to_device(self):
        state = State(tensors={"latent": torch.randn(4, 512)})
        if torch.cuda.is_available():
            state_cuda = state.to(torch.device("cuda"))
            assert state_cuda.tensors["latent"].device.type == "cuda"

    def test_detach(self):
        x = torch.randn(4, 512, requires_grad=True)
        state = State(tensors={"latent": x})
        detached = state.detach()
        assert not detached.tensors["latent"].requires_grad

    def test_clone(self):
        state = State(tensors={"latent": torch.randn(4, 512)})
        cloned = state.clone()

        state.tensors["latent"].fill_(0)
        assert not torch.allclose(cloned.tensors["latent"], state.tensors["latent"])

    def test_device_property(self):
        state = State(tensors={"latent": torch.randn(4, 512)})
        assert state.device == torch.device("cpu")

    def test_empty_state_batch_size_raises(self):
        state = State()
        with pytest.raises(ValueError):
            _ = state.batch_size
