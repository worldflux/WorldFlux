"""Tests for JEPA base world model."""

import pytest
import torch

from worldflux import AutoWorldModel
from worldflux.core.batch import Batch
from worldflux.core.config import JEPABaseConfig
from worldflux.core.exceptions import ConfigurationError
from worldflux.core.spec import Capability
from worldflux.models.jepa import JEPABaseWorldModel


class TestJEPABaseConfig:
    """JEPA config validation tests."""

    def test_invalid_dims_raise(self):
        with pytest.raises(ConfigurationError):
            JEPABaseConfig(encoder_dim=0)
        with pytest.raises(ConfigurationError):
            JEPABaseConfig(predictor_dim=-1)
        with pytest.raises(ConfigurationError):
            JEPABaseConfig(projection_dim=0)
        with pytest.raises(ConfigurationError):
            JEPABaseConfig(num_layers=0)
        with pytest.raises(ConfigurationError):
            JEPABaseConfig(num_heads=0)


class TestJEPABaseWorldModel:
    """JEPA model tests."""

    @pytest.fixture
    def model(self):
        config = JEPABaseConfig(obs_shape=(4,), action_dim=2, encoder_dim=32, predictor_dim=32)
        return JEPABaseWorldModel(config)

    def test_encode_tensor(self, model):
        obs = torch.randn(3, 4)
        state = model.encode(obs)
        assert "rep" in state.tensors
        assert state.tensors["rep"].shape[0] == 3

    def test_encode_dict(self, model):
        obs = torch.randn(2, 4)
        state = model.encode({"obs": obs})
        assert "rep" in state.tensors

    def test_decode(self, model):
        obs = torch.randn(2, 4)
        state = model.encode(obs)
        output = model.decode(state)
        assert "representation" in output.preds

    def test_loss_with_context_target(self, model):
        context = torch.randn(3, 4)
        target = torch.randn(3, 4)
        mask = torch.ones(3, 1)
        batch = Batch(obs=context, context=context, target=target, mask=mask)
        loss_out = model.loss(batch)
        assert "jepa" in loss_out.components
        assert torch.isfinite(loss_out.loss)

    def test_capability_flag(self, model):
        assert Capability.REPRESENTATION in model.capabilities

    def test_auto_world_model(self):
        model = AutoWorldModel.from_pretrained("jepa:base")
        assert isinstance(model, JEPABaseWorldModel)
