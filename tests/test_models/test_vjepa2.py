"""Tests for V-JEPA2 world model."""

import pytest
import torch

from worldflux import AutoWorldModel
from worldflux.core.batch import Batch
from worldflux.core.config import VJEPA2Config
from worldflux.models.vjepa2 import VJEPA2WorldModel


def _config() -> VJEPA2Config:
    return VJEPA2Config.from_size(
        "ci",
        obs_shape=(4,),
        action_dim=1,
        encoder_dim=32,
        predictor_dim=32,
        projection_dim=16,
    )


def test_vjepa2_encode_decode():
    model = VJEPA2WorldModel(_config())
    obs = torch.randn(3, 4)
    state = model.encode(obs)
    output = model.decode(state)
    assert state.tensors["rep"].shape == (3, 32)
    assert output.preds["representation"].shape == (3, 16)


def test_vjepa2_loss_with_context_target_mask():
    model = VJEPA2WorldModel(_config())
    context = torch.randn(2, 4)
    target = torch.randn(2, 4)
    mask = torch.ones(2, 1)
    batch = Batch(obs=context, context=context, target=target, mask=mask)
    loss_out = model.loss(batch)
    assert "vjepa2" in loss_out.components
    assert torch.isfinite(loss_out.loss)


def test_vjepa2_loss_context_target_fallback():
    model = VJEPA2WorldModel(_config())
    obs = torch.randn(2, 4)
    batch = Batch(obs=obs)
    loss_out = model.loss(batch)
    assert "vjepa2" in loss_out.components


def test_vjepa2_loss_rejects_bad_mask_shape():
    model = VJEPA2WorldModel(_config())
    context = torch.randn(3, 4)
    target = torch.randn(3, 4)
    bad_mask = torch.ones(3, 2)
    batch = Batch(obs=context, context=context, target=target, mask=bad_mask)
    with pytest.raises(ValueError, match="mask shape"):
        model.loss(batch)


def test_vjepa2_io_contract():
    model = VJEPA2WorldModel(_config())
    contract = model.io_contract()
    assert contract.required_state_keys == ("rep",)
    assert "representation" in contract.prediction_spec.tensors


def test_vjepa2_set_encoder_trainable_toggles_requires_grad():
    model = VJEPA2WorldModel(_config())
    model.set_encoder_trainable(False)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    model.set_encoder_trainable(True)
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_vjepa2_auto_world_model_and_save_load(tmp_path):
    model = AutoWorldModel.from_pretrained("vjepa2:base", obs_shape=(4,), action_dim=1)
    save_path = str(tmp_path / "vjepa2_test")
    model.save_pretrained(save_path)

    loaded = AutoWorldModel.from_pretrained(save_path)
    assert isinstance(loaded, VJEPA2WorldModel)
    assert loaded.config.model_type == "vjepa2"
