"""Tests for token world model."""

import torch

from worldflux.core.batch import Batch
from worldflux.core.config import TokenWorldModelConfig
from worldflux.models.token import TokenWorldModel


def test_token_encode_decode():
    config = TokenWorldModelConfig(
        obs_shape=(8,),
        action_dim=1,
        vocab_size=32,
        token_dim=16,
        num_layers=1,
        num_heads=1,
    )
    model = TokenWorldModel(config)
    tokens = torch.randint(0, config.vocab_size, (2, 4))
    state = model.encode(tokens)
    output = model.decode(state)
    assert output.preds["logits"].shape[:2] == (2, 4)


def test_token_loss():
    config = TokenWorldModelConfig(
        obs_shape=(8,),
        action_dim=1,
        vocab_size=32,
        token_dim=16,
        num_layers=1,
        num_heads=1,
    )
    model = TokenWorldModel(config)
    tokens = torch.randint(0, config.vocab_size, (2, 4))
    batch = Batch(obs=tokens, target=tokens)
    loss_out = model.loss(batch)
    assert "token_ce" in loss_out.components
    assert torch.isfinite(loss_out.loss)


def test_token_io_contract():
    config = TokenWorldModelConfig(
        obs_shape=(8,),
        action_dim=1,
        vocab_size=32,
        token_dim=16,
        num_layers=1,
        num_heads=1,
    )
    model = TokenWorldModel(config)
    contract = model.io_contract()
    assert contract.required_state_keys == ("tokens",)
    assert contract.sequence_layout.axes_by_field["obs"] == "BT"
