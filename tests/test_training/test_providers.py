"""Tests for provider protocol behavior and strict layout enforcement."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from worldflux.core.batch import Batch
from worldflux.core.exceptions import TrainingError
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput
from worldflux.core.state import State
from worldflux.training import TokenSequenceProvider, Trainer, TrainingConfig


class _MiniModel(WorldModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def encode(self, obs, deterministic: bool = False) -> State:
        if isinstance(obs, dict):
            obs = obs["obs"]
        return State(tensors={"latent": obs.float()})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        return state

    def update(self, state: State, action: torch.Tensor, obs) -> State:
        return self.encode(obs)

    def decode(self, state: State):
        return None

    def loss(self, batch) -> LossOutput:
        obs = batch.obs
        if isinstance(obs, dict):
            obs = obs["obs"]
        # Use first feature channel to keep this test lightweight.
        if obs.dim() > 2:
            obs = obs[..., 0]
        pred = self.linear(obs.float())
        loss = pred.mean()
        return LossOutput(loss=loss, components={"mini": loss})


def test_token_sequence_provider_emits_strict_layout_batch():
    tokens = np.random.randint(0, 32, size=(4, 16), dtype=np.int64)
    provider = TokenSequenceProvider(tokens=tokens)
    batch = provider.sample(batch_size=2, seq_len=8, device="cpu")
    assert batch.strict_layout is True
    assert batch.layouts["obs"] == "BT"
    assert batch.layouts["target"] == "BT"


def test_trainer_rejects_provider_with_unknown_layout_field():
    class BadLayoutProvider:
        def batch_layout(self) -> dict[str, str]:
            return {"imaginary": "BT"}

        def sample(self, batch_size: int, seq_len: int | None = None, device="cpu") -> Batch:
            seq = 4 if seq_len is None else seq_len
            return Batch(
                obs=torch.randn(batch_size, seq, 4, device=device),
                layouts={"imaginary": "BT"},
                strict_layout=True,
            )

    model = _MiniModel()
    trainer = Trainer(
        model,
        TrainingConfig(total_steps=1, batch_size=2, sequence_length=4, device="cpu"),
        callbacks=[],
    )
    with pytest.raises(TrainingError, match="Unknown layout field"):
        trainer._next_batch(BadLayoutProvider())


def test_trainer_accepts_batch_provider_v2_request_shape():
    class RequestProvider:
        def sample(self, request) -> Batch:
            seq = 4 if request.seq_len is None else request.seq_len
            lengths = torch.full(
                (request.batch_size,), seq, device=request.device, dtype=torch.int64
            )
            return Batch(
                obs=torch.randn(request.batch_size, seq, 4, device=request.device),
                lengths={"obs": lengths},
                layouts={"obs": "BT..."},
                strict_layout=True,
            )

    model = _MiniModel()
    trainer = Trainer(
        model,
        TrainingConfig(total_steps=1, batch_size=2, sequence_length=4, device="cpu"),
        callbacks=[],
    )
    batch = trainer._next_batch(RequestProvider())
    assert isinstance(batch, Batch)
    assert batch.obs is not None
