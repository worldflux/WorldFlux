# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for training run manifest and checkpoint schema metadata."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from worldflux.core.batch import Batch
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput
from worldflux.core.state import State
from worldflux.training import Trainer, TrainingConfig


class _MiniModel(WorldModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def encode(self, obs, deterministic: bool = False) -> State:
        del deterministic
        if isinstance(obs, dict):
            obs = obs["obs"]
        return State(tensors={"latent": obs.float()})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        del action, deterministic
        return state

    def update(self, state: State, action: torch.Tensor, obs) -> State:
        del state, action
        return self.encode(obs)

    def decode(self, state: State):
        return None

    def loss(self, batch) -> LossOutput:
        obs = batch.obs
        if isinstance(obs, dict):
            obs = obs["obs"]
        pred = self.linear(obs.float())
        loss = pred.mean()
        return LossOutput(loss=loss, components={"mini": loss})


def _provider():
    class _Provider:
        def sample(self, batch_size: int, seq_len: int | None = None, device: str = "cpu"):
            del seq_len
            return Batch(obs=torch.randn(batch_size, 4, device=device))

    return _Provider()


def test_train_writes_run_manifest(tmp_path: Path) -> None:
    trainer = Trainer(
        _MiniModel(),
        TrainingConfig(
            total_steps=1,
            batch_size=2,
            sequence_length=1,
            output_dir=str(tmp_path),
            device="cpu",
            auto_quality_check=False,
        ),
        callbacks=[],
    )

    trainer.train(_provider(), num_steps=1)

    manifest_path = tmp_path / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "worldflux.training.run_manifest.v1"
    assert payload["backend"] == "native_torch"
    assert payload["checkpoint_schema_version"] == 1
    assert payload["global_step"] == 1


def test_save_checkpoint_includes_schema_version(tmp_path: Path) -> None:
    trainer = Trainer(
        _MiniModel(),
        TrainingConfig(
            total_steps=1,
            batch_size=2,
            sequence_length=1,
            output_dir=str(tmp_path),
            device="cpu",
            auto_quality_check=False,
        ),
        callbacks=[],
    )

    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_schema_version"] == 1
