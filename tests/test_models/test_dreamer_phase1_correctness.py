# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Phase 1 correctness regressions for DreamerV3."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from worldflux import Batch, DreamerV3Config
from worldflux.core.state import State
from worldflux.models.dreamer.world_model import DreamerV3WorldModel


def test_dreamer_cnn_reconstruction_does_not_symlog_image_targets(monkeypatch) -> None:
    model = DreamerV3WorldModel(
        DreamerV3Config(
            obs_shape=(3, 64, 64),
            action_dim=6,
            encoder_type="cnn",
            decoder_type="cnn",
            hidden_dim=64,
            deter_dim=64,
            stoch_discrete=4,
            stoch_classes=4,
            use_symlog=True,
        )
    )
    batch = Batch(
        obs=torch.ones(2, 1, 3, 64, 64),
        actions=torch.zeros(2, 1, 6),
        rewards=torch.zeros(2, 1),
        terminations=torch.zeros(2, 1),
    )

    state = State(
        tensors={
            "deter": torch.zeros(2, model.config.deter_dim),
            "stoch": torch.zeros(2, model.config.stoch_discrete, model.config.stoch_classes),
        }
    )

    monkeypatch.setattr(model, "initial_state", lambda batch_size, device: state)
    monkeypatch.setattr(model, "update", lambda prev, action, obs: state)
    monkeypatch.setattr(
        model,
        "decode",
        lambda current_state: SimpleNamespace(
            preds={
                "obs": torch.zeros(2, 3, 64, 64),
                "continue": torch.zeros(2, 1),
            }
        ),
    )
    monkeypatch.setattr(
        "worldflux.models.dreamer.world_model.symlog",
        lambda tensor: tensor + 100.0,
    )

    loss_out = model.loss(batch)

    assert torch.isclose(loss_out.components["reconstruction"], torch.tensor(1.0))


def test_dreamer_mlp_reconstruction_still_symlogs_vector_targets(monkeypatch) -> None:
    model = DreamerV3WorldModel(
        DreamerV3Config(
            obs_shape=(8,),
            action_dim=2,
            encoder_type="mlp",
            decoder_type="mlp",
            hidden_dim=32,
            deter_dim=32,
            stoch_discrete=4,
            stoch_classes=4,
            use_symlog=True,
        )
    )
    batch = Batch(
        obs=torch.ones(2, 1, 8),
        actions=torch.zeros(2, 1, 2),
        rewards=torch.zeros(2, 1),
        terminations=torch.zeros(2, 1),
    )

    state = State(
        tensors={
            "deter": torch.zeros(2, model.config.deter_dim),
            "stoch": torch.zeros(2, model.config.stoch_discrete, model.config.stoch_classes),
        }
    )

    monkeypatch.setattr(model, "initial_state", lambda batch_size, device: state)
    monkeypatch.setattr(model, "update", lambda prev, action, obs: state)
    monkeypatch.setattr(
        model,
        "decode",
        lambda current_state: SimpleNamespace(
            preds={
                "obs": torch.zeros(2, 8),
                "continue": torch.zeros(2, 1),
            }
        ),
    )
    monkeypatch.setattr(
        "worldflux.models.dreamer.world_model.symlog",
        lambda tensor: tensor + 100.0,
    )

    loss_out = model.loss(batch)

    assert torch.isclose(loss_out.components["reconstruction"], torch.tensor(10201.0))
