# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Reference-fidelity tests for TD-MPC2 training surfaces."""

from __future__ import annotations

import torch

from worldflux import Batch, TDMPC2Config
from worldflux.models.tdmpc2.world_model import TDMPC2WorldModel


def test_tdmpc2_proof_profile_exposes_alignment_metadata() -> None:
    config = TDMPC2Config.from_size("proof_5m", obs_shape=(39,), action_dim=6)

    assert config.training_tier == "proof"
    assert config.parity_profile == "proof_5m"


def test_tdmpc2_reference_loss_contains_policy_component() -> None:
    model = TDMPC2WorldModel(
        TDMPC2Config(
            obs_shape=(39,),
            action_dim=6,
            latent_dim=64,
            hidden_dim=64,
            num_q_networks=2,
        )
    )
    batch = Batch(
        obs=torch.randn(2, 6, 39),
        actions=torch.randn(2, 6, 6),
        rewards=torch.randn(2, 6),
    )

    loss_out = model.loss(batch)
    assert "policy" in loss_out.components


def test_tdmpc2_soft_update_moves_target_q_parameters() -> None:
    model = TDMPC2WorldModel(
        TDMPC2Config(
            obs_shape=(39,),
            action_dim=6,
            latent_dim=32,
            hidden_dim=32,
            num_q_networks=2,
        )
    )

    for param in model._q_ensemble.parameters():
        param.data.add_(0.5)

    before = [param.detach().clone() for param in model._target_q_ensemble.parameters()]
    model._soft_update_target_q()
    after = [param.detach().clone() for param in model._target_q_ensemble.parameters()]

    assert any(not torch.allclose(old, new) for old, new in zip(before, after))
