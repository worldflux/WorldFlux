# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Phase 1 correctness regressions for TD-MPC2."""

from __future__ import annotations

import torch

from worldflux import Batch, TDMPC2Config
from worldflux.models.tdmpc2.world_model import TDMPC2WorldModel
from worldflux.training import Trainer, TrainingConfig


def _tiny_model() -> TDMPC2WorldModel:
    return TDMPC2WorldModel(
        TDMPC2Config(
            obs_shape=(39,),
            action_dim=6,
            latent_dim=32,
            hidden_dim=32,
            num_q_networks=2,
        )
    )


def test_tdmpc2_td_target_ignores_bootstrap_when_transition_is_terminal(
    monkeypatch,
) -> None:
    model = _tiny_model()
    batch = Batch(
        obs=torch.randn(1, 2, 39),
        actions=torch.randn(1, 2, 6),
        rewards=torch.tensor([[0.0, 0.5]], dtype=torch.float32),
        terminations=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    )

    monkeypatch.setattr(
        model,
        "predict_q",
        lambda state, action: torch.zeros(model.config.num_q_networks, action.shape[0]),
    )
    monkeypatch.setattr(
        model._policy,
        "forward",
        lambda z: torch.zeros(z.shape[0], model.config.action_dim, device=z.device),
    )
    monkeypatch.setattr(
        model._target_q_ensemble,
        "forward",
        lambda z, action: torch.ones(
            model.config.num_q_networks,
            z.shape[0],
            1,
            device=z.device,
        ),
    )

    loss_out = model.loss(batch)

    assert torch.isclose(loss_out.components["td"], torch.tensor(0.25), atol=1e-6)


def test_tdmpc2_loss_does_not_mutate_target_q_parameters() -> None:
    model = _tiny_model()
    batch = Batch(
        obs=torch.randn(2, 4, 39),
        actions=torch.randn(2, 4, 6),
        rewards=torch.randn(2, 4),
    )

    for param in model._q_ensemble.parameters():
        param.data.add_(0.5)

    before = [param.detach().clone() for param in model._target_q_ensemble.parameters()]
    model.loss(batch)
    after = [param.detach().clone() for param in model._target_q_ensemble.parameters()]

    assert all(torch.allclose(old, new) for old, new in zip(before, after))


def test_tdmpc2_train_step_updates_target_q_after_optimizer_step(tmp_path) -> None:
    model = _tiny_model()
    trainer = Trainer(
        model,
        TrainingConfig(
            total_steps=1,
            batch_size=2,
            sequence_length=4,
            output_dir=str(tmp_path),
            device="cpu",
        ),
        callbacks=[],
    )

    class _Provider:
        def sample(self, batch_size: int, seq_len: int | None = None, device: str = "cpu"):
            assert seq_len is not None
            return Batch(
                obs=torch.randn(batch_size, seq_len, 39, device=device),
                actions=torch.randn(batch_size, seq_len, 6, device=device),
                rewards=torch.randn(batch_size, seq_len, device=device),
                terminations=torch.zeros(batch_size, seq_len, device=device),
            )

    for param in model._q_ensemble.parameters():
        param.data.add_(0.5)

    before = [param.detach().clone() for param in model._target_q_ensemble.parameters()]
    trainer._train_step(_Provider())
    after = [param.detach().clone() for param in model._target_q_ensemble.parameters()]

    assert any(not torch.allclose(old, new) for old, new in zip(before, after))
