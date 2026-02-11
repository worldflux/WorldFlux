"""Stress tests for numerical hardening guards in model dynamics."""

from __future__ import annotations

import torch

from worldflux.models.dreamer.rssm import RSSM
from worldflux.models.tdmpc2.dynamics import Dynamics


def _assert_finite_tensor(tensor: torch.Tensor) -> None:
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_rssm_stability_guards_handle_extreme_inputs() -> None:
    rssm = RSSM(
        embed_dim=4,
        action_dim=2,
        deter_dim=16,
        stoch_discrete=4,
        stoch_classes=4,
        hidden_dim=32,
    )
    state = rssm.initial_state(batch_size=3, device=torch.device("cpu"))

    action = torch.tensor(
        [[float("inf"), -float("inf")], [float("nan"), 1.0], [1e20, -1e20]],
        dtype=torch.float32,
    )
    obs_embed = torch.tensor(
        [
            [float("nan"), float("inf"), -float("inf"), 0.0],
            [1e20, -1e20, 0.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )

    prior_state = rssm.prior_step(state, action, deterministic=False)
    posterior_state = rssm.posterior_step(state, action, obs_embed)

    for tensor in prior_state.tensors.values():
        _assert_finite_tensor(tensor)
    for tensor in posterior_state.tensors.values():
        _assert_finite_tensor(tensor)


def test_tdmpc2_dynamics_guards_bound_extreme_residuals() -> None:
    dynamics = Dynamics(
        latent_dim=8,
        action_dim=2,
        hidden_dim=32,
        num_tasks=4,
        task_dim=3,
    )

    z = torch.tensor(
        [
            [float("nan"), float("inf"), -float("inf"), 0.0, 1e12, -1e12, 1.0, -1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ],
        dtype=torch.float32,
    )
    action = torch.tensor([[float("inf"), -float("inf")], [float("nan"), 1.0]])

    delta_default_task = dynamics(z, action, task_id=None)
    _assert_finite_tensor(delta_default_task)
    assert delta_default_task.shape == (2, 8)

    # Deliberately provide out-of-range task ids to validate clamped embedding path.
    task_id = torch.tensor([99, -5], dtype=torch.int64)
    delta_with_task = dynamics(z, action, task_id=task_id)
    _assert_finite_tensor(delta_with_task)
    assert delta_with_task.shape == (2, 8)

    max_norm = delta_with_task.norm(dim=-1).max().item()
    assert max_norm <= 10.0001
