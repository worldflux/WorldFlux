"""Tests for TD-MPC2 component-level match verification."""

from __future__ import annotations

import pytest
import torch

from worldflux.parity.component_match import (
    match_backward,
    match_forward,
)


class TestTDMPC2ComponentMatch:
    @pytest.fixture
    def dynamics_mlp(self):
        return torch.nn.Sequential(
            torch.nn.Linear(20, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 12),
        )

    @pytest.fixture
    def encoder_mlp(self):
        return torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 12),
        )

    @pytest.fixture
    def reward_mlp(self):
        return torch.nn.Sequential(
            torch.nn.Linear(18, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 1),
        )

    @pytest.fixture
    def policy_mlp(self):
        return torch.nn.Sequential(
            torch.nn.Linear(12, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 6),
            torch.nn.Tanh(),
        )

    def test_dynamics_self_match(self, dynamics_mlp) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 20)
        result = match_forward(dynamics_mlp, dynamics_mlp, [x], component="dynamics")
        assert result.shape_match is True
        assert result.max_abs_diff == 0.0

    def test_encoder_self_match(self, encoder_mlp) -> None:
        torch.manual_seed(43)
        x = torch.randn(2, 10)
        result = match_forward(encoder_mlp, encoder_mlp, [x], component="encoder")
        assert result.shape_match is True
        assert result.max_abs_diff == 0.0

    def test_reward_self_match(self, reward_mlp) -> None:
        torch.manual_seed(44)
        x = torch.randn(2, 18)
        result = match_forward(reward_mlp, reward_mlp, [x], component="reward")
        assert result.shape_match is True
        assert result.max_abs_diff == 0.0

    def test_policy_self_match(self, policy_mlp) -> None:
        torch.manual_seed(45)
        x = torch.randn(2, 12)
        result = match_forward(policy_mlp, policy_mlp, [x], component="policy")
        assert result.shape_match is True
        assert result.max_abs_diff == 0.0

    def test_dynamics_backward_match(self, dynamics_mlp) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 20)
        result = match_backward(dynamics_mlp, dynamics_mlp, [x], component="dynamics_bwd")
        assert result.rtol_pass is True

    def test_shared_weights_dynamics(self) -> None:
        torch.manual_seed(10)
        mlp_a = torch.nn.Sequential(
            torch.nn.Linear(20, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 12),
        )
        mlp_b = torch.nn.Sequential(
            torch.nn.Linear(20, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 12),
        )
        mlp_b.load_state_dict(mlp_a.state_dict())
        x = torch.randn(3, 20)
        result = match_forward(mlp_a, mlp_b, [x], component="dyn_shared")
        assert result.max_abs_diff == 0.0
        assert result.rtol_pass is True

    def test_q_ensemble_self_match(self) -> None:
        torch.manual_seed(50)
        q_net = torch.nn.Sequential(
            torch.nn.Linear(18, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 16),
            torch.nn.LayerNorm(16),
            torch.nn.Mish(),
            torch.nn.Linear(16, 1),
        )
        x = torch.randn(2, 18)
        result = match_forward(q_net, q_net, [x], component="q_net")
        assert result.max_abs_diff == 0.0
