"""Tests for DreamerV3 component-level match verification."""

from __future__ import annotations

import pytest
import torch

from worldflux.parity.component_match import (
    ComponentMatchReport,
    MatchResult,
    match_backward,
    match_forward,
)


class TestMatchForward:
    def test_identical_functions_pass(self) -> None:
        linear = torch.nn.Linear(8, 4)
        x = torch.randn(2, 8)
        result = match_forward(linear, linear, [x], component="test")
        assert result.shape_match is True
        assert result.max_abs_diff == 0.0
        assert result.rtol_pass is True
        assert result.atol_pass is True

    def test_shape_mismatch_detected(self) -> None:
        linear_a = torch.nn.Linear(8, 4)
        linear_b = torch.nn.Linear(8, 6)
        x = torch.randn(2, 8)
        result = match_forward(linear_a, linear_b, [x], component="mismatch")
        assert result.shape_match is False
        assert result.rtol_pass is False

    def test_different_weights_may_fail_atol(self) -> None:
        torch.manual_seed(0)
        linear_a = torch.nn.Linear(8, 4)
        torch.manual_seed(1)
        linear_b = torch.nn.Linear(8, 4)
        x = torch.randn(2, 8)
        result = match_forward(linear_a, linear_b, [x], component="diff", atol=1e-10)
        assert result.max_abs_diff > 0


class TestMatchBackward:
    def test_identical_functions_pass(self) -> None:
        linear = torch.nn.Linear(8, 4)
        x = torch.randn(2, 8)
        result = match_backward(linear, linear, [x], component="bwd_test")
        assert result.shape_match is True
        assert result.rtol_pass is True


class TestDreamerV3ComponentMatch:
    @pytest.fixture
    def rssm_prior(self):
        return torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.LayerNorm(16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 8),
        )

    @pytest.fixture
    def rssm_posterior(self):
        return torch.nn.Sequential(
            torch.nn.Linear(40, 16),
            torch.nn.LayerNorm(16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 8),
        )

    @pytest.fixture
    def gru_cell(self):
        return torch.nn.GRUCell(12, 32)

    def test_prior_net_self_match(self, rssm_prior) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 32)
        result = match_forward(rssm_prior, rssm_prior, [x], component="prior")
        assert result.shape_match is True
        assert result.max_abs_diff == 0.0

    def test_posterior_net_self_match(self, rssm_posterior) -> None:
        torch.manual_seed(43)
        x = torch.randn(2, 40)
        result = match_forward(rssm_posterior, rssm_posterior, [x], component="posterior")
        assert result.shape_match is True
        assert result.max_abs_diff == 0.0

    def test_gru_self_match(self, gru_cell) -> None:
        torch.manual_seed(44)
        x = torch.randn(2, 12)
        h = torch.randn(2, 32)
        result = match_forward(gru_cell, gru_cell, [x, h], component="gru")
        assert result.shape_match is True
        assert result.max_abs_diff == 0.0

    def test_prior_net_backward_match(self, rssm_prior) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 32)
        result = match_backward(rssm_prior, rssm_prior, [x], component="prior_bwd")
        assert result.rtol_pass is True

    def test_shared_weights_produce_identical_output(self) -> None:
        torch.manual_seed(10)
        net_a = torch.nn.Sequential(
            torch.nn.Linear(16, 8),
            torch.nn.LayerNorm(8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 4),
        )
        net_b = torch.nn.Sequential(
            torch.nn.Linear(16, 8),
            torch.nn.LayerNorm(8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 4),
        )
        net_b.load_state_dict(net_a.state_dict())
        x = torch.randn(3, 16)
        result = match_forward(net_a, net_b, [x], component="shared_weights")
        assert result.max_abs_diff == 0.0
        assert result.rtol_pass is True
        assert result.atol_pass is True

    def test_component_match_report_pass(self) -> None:
        results = (
            MatchResult("a", 0.0, 0.0, True, True, True),
            MatchResult("b", 0.0, 0.0, True, True, True),
        )
        report = ComponentMatchReport(family="dreamerv3", results=results)
        assert report.all_pass is True

    def test_component_match_report_fail(self) -> None:
        results = (
            MatchResult("a", 0.0, 0.0, True, True, True),
            MatchResult("b", 1.0, 0.5, False, False, True),
        )
        report = ComponentMatchReport(family="dreamerv3", results=results)
        assert report.all_pass is False
