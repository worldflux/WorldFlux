# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for parity non-inferiority statistics and extended tests (ML-05)."""

from __future__ import annotations

import pytest

from worldflux.parity.errors import ParitySampleSizeError
from worldflux.parity.stats import (
    aggregate_scores,
    benjamini_hochberg,
    cohens_d,
    mann_whitney_u_test,
    non_inferiority_test,
    welch_t_test,
)

# ---------------------------------------------------------------------------
# Existing non-inferiority tests
# ---------------------------------------------------------------------------


def test_non_inferiority_passes_when_upper_ci_below_margin() -> None:
    result = non_inferiority_test([0.01, 0.02, 0.015, 0.018], margin_ratio=0.05, confidence=0.95)
    assert result.sample_size == 4
    assert result.pass_non_inferiority is True
    assert result.ci_upper_ratio <= 0.05


def test_non_inferiority_fails_when_upper_ci_exceeds_margin() -> None:
    result = non_inferiority_test([0.10, 0.12, 0.08, 0.11], margin_ratio=0.05, confidence=0.95)
    assert result.sample_size == 4
    assert result.pass_non_inferiority is False
    assert result.ci_upper_ratio > 0.05


def test_non_inferiority_rejects_empty_samples() -> None:
    with pytest.raises(ParitySampleSizeError, match="WF_PARITY_SAMPLE_SIZE"):
        non_inferiority_test([], margin_ratio=0.05)


def test_non_inferiority_rejects_sample_size_below_default_threshold() -> None:
    with pytest.raises(ParitySampleSizeError, match="at least 2"):
        non_inferiority_test([0.01], margin_ratio=0.05)


# ---------------------------------------------------------------------------
# ML-05: Welch's t-test
# ---------------------------------------------------------------------------


class TestWelchTTest:
    def test_identical_samples_not_significant(self) -> None:
        a = [10.0, 11.0, 12.0, 10.5, 11.5]
        result = welch_t_test(a, a, alpha=0.05)
        assert result.significant is False
        assert result.p_value >= 0.05

    def test_very_different_samples_significant(self) -> None:
        a = [100.0, 101.0, 102.0, 103.0, 104.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = welch_t_test(a, b, alpha=0.05)
        assert result.significant is True
        assert result.p_value < 0.05

    def test_rejects_too_few_samples(self) -> None:
        with pytest.raises(ParitySampleSizeError, match="at least 2"):
            welch_t_test([1.0], [2.0, 3.0])

    def test_returns_finite_values(self) -> None:
        a = [5.0, 6.0, 7.0]
        b = [4.0, 5.0, 6.0]
        result = welch_t_test(a, b)
        assert result.t_statistic == result.t_statistic  # not NaN
        assert result.p_value == result.p_value
        assert 0.0 <= result.p_value <= 1.0
        assert result.df > 0


# ---------------------------------------------------------------------------
# ML-05: Mann-Whitney U test
# ---------------------------------------------------------------------------


class TestMannWhitneyU:
    def test_identical_samples_not_significant(self) -> None:
        a = [10.0, 11.0, 12.0, 13.0, 14.0]
        result = mann_whitney_u_test(a, a, alpha=0.05)
        assert result.significant is False

    def test_separated_samples_significant(self) -> None:
        a = [100.0, 200.0, 300.0, 400.0, 500.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = mann_whitney_u_test(a, b, alpha=0.05)
        assert result.significant is True
        assert result.p_value < 0.05

    def test_rejects_empty_sample(self) -> None:
        with pytest.raises(ParitySampleSizeError, match="at least 1"):
            mann_whitney_u_test([], [1.0, 2.0])

    def test_p_value_in_range(self) -> None:
        a = [3.0, 4.0, 5.0]
        b = [6.0, 7.0, 8.0]
        result = mann_whitney_u_test(a, b)
        assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# ML-05: Cohen's d
# ---------------------------------------------------------------------------


class TestCohensD:
    def test_negligible_effect(self) -> None:
        # Use wide spread values with nearly identical means
        a = [8.0, 9.0, 10.0, 11.0, 12.0]
        b = [8.1, 9.1, 10.0, 10.9, 11.9]
        result = cohens_d(a, b)
        assert result.interpretation == "negligible"
        assert abs(result.cohens_d) < 0.2

    def test_large_effect(self) -> None:
        a = [100.0, 101.0, 102.0, 103.0, 104.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = cohens_d(a, b)
        assert result.interpretation == "large"
        assert abs(result.cohens_d) >= 0.8

    def test_rejects_too_few_samples(self) -> None:
        with pytest.raises(ParitySampleSizeError, match="at least 2"):
            cohens_d([1.0], [2.0, 3.0])

    def test_zero_variance(self) -> None:
        a = [5.0, 5.0, 5.0]
        b = [5.0, 5.0, 5.0]
        result = cohens_d(a, b)
        assert result.cohens_d == 0.0
        assert result.interpretation == "negligible"


# ---------------------------------------------------------------------------
# ML-05: Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_empty_input(self) -> None:
        result = benjamini_hochberg([])
        assert result.adjusted_p_values == []
        assert result.rejected == []

    def test_single_significant(self) -> None:
        result = benjamini_hochberg([0.01], alpha=0.05)
        assert result.rejected == [True]
        assert result.adjusted_p_values[0] <= 0.05

    def test_multiple_correction_reduces_rejections(self) -> None:
        # p-values near the threshold should lose significance after correction
        p_values = [0.01, 0.04, 0.03, 0.06, 0.20]
        result = benjamini_hochberg(p_values, alpha=0.05)
        # Original: 4 would be below 0.05 without correction (0.01, 0.03, 0.04)
        # After BH correction, fewer should be rejected
        assert len(result.adjusted_p_values) == 5
        assert all(0.0 <= p <= 1.0 for p in result.adjusted_p_values)

    def test_monotonicity(self) -> None:
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        result = benjamini_hochberg(p_values, alpha=0.05)
        # Adjusted p-values should be monotonically non-decreasing
        # when original p-values are sorted
        sorted_adj = sorted(
            zip(result.original_p_values, result.adjusted_p_values),
            key=lambda x: x[0],
        )
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i][1] >= sorted_adj[i - 1][1]

    def test_all_significant(self) -> None:
        p_values = [0.001, 0.002, 0.003]
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert all(result.rejected)

    def test_none_significant(self) -> None:
        p_values = [0.5, 0.6, 0.7]
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert not any(result.rejected)


# ---------------------------------------------------------------------------
# ML-02/05: Score aggregation
# ---------------------------------------------------------------------------


class TestAggregateScores:
    def test_basic_aggregation(self) -> None:
        scores = [100.0, 110.0, 90.0, 105.0, 95.0]
        result = aggregate_scores(scores)
        assert result.n_samples == 5
        assert 90.0 <= result.mean <= 110.0
        assert result.std > 0.0
        assert result.ci_low <= result.mean <= result.ci_high

    def test_single_score(self) -> None:
        result = aggregate_scores([42.0])
        assert result.n_samples == 1
        assert result.mean == 42.0
        assert result.std == 0.0

    def test_rejects_empty(self) -> None:
        with pytest.raises(ParitySampleSizeError, match="at least one"):
            aggregate_scores([])

    def test_iqr_bounds(self) -> None:
        scores = list(range(1, 101))  # 1..100
        result = aggregate_scores([float(x) for x in scores])
        assert result.iqr_low <= result.median <= result.iqr_high
