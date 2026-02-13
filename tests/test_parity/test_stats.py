"""Tests for parity non-inferiority statistics."""

from __future__ import annotations

import pytest

from worldflux.parity.errors import ParitySampleSizeError
from worldflux.parity.stats import non_inferiority_test


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
