"""Statistical utilities for non-inferiority parity checks."""

from __future__ import annotations

import numpy as np

from .errors import ParityError, ParitySampleSizeError
from .types import NonInferiorityResult


def _round_ratio(value: float) -> float:
    return float(f"{value:.10f}")


def non_inferiority_test(
    drop_ratios: list[float],
    *,
    margin_ratio: float = 0.05,
    confidence: float = 0.95,
    bootstrap_samples: int = 4000,
    seed: int = 0,
    min_samples: int = 2,
) -> NonInferiorityResult:
    """
    Evaluate one-sided non-inferiority over mean relative drop ratios.

    ``drop_ratio`` is defined as positive when WorldFlux underperforms upstream
    (e.g., ``(upstream - worldflux) / abs(upstream)`` for higher-is-better metrics).
    Non-inferiority passes when the one-sided upper confidence bound is below margin.
    """
    if not 0.0 < confidence < 1.0:
        raise ParityError(f"confidence must be in (0, 1), got {confidence!r}")
    if margin_ratio < 0:
        raise ParityError(f"margin_ratio must be non-negative, got {margin_ratio!r}")
    if min_samples < 1:
        raise ParityError(f"min_samples must be >= 1, got {min_samples!r}")

    arr = np.asarray(drop_ratios, dtype=np.float64)
    if arr.size == 0:
        raise ParitySampleSizeError("non_inferiority_test requires at least one paired sample")
    if arr.size < min_samples:
        raise ParitySampleSizeError(
            f"non_inferiority_test requires at least {min_samples} paired samples, got {arr.size}"
        )

    mean_drop = float(arr.mean())
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(bootstrap_samples, dtype=np.float64)
    for idx in range(bootstrap_samples):
        sampled = rng.choice(arr, size=arr.size, replace=True)
        sampled_means[idx] = sampled.mean()

    alpha = 1.0 - confidence
    ci_low = float(np.quantile(sampled_means, alpha))
    ci_high = float(np.quantile(sampled_means, confidence))

    passed = bool(ci_high <= margin_ratio)
    verdict_reason = (
        f"PASS: one-sided upper CI {_round_ratio(ci_high):.6f} <= margin {_round_ratio(margin_ratio):.6f}"
        if passed
        else f"FAIL: one-sided upper CI {_round_ratio(ci_high):.6f} > margin {_round_ratio(margin_ratio):.6f}"
    )

    return NonInferiorityResult(
        sample_size=int(arr.size),
        mean_drop_ratio=_round_ratio(mean_drop),
        ci_lower_ratio=_round_ratio(ci_low),
        ci_upper_ratio=_round_ratio(ci_high),
        confidence=_round_ratio(confidence),
        margin_ratio=_round_ratio(margin_ratio),
        pass_non_inferiority=passed,
        verdict_reason=verdict_reason,
    )
