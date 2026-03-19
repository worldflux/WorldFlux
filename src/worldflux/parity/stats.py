"""Statistical utilities for non-inferiority parity checks.

Includes bootstrap-based non-inferiority testing, classical parametric and
non-parametric tests (Welch's t-test, Mann-Whitney U), effect size measures
(Cohen's d), and multiple-testing correction (Benjamini-Hochberg FDR).

When ``scipy`` is available the implementations delegate to
``scipy.stats``; otherwise a pure-Python fallback is used so that the
module works without optional dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import ParityError, ParitySampleSizeError
from .types import NonInferiorityResult

# ---------------------------------------------------------------------------
# Result types for extended statistical tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WelchTTestResult:
    """Result of Welch's two-sample t-test."""

    t_statistic: float
    p_value: float
    df: float
    significant: bool
    alpha: float


@dataclass(frozen=True)
class MannWhitneyResult:
    """Result of Mann-Whitney U test."""

    u_statistic: float
    p_value: float
    significant: bool
    alpha: float


@dataclass(frozen=True)
class EffectSizeResult:
    """Cohen's d effect size with qualitative interpretation."""

    cohens_d: float
    interpretation: str  # negligible / small / medium / large


@dataclass(frozen=True)
class FDRCorrectionResult:
    """Benjamini-Hochberg FDR correction output."""

    original_p_values: list[float]
    adjusted_p_values: list[float]
    rejected: list[bool]
    alpha: float


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


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------


def _t_cdf_fallback(t: float, df: float) -> float:
    """Approximate the CDF of Student's t-distribution using the regularized
    incomplete beta function identity.  Accuracy is sufficient for parity
    reporting when scipy is unavailable.
    """
    x = df / (df + t * t)
    # Regularized incomplete beta via continued-fraction approximation
    a = df / 2.0
    b = 0.5
    # Use the symmetry: I_x(a,b) = 1 - I_{1-x}(b,a) when needed
    result = _regularized_beta(x, a, b)
    if t >= 0:
        return 1.0 - 0.5 * result
    return 0.5 * result


def _regularized_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """Regularized incomplete beta function I_x(a,b) via Lentz's continued
    fraction algorithm.  Good enough for moderate df values used in parity
    checks.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use the identity I_x(a,b) = 1 - I_{1-x}(b,a) for numerical stability
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_beta(1.0 - x, b, a, max_iter)

    # Log-beta function
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta) / a

    # Lentz's continued fraction
    tiny = 1e-30
    f = 1.0 + tiny
    c = f
    d = 0.0

    for m in range(1, max_iter + 1):
        # Even step
        numerator: float
        m2 = 2 * m
        numerator = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        f *= c * d

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return front * (f - 1.0)


def welch_t_test(
    a: list[float],
    b: list[float],
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> WelchTTestResult:
    """Welch's t-test for two independent samples with unequal variances.

    Parameters
    ----------
    a, b:
        Score lists for the two groups (e.g. worldflux vs upstream).
    alpha:
        Significance level.
    alternative:
        ``"two-sided"``, ``"less"``, or ``"greater"``.

    Returns
    -------
    WelchTTestResult
    """
    if len(a) < 2 or len(b) < 2:
        raise ParitySampleSizeError("Welch's t-test requires at least 2 samples in each group")

    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    try:
        from scipy import stats as sp_stats  # type: ignore[import-untyped]

        result = sp_stats.ttest_ind(arr_a, arr_b, equal_var=False, alternative=alternative)
        t_stat = float(result.statistic)
        p_val = float(result.pvalue)
        n_a, n_b = len(a), len(b)
        var_a, var_b = float(arr_a.var(ddof=1)), float(arr_b.var(ddof=1))
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / denom if denom > 0 else 1.0
    except ImportError:
        n_a, n_b = len(a), len(b)
        mean_a, mean_b = float(arr_a.mean()), float(arr_b.mean())
        var_a = float(arr_a.var(ddof=1))
        var_b = float(arr_b.var(ddof=1))
        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            t_stat = 0.0
            df = 1.0
            p_val = 1.0
        else:
            t_stat = (mean_a - mean_b) / se
            num = (var_a / n_a + var_b / n_b) ** 2
            denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
            df = num / denom if denom > 0 else 1.0

            cdf_val = _t_cdf_fallback(abs(t_stat), df)
            if alternative == "two-sided":
                p_val = 2.0 * (1.0 - cdf_val)
            elif alternative == "less":
                p_val = _t_cdf_fallback(t_stat, df)
            elif alternative == "greater":
                p_val = 1.0 - _t_cdf_fallback(t_stat, df)
            else:
                raise ParityError(
                    f"alternative must be 'two-sided', 'less', or 'greater', got {alternative!r}"
                )

    return WelchTTestResult(
        t_statistic=_round_ratio(t_stat),
        p_value=_round_ratio(min(max(p_val, 0.0), 1.0)),
        df=_round_ratio(df),
        significant=bool(p_val < alpha),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Mann-Whitney U test
# ---------------------------------------------------------------------------


def mann_whitney_u_test(
    a: list[float],
    b: list[float],
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> MannWhitneyResult:
    """Mann-Whitney U test (non-parametric) for two independent samples.

    Parameters
    ----------
    a, b:
        Score lists for two groups.
    alpha:
        Significance level.
    alternative:
        ``"two-sided"``, ``"less"``, or ``"greater"``.

    Returns
    -------
    MannWhitneyResult
    """
    if len(a) < 1 or len(b) < 1:
        raise ParitySampleSizeError("Mann-Whitney U test requires at least 1 sample in each group")

    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    try:
        from scipy import stats as sp_stats  # type: ignore[import-untyped]

        result = sp_stats.mannwhitneyu(arr_a, arr_b, alternative=alternative)
        u_stat = float(result.statistic)
        p_val = float(result.pvalue)
    except ImportError:
        # Pure-Python fallback using normal approximation for large samples
        n_a, n_b = len(a), len(b)
        # Count how many b values each a value exceeds
        u_stat = 0.0
        for va in arr_a:
            for vb in arr_b:
                if va > vb:
                    u_stat += 1.0
                elif va == vb:
                    u_stat += 0.5

        mean_u = n_a * n_b / 2.0
        std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12.0)

        if std_u == 0:
            p_val = 1.0
        else:
            z = (u_stat - mean_u) / std_u
            # Normal CDF approximation
            cdf_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            if alternative == "two-sided":
                p_val = 2.0 * min(cdf_z, 1.0 - cdf_z)
            elif alternative == "less":
                p_val = cdf_z
            elif alternative == "greater":
                p_val = 1.0 - cdf_z
            else:
                raise ParityError(
                    f"alternative must be 'two-sided', 'less', or 'greater', got {alternative!r}"
                )

    return MannWhitneyResult(
        u_statistic=_round_ratio(u_stat),
        p_value=_round_ratio(min(max(p_val, 0.0), 1.0)),
        significant=bool(p_val < alpha),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Cohen's d effect size
# ---------------------------------------------------------------------------


def cohens_d(a: list[float], b: list[float]) -> EffectSizeResult:
    """Compute Cohen's d effect size between two independent samples.

    Uses the pooled standard deviation as the denominator.

    Parameters
    ----------
    a, b:
        Score lists.

    Returns
    -------
    EffectSizeResult
        Includes qualitative interpretation per Cohen (1988):
        negligible (< 0.2), small (0.2-0.5), medium (0.5-0.8), large (>= 0.8).
    """
    if len(a) < 2 or len(b) < 2:
        raise ParitySampleSizeError("Cohen's d requires at least 2 samples in each group")

    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    n_a, n_b = len(a), len(b)
    var_a = float(arr_a.var(ddof=1))
    var_b = float(arr_b.var(ddof=1))

    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        d = 0.0
    else:
        d = (float(arr_a.mean()) - float(arr_b.mean())) / pooled_std

    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizeResult(
        cohens_d=_round_ratio(d),
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------


def benjamini_hochberg(
    p_values: list[float],
    *,
    alpha: float = 0.05,
) -> FDRCorrectionResult:
    """Apply Benjamini-Hochberg FDR correction for multiple comparisons.

    Parameters
    ----------
    p_values:
        List of raw p-values from multiple tests (e.g. 26 Atari games).
    alpha:
        Desired false discovery rate.

    Returns
    -------
    FDRCorrectionResult
    """
    if not p_values:
        return FDRCorrectionResult(
            original_p_values=[],
            adjusted_p_values=[],
            rejected=[],
            alpha=alpha,
        )

    m = len(p_values)
    # Sort p-values and track original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    rejected = [False] * m

    # Adjust p-values from largest to smallest, enforcing monotonicity
    prev_adj = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1  # 1-based rank
        adj_p = min(p * m / rank, 1.0)
        adj_p = min(adj_p, prev_adj)  # enforce monotonicity
        adjusted[orig_idx] = _round_ratio(adj_p)
        rejected[orig_idx] = bool(adj_p < alpha)
        prev_adj = adj_p

    return FDRCorrectionResult(
        original_p_values=[_round_ratio(p) for p in p_values],
        adjusted_p_values=adjusted,
        rejected=rejected,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers for multi-seed results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregatedScores:
    """Aggregated statistics from multi-seed evaluation results."""

    mean: float
    std: float
    median: float
    iqr_low: float
    iqr_high: float
    ci_low: float
    ci_high: float
    n_samples: int


def aggregate_scores(
    scores: list[float],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 4000,
    seed: int = 0,
) -> AggregatedScores:
    """Compute mean, std, median, IQR, and bootstrap CI for a list of scores.

    Parameters
    ----------
    scores:
        Raw score values (e.g. from multiple seeds).
    confidence:
        Confidence level for the bootstrap CI.
    bootstrap_samples:
        Number of bootstrap re-samples.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    AggregatedScores
    """
    if not scores:
        raise ParitySampleSizeError("aggregate_scores requires at least one sample")

    arr = np.asarray(scores, dtype=np.float64)
    mean_val = float(arr.mean())
    std_val = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    median_val = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))

    # Bootstrap CI for the mean
    rng = np.random.default_rng(seed)
    boot_means = np.empty(bootstrap_samples, dtype=np.float64)
    for i in range(bootstrap_samples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - confidence
    ci_low = float(np.quantile(boot_means, alpha / 2.0))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

    return AggregatedScores(
        mean=_round_ratio(mean_val),
        std=_round_ratio(std_val),
        median=_round_ratio(median_val),
        iqr_low=_round_ratio(q25),
        iqr_high=_round_ratio(q75),
        ci_low=_round_ratio(ci_low),
        ci_high=_round_ratio(ci_high),
        n_samples=arr.size,
    )
