#!/usr/bin/env python3
"""Bayesian bootstrap helpers for parity effect-size reports."""

from __future__ import annotations

import math
import random
from typing import Any


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0.0:
        return float(sorted_values[0])
    if q >= 1.0:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight)


def bayesian_bootstrap_means(*, effects: list[float], draws: int, seed: int) -> list[float]:
    if draws <= 0:
        raise ValueError("draws must be > 0")
    if not effects:
        return []

    rng = random.Random(seed)
    n = len(effects)
    means: list[float] = []
    for _ in range(draws):
        weights = [rng.gammavariate(1.0, 1.0) for _ in range(n)]
        total = sum(weights)
        if total <= 0.0:
            means.append(sum(effects) / float(n))
            continue
        weighted = sum(effect * weight for effect, weight in zip(effects, weights, strict=True))
        means.append(weighted / total)
    return means


def bayesian_equivalence_report(
    *,
    effects: list[float],
    draws: int,
    seed: int,
    lower_equivalence: float,
    upper_equivalence: float,
    lower_noninferiority: float,
    probability_threshold_equivalence: float,
    probability_threshold_noninferiority: float,
    min_pairs: int,
) -> dict[str, Any]:
    n_pairs = len(effects)
    if n_pairs < min_pairs:
        return {
            "status": "insufficient_pairs",
            "n_pairs": n_pairs,
            "reason": f"Need >= {min_pairs} paired seeds",
        }

    posterior = bayesian_bootstrap_means(effects=effects, draws=draws, seed=seed)
    if not posterior:
        return {
            "status": "insufficient_pairs",
            "n_pairs": n_pairs,
            "reason": f"Need >= {min_pairs} paired seeds",
        }

    posterior_sorted = sorted(posterior)
    p_equivalence = sum(
        1 for value in posterior if lower_equivalence <= value <= upper_equivalence
    ) / float(len(posterior))
    p_noninferior = sum(1 for value in posterior if value >= lower_noninferiority) / float(
        len(posterior)
    )
    posterior_mean = sum(posterior) / float(len(posterior))

    return {
        "status": "ok",
        "n_pairs": n_pairs,
        "draws": draws,
        "seed": seed,
        "posterior_mean": posterior_mean,
        "posterior_ci90": [
            _quantile(posterior_sorted, 0.05),
            _quantile(posterior_sorted, 0.95),
        ],
        "posterior_ci95": [
            _quantile(posterior_sorted, 0.025),
            _quantile(posterior_sorted, 0.975),
        ],
        "p_equivalence": p_equivalence,
        "p_noninferior": p_noninferior,
        "probability_threshold_equivalence": probability_threshold_equivalence,
        "probability_threshold_noninferiority": probability_threshold_noninferiority,
        "pass_equivalence": p_equivalence >= probability_threshold_equivalence,
        "pass_noninferiority": p_noninferior >= probability_threshold_noninferiority,
        "pass_all": (
            p_equivalence >= probability_threshold_equivalence
            and p_noninferior >= probability_threshold_noninferiority
        ),
    }


__all__ = [
    "bayesian_bootstrap_means",
    "bayesian_equivalence_report",
]
