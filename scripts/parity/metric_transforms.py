#!/usr/bin/env python3
"""Effect-size transforms for parity statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass

SUPPORTED_EFFECT_TRANSFORMS: set[str] = {
    "paired_log_ratio",
    "normalized_difference",
}


@dataclass(frozen=True)
class EffectBounds:
    lower_equivalence: float
    upper_equivalence: float
    lower_noninferiority: float


def _directional_delta(
    *,
    official: float,
    worldflux: float,
    higher_is_better: bool,
) -> tuple[float, float]:
    if higher_is_better:
        return official, worldflux
    return -official, -worldflux


def paired_log_ratio(
    *,
    official: float,
    worldflux: float,
    higher_is_better: bool = True,
    eps: float = 1e-8,
) -> float:
    """Return paired log-ratio with shift for non-positive values."""
    off, wf = _directional_delta(
        official=float(official),
        worldflux=float(worldflux),
        higher_is_better=bool(higher_is_better),
    )

    floor = min(off, wf)
    shift = (-floor + float(eps)) if floor <= 0 else 0.0
    numerator = wf + shift + float(eps)
    denominator = off + shift + float(eps)
    return math.log(numerator / denominator)


def normalized_difference(
    *,
    official: float,
    worldflux: float,
    higher_is_better: bool = True,
    eps: float = 1e-8,
) -> float:
    """Return normalized paired difference anchored to official scale."""
    off, wf = _directional_delta(
        official=float(official),
        worldflux=float(worldflux),
        higher_is_better=bool(higher_is_better),
    )
    scale = abs(off) + float(eps)
    return (wf - off) / scale


def transform_pair(
    *,
    transform: str,
    official: float,
    worldflux: float,
    higher_is_better: bool,
    eps: float,
) -> float:
    normalized = transform.strip().lower()
    if normalized == "paired_log_ratio":
        return paired_log_ratio(
            official=official,
            worldflux=worldflux,
            higher_is_better=higher_is_better,
            eps=eps,
        )
    if normalized == "normalized_difference":
        return normalized_difference(
            official=official,
            worldflux=worldflux,
            higher_is_better=higher_is_better,
            eps=eps,
        )
    raise ValueError(
        f"Unsupported effect transform '{transform}'. Supported: {sorted(SUPPORTED_EFFECT_TRANSFORMS)}"
    )


def equivalence_bounds(
    *,
    transform: str,
    equivalence_margin: float,
    noninferiority_margin: float,
) -> EffectBounds:
    normalized = transform.strip().lower()
    if normalized == "paired_log_ratio":
        return EffectBounds(
            lower_equivalence=math.log(1.0 - float(equivalence_margin)),
            upper_equivalence=math.log(1.0 + float(equivalence_margin)),
            lower_noninferiority=math.log(1.0 - float(noninferiority_margin)),
        )
    if normalized == "normalized_difference":
        margin = float(equivalence_margin)
        return EffectBounds(
            lower_equivalence=-margin,
            upper_equivalence=margin,
            lower_noninferiority=-float(noninferiority_margin),
        )
    raise ValueError(
        f"Unsupported effect transform '{transform}'. Supported: {sorted(SUPPORTED_EFFECT_TRANSFORMS)}"
    )


__all__ = [
    "EffectBounds",
    "SUPPORTED_EFFECT_TRANSFORMS",
    "equivalence_bounds",
    "normalized_difference",
    "paired_log_ratio",
    "transform_pair",
]
