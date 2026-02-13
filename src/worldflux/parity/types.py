"""Typed objects for parity harness run artifacts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScorePoint:
    """A scalar score observed at a specific task/seed/step."""

    task: str
    seed: int
    step: int
    score: float

    @property
    def key(self) -> tuple[str, int]:
        return (self.task, self.seed)


@dataclass(frozen=True)
class NonInferiorityResult:
    """Result of one-sided non-inferiority test over score drops."""

    sample_size: int
    mean_drop_ratio: float
    ci_lower_ratio: float
    ci_upper_ratio: float
    confidence: float
    margin_ratio: float
    pass_non_inferiority: bool
    verdict_reason: str | None = None
