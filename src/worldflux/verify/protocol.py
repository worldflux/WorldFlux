"""Verify protocol specification and versioning.

Defines the protocol version, result types, and metadata for the
WorldFlux verification system.  Both ``quick`` and ``proof`` modes
emit results conforming to this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "1.0"


@dataclass(frozen=True)
class QuickVerifyResult:
    """Result of a quick (lightweight) verification run.

    This is the primary result type for ``worldflux verify`` when used
    by pip-install users who verify a checkpoint against bundled
    baseline statistics.

    Attributes
    ----------
    passed:
        Whether the model passes the equivalence threshold.
    target:
        Path or identifier of the model being verified.
    env:
        Environment used for evaluation (e.g. ``"atari/pong"``).
    episodes:
        Number of evaluation episodes run.
    mean_score:
        Mean score achieved by the model across episodes.
    baseline_mean:
        Mean score of the reference baseline distribution.
    elapsed_seconds:
        Wall-clock time for the verification run.
    protocol_version:
        Verification protocol version (always ``PROTOCOL_VERSION``).
    stats:
        Detailed statistical test results (TOST, Bayesian, etc.).
    verdict_reason:
        Human-readable reason for the pass/fail verdict.
    """

    passed: bool
    target: str
    env: str
    episodes: int
    mean_score: float
    baseline_mean: float
    elapsed_seconds: float
    protocol_version: str = PROTOCOL_VERSION
    stats: dict[str, Any] = field(default_factory=dict)
    verdict_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "passed": self.passed,
            "target": self.target,
            "env": self.env,
            "episodes": self.episodes,
            "mean_score": self.mean_score,
            "baseline_mean": self.baseline_mean,
            "elapsed_seconds": self.elapsed_seconds,
            "protocol_version": self.protocol_version,
            "stats": dict(self.stats),
            "verdict_reason": self.verdict_reason,
        }
