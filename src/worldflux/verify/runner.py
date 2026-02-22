"""Parity verification runner for end-user model validation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerifyResult:
    """Result of a parity verification run."""

    passed: bool
    target: str
    baseline: str
    env: str
    demo: bool
    elapsed_seconds: float
    stats: dict[str, Any] = field(default_factory=dict)
    verdict_reason: str = ""


class ParityVerifier:
    """Run parity verification between a target model and an official baseline."""

    @classmethod
    def run(
        cls,
        *,
        target: str,
        baseline: str = "official/dreamerv3",
        env: str = "atari/pong",
        demo: bool = False,
        device: str = "cpu",
    ) -> VerifyResult:
        """Execute verification and return a :class:`VerifyResult`.

        Parameters
        ----------
        target:
            Path or registry ID of the custom model.
        baseline:
            Baseline model identifier.
        env:
            Target simulation environment.
        demo:
            When *True*, return synthetic results suitable for demonstrations.
        device:
            Execution device string.
        """
        if demo:
            return cls._run_demo(target=target, baseline=baseline, env=env, device=device)
        return cls._run_real(target=target, baseline=baseline, env=env, device=device)

    @classmethod
    def _run_demo(
        cls,
        *,
        target: str,
        baseline: str,
        env: str,
        device: str,
    ) -> VerifyResult:
        start = time.monotonic()
        rng = random.Random(42)
        samples = 500
        mean_drop = rng.uniform(0.001, 0.015)
        ci_upper = mean_drop + rng.uniform(0.005, 0.02)
        margin = 0.05
        bayesian_hdi = round(rng.uniform(0.975, 0.995), 3)
        tost_p = round(rng.uniform(0.005, 0.025), 3)
        time.sleep(rng.uniform(2.5, 3.5))
        elapsed = time.monotonic() - start
        return VerifyResult(
            passed=True,
            target=target,
            baseline=baseline,
            env=env,
            demo=True,
            elapsed_seconds=round(elapsed, 3),
            stats={
                "samples": samples,
                "mean_drop_ratio": round(mean_drop, 6),
                "ci_upper_ratio": round(ci_upper, 6),
                "margin_ratio": margin,
                "bayesian_equivalence_hdi": bayesian_hdi,
                "tost_p_value": tost_p,
                "device": device,
            },
            verdict_reason="Demo mode: synthetic pass",
        )

    @classmethod
    def _run_real(
        cls,
        *,
        target: str,
        baseline: str,
        env: str,
        device: str,
    ) -> VerifyResult:
        raise NotImplementedError(
            "Real parity verification is not yet available. "
            "Use --demo for synthetic results or run the internal parity suite directly "
            "via `worldflux parity proof-run`."
        )
