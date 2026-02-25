"""Model evaluation framework for WorldFlux."""

from __future__ import annotations

from .metrics import (
    imagination_coherence,
    latent_consistency,
    latent_utilization,
    reconstruction_fidelity,
    reward_prediction_accuracy,
)
from .result import EvalReport, EvalResult
from .suite import SUITE_CONFIGS, run_eval_suite

__all__ = [
    "EvalReport",
    "EvalResult",
    "SUITE_CONFIGS",
    "imagination_coherence",
    "latent_consistency",
    "latent_utilization",
    "reconstruction_fidelity",
    "reward_prediction_accuracy",
    "run_eval_suite",
]
