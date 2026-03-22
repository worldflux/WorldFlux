# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Model evaluation framework for WorldFlux."""

from __future__ import annotations

from .env_policy import EnvPolicyRollout, collect_env_policy_rollout
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
    "EnvPolicyRollout",
    "EvalReport",
    "EvalResult",
    "SUITE_CONFIGS",
    "collect_env_policy_rollout",
    "imagination_coherence",
    "latent_consistency",
    "latent_utilization",
    "reconstruction_fidelity",
    "reward_prediction_accuracy",
    "run_eval_suite",
]
