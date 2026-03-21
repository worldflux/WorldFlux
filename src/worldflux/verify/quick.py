# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Synthetic smoke verification for pip-install users.

Provides model verification without requiring a source checkout or
``scripts/parity/``.  Loads a trained model, runs evaluation episodes,
and compares results against bundled baseline statistics using the
existing TOST non-inferiority engine.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .protocol import PROTOCOL_VERSION, QuickVerifyResult

logger = logging.getLogger(__name__)


class QualityTier(str, Enum):
    """Quality tier for training run evaluation."""

    SMOKE = "smoke"  # finite output, no NaN
    BASELINE = "baseline"  # above CI baseline
    PRODUCTION = "production"  # reference-level thresholds


class QuickVerifyTier(str, Enum):
    """Execution tier for quick verification."""

    SYNTHETIC = "synthetic"
    OFFLINE = "offline"
    REAL_ENV_SMOKE = "real_env_smoke"


@dataclass(frozen=True)
class QualityCheckResult:
    """Result of a quality tier check."""

    tier: QualityTier
    achieved_tier: QualityTier
    score: float  # 0.0-1.0
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


# Bundled baseline statistics for the synthetic smoke workload.
# Each entry maps env -> {mean, std, n, margin_ratio}.
# These are used as the reference for non-inferiority testing.
_BUILTIN_BASELINES: dict[str, dict[str, Any]] = {
    "atari/pong": {
        "model": "dreamer:ci",
        "mean": 0.85,
        "std": 0.12,
        "n": 50,
        "margin_ratio": 0.15,
    },
    "mujoco/halfcheetah": {
        "model": "tdmpc2:ci",
        "mean": 0.78,
        "std": 0.15,
        "n": 50,
        "margin_ratio": 0.15,
    },
}


def _load_baselines(baseline_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load baseline statistics from a JSON file or return builtins."""
    if baseline_path is not None and baseline_path.exists():
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return dict(_BUILTIN_BASELINES)


def _evaluate_model(
    model: torch.nn.Module,
    *,
    obs_shape: tuple[int, ...],
    action_dim: int,
    episodes: int,
    horizon: int,
    device: str,
) -> list[float]:
    """Run evaluation episodes and collect mean scores.

    Uses the model's rollout capability to generate trajectories and
    collect predicted rewards.  If the model doesn't predict rewards,
    uses reconstruction loss as a proxy metric.
    """
    model.eval()
    scores: list[float] = []
    torch_device = torch.device(device)

    with torch.no_grad():
        for ep in range(episodes):
            seed = 42 + ep
            rng = torch.Generator(device="cpu").manual_seed(seed)

            obs = torch.randn(1, *obs_shape, generator=rng).to(torch_device)
            actions = torch.randn(horizon, 1, action_dim, generator=rng).to(torch_device)

            try:
                state = model.encode(obs)  # type: ignore[operator]
                trajectory = model.rollout(state, actions)  # type: ignore[operator]

                if trajectory.rewards is not None:
                    episode_score = float(trajectory.rewards.sum().item())
                else:
                    # Use negative state norm as proxy when rewards unavailable
                    final_state = trajectory.states[-1]
                    state_tensor = next(iter(final_state.tensors.values()))
                    episode_score = float(-state_tensor.norm().item())
            except (NotImplementedError, RuntimeError, AttributeError) as exc:
                logger.debug("Episode %d evaluation fallback: %s", ep, exc)
                episode_score = 0.0

            scores.append(episode_score)

    return scores


def _compute_drop_ratios(
    scores: list[float],
    baseline_mean: float,
) -> list[float]:
    """Compute per-episode drop ratios relative to baseline mean.

    Returns (baseline_mean - score) / |baseline_mean| for each episode.
    Positive values indicate the model underperforms the baseline.
    """
    if abs(baseline_mean) < 1e-10:
        return [0.0] * len(scores)
    return [(baseline_mean - s) / abs(baseline_mean) for s in scores]


def _normalize_quick_verify_tier(tier: str | QuickVerifyTier) -> QuickVerifyTier:
    if isinstance(tier, QuickVerifyTier):
        return tier
    normalized = str(tier).strip().lower()
    for candidate in QuickVerifyTier:
        if candidate.value == normalized:
            return candidate
    raise ValueError(
        "Unknown verification tier: "
        f"{tier!r}. Expected one of {[candidate.value for candidate in QuickVerifyTier]}"
    )


def quick_verify(
    target: str,
    *,
    env: str = "atari/pong",
    tier: str | QuickVerifyTier = QuickVerifyTier.SYNTHETIC,
    episodes: int = 10,
    horizon: int = 15,
    device: str = "cpu",
    baseline_path: Path | None = None,
) -> QuickVerifyResult:
    """Run synthetic smoke verification of a trained model checkpoint.

    Parameters
    ----------
    target:
        Path to a checkpoint directory (containing ``config.json`` and
        ``model.pt``) or a single ``.pt`` checkpoint file.
    env:
        Environment key for selecting baseline statistics.
    episodes:
        Number of evaluation episodes to run.
    horizon:
        Rollout horizon per episode.
    device:
        Device for model execution.
    baseline_path:
        Optional path to custom baseline statistics JSON file.

    Returns
    -------
    QuickVerifyResult
        Verification result with pass/fail verdict and statistics.
    """
    from worldflux.parity.stats import non_inferiority_test

    start = time.monotonic()
    verification_tier = _normalize_quick_verify_tier(tier)

    # Load baseline statistics
    baselines = _load_baselines(baseline_path)
    baseline = baselines.get(env)
    if baseline is None:
        # Fall back to generic baseline
        baseline = {"mean": 0.0, "std": 1.0, "n": 50, "margin_ratio": 0.20}
        logger.warning("No baseline found for env=%r, using generic threshold", env)

    baseline_mean = float(baseline["mean"])
    margin_ratio = float(baseline.get("margin_ratio", 0.15))

    # Load model
    target_path = Path(target)
    model = _load_model_from_target(target_path, device=device)

    # Extract model config for evaluation parameters
    config = getattr(model, "config", None)
    obs_shape = tuple(getattr(config, "obs_shape", (3, 64, 64)))
    action_dim = int(getattr(config, "action_dim", 6))

    # Run evaluation
    eval_horizon = int(horizon)
    if verification_tier == QuickVerifyTier.REAL_ENV_SMOKE:
        # Keep smoke runs deliberately short until a real-env runner is introduced.
        eval_horizon = min(eval_horizon, 5)
    scores = _evaluate_model(
        model,
        obs_shape=obs_shape,
        action_dim=action_dim,
        episodes=episodes,
        horizon=eval_horizon,
        device=device,
    )

    mean_score = float(np.mean(scores))

    # Compute drop ratios and run non-inferiority test
    drop_ratios = _compute_drop_ratios(scores, baseline_mean)
    ni_result = non_inferiority_test(
        drop_ratios,
        margin_ratio=margin_ratio,
        confidence=0.95,
        bootstrap_samples=2000,
        min_samples=2,
    )

    elapsed = time.monotonic() - start
    passed = ni_result.pass_non_inferiority

    stats = {
        "verification_tier": verification_tier.value,
        "mean_drop_ratio": ni_result.mean_drop_ratio,
        "ci_upper_ratio": ni_result.ci_upper_ratio,
        "ci_lower_ratio": ni_result.ci_lower_ratio,
        "margin_ratio": ni_result.margin_ratio,
        "sample_size": ni_result.sample_size,
        "confidence": ni_result.confidence,
        "scores": scores,
        "baseline_mean": baseline_mean,
        "baseline_std": float(baseline.get("std", 0.0)),
        "evaluation_horizon": eval_horizon,
    }

    return QuickVerifyResult(
        passed=passed,
        target=str(target),
        env=env,
        episodes=episodes,
        mean_score=mean_score,
        baseline_mean=baseline_mean,
        elapsed_seconds=round(elapsed, 3),
        protocol_version=PROTOCOL_VERSION,
        stats=stats,
        verdict_reason=ni_result.verdict_reason
        or f"{verification_tier.value} quick verification workload only",
    )


def _load_model_from_target(target_path: Path, *, device: str) -> torch.nn.Module:
    """Load a world model from a checkpoint path.

    Supports two layouts:
    1. Directory with ``config.json`` + ``model.pt`` (save_pretrained format)
    2. Single ``.pt`` file (Trainer checkpoint with ``model_state_dict``)
    """
    from worldflux.core.registry import WorldModelRegistry

    if target_path.is_dir():
        # save_pretrained directory layout
        config_json = target_path / "config.json"
        model_pt = target_path / "model.pt"

        if config_json.exists() and model_pt.exists():
            return WorldModelRegistry.from_pretrained(str(target_path), device=device)

        # Check for trainer checkpoint in the directory
        checkpoint_candidates = [
            target_path / "checkpoint_final.pt",
            target_path / "checkpoint_best.pt",
        ]
        for candidate in checkpoint_candidates:
            if candidate.exists():
                return _load_from_trainer_checkpoint(candidate, device=device)

        raise FileNotFoundError(
            f"No model found in {target_path}. Expected config.json + model.pt "
            f"or a trainer checkpoint file."
        )

    if target_path.is_file() and target_path.suffix == ".pt":
        return _load_from_trainer_checkpoint(target_path, device=device)

    raise FileNotFoundError(f"Target not found: {target_path}")


def _load_from_trainer_checkpoint(checkpoint_path: Path, *, device: str) -> torch.nn.Module:
    """Load model from a Trainer-format checkpoint (``.pt`` file)."""
    checkpoint = torch.load(  # nosec B614
        checkpoint_path, map_location=device, weights_only=False
    )

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint format: {checkpoint_path}. "
            f"Expected a Trainer checkpoint with 'model_state_dict'."
        )

    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, dict):
        model_config = {}

    model = _build_model_from_config_payload(
        model_config,
        device=device,
        require_model_name=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _build_model_from_config_payload(
    config_payload: dict[str, Any],
    *,
    device: str,
    require_model_name: bool = False,
) -> torch.nn.Module:
    from worldflux.core.config import WorldModelConfig
    from worldflux.core.registry import ConfigRegistry, WorldModelRegistry

    if not isinstance(config_payload, dict):
        raise ValueError("Saved checkpoint is missing a valid model_config dictionary.")

    normalized = dict(config_payload)
    model_type = str(normalized.get("model_type", "")).strip().lower()
    if not model_type:
        raise ValueError("Saved checkpoint is missing model_config.model_type.")

    model_name = str(normalized.get("model_name", "")).strip()
    if require_model_name and not model_name:
        raise ValueError(
            "Saved checkpoint is missing model_config.model_name required for exact restoration."
        )

    normalized["device"] = device

    preset_config: Any | None = None
    if model_name:
        try:
            preset_config = ConfigRegistry.from_pretrained(
                f"{model_type}:{model_name}", device=device
            )
        except Exception:
            preset_config = None

    if preset_config is not None:
        merged = preset_config.to_dict()
        merged.update(normalized)
        config_class = type(preset_config)
        config = config_class.from_dict(merged)
    else:
        config_class = ConfigRegistry._registry.get(model_type, WorldModelConfig)
        config = config_class.from_dict(normalized)

    WorldModelRegistry.load_entrypoint_plugins()
    WorldModelRegistry._load_builtin_models()
    model_class = WorldModelRegistry._model_registry.get(model_type)
    if model_class is None:
        raise ValueError(f"Unsupported model_type in saved checkpoint: {model_type!r}")
    return model_class(config)


def quality_check(
    model: torch.nn.Module,
    *,
    tier: QualityTier = QualityTier.SMOKE,
    device: str = "cpu",
) -> QualityCheckResult:
    """Run a quality check on a trained model.

    Parameters
    ----------
    model:
        Trained world model to check.
    tier:
        Target quality tier to check against.
    device:
        Device for model execution.

    Returns
    -------
    QualityCheckResult
        Result with achieved tier and score.
    """
    model.eval()
    torch_device = torch.device(device)
    model = model.to(torch_device)

    config = getattr(model, "config", None)
    obs_shape = tuple(getattr(config, "obs_shape", (3, 64, 64)))
    action_dim = int(getattr(config, "action_dim", 6))

    details: dict[str, Any] = {}
    achieved = QualityTier.SMOKE
    score = 0.0

    # SMOKE check: finite outputs, no NaN
    smoke_ok = _check_smoke(model, obs_shape, action_dim, torch_device, details)
    if not smoke_ok:
        return QualityCheckResult(
            tier=tier,
            achieved_tier=QualityTier.SMOKE,
            score=0.0,
            passed=False,
            details=details,
        )
    score = 0.33

    if tier == QualityTier.SMOKE:
        return QualityCheckResult(
            tier=tier,
            achieved_tier=QualityTier.SMOKE,
            score=score,
            passed=True,
            details=details,
        )

    # BASELINE check: eval suite passes
    baseline_ok = _check_baseline(model, device, details)
    if baseline_ok:
        achieved = QualityTier.BASELINE
        score = 0.66

    if tier == QualityTier.BASELINE:
        return QualityCheckResult(
            tier=tier,
            achieved_tier=achieved,
            score=score,
            passed=baseline_ok,
            details=details,
        )

    # PRODUCTION check: comprehensive eval suite
    production_ok = _check_production(model, device, details)
    if production_ok:
        achieved = QualityTier.PRODUCTION
        score = 1.0

    return QualityCheckResult(
        tier=tier,
        achieved_tier=achieved,
        score=score,
        passed=production_ok,
        details=details,
    )


def _check_smoke(
    model: torch.nn.Module,
    obs_shape: tuple[int, ...],
    action_dim: int,
    device: torch.device,
    details: dict[str, Any],
) -> bool:
    """SMOKE tier: verify finite outputs and no NaN."""
    try:
        with torch.no_grad():
            obs = torch.randn(1, *obs_shape, device=device)
            actions = torch.randn(5, 1, action_dim, device=device)

            state = model.encode(obs)  # type: ignore[attr-defined, operator]

            # Check state tensors are finite
            for key, tensor in state.tensors.items():
                if not torch.isfinite(tensor).all():
                    details["smoke_failure"] = f"Non-finite state tensor: {key}"
                    return False

            trajectory = model.rollout(state, actions)  # type: ignore[attr-defined, operator]

            # Check trajectory states
            for i, s in enumerate(trajectory.states):
                for key, tensor in s.tensors.items():
                    if not torch.isfinite(tensor).all():
                        details["smoke_failure"] = f"Non-finite trajectory state at step {i}: {key}"
                        return False

            if trajectory.rewards is not None:
                if not torch.isfinite(trajectory.rewards).all():
                    details["smoke_failure"] = "Non-finite rewards in trajectory"
                    return False

        details["smoke_passed"] = True
        return True
    except Exception as exc:
        details["smoke_failure"] = str(exc)
        return False


def _check_baseline(
    model: torch.nn.Module,
    device: str,
    details: dict[str, Any],
) -> bool:
    """BASELINE tier: run quick eval suite and check consistency."""
    try:
        from worldflux.evals.suite import run_eval_suite

        report = run_eval_suite(
            model,  # type: ignore[arg-type]
            suite="quick",
            model_id=type(model).__name__,
        )
        details["baseline_eval_results"] = report.to_dict()

        # Pass if no explicit failures
        if report.all_passed is False:
            return False
        return True
    except Exception as exc:
        details["baseline_failure"] = str(exc)
        return False


def _check_production(
    model: torch.nn.Module,
    device: str,
    details: dict[str, Any],
) -> bool:
    """PRODUCTION tier: run standard eval suite."""
    try:
        from worldflux.evals.suite import run_eval_suite

        report = run_eval_suite(
            model,  # type: ignore[arg-type]
            suite="standard",
            model_id=type(model).__name__,
        )
        details["production_eval_results"] = report.to_dict()

        if report.all_passed is False:
            return False
        return True
    except Exception as exc:
        details["production_failure"] = str(exc)
        return False
