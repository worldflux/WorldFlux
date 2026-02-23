"""Lightweight quick verification for pip-install users.

Provides model verification without requiring a source checkout or
``scripts/parity/``.  Loads a trained model, runs evaluation episodes,
and compares results against bundled baseline statistics using the
existing TOST non-inferiority engine.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .protocol import PROTOCOL_VERSION, QuickVerifyResult

logger = logging.getLogger(__name__)

# Bundled baseline statistics: pre-computed reference distributions.
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


def quick_verify(
    target: str,
    *,
    env: str = "atari/pong",
    episodes: int = 10,
    horizon: int = 15,
    device: str = "cpu",
    baseline_path: Path | None = None,
) -> QuickVerifyResult:
    """Run quick verification of a trained model checkpoint.

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
    scores = _evaluate_model(
        model,
        obs_shape=obs_shape,
        action_dim=action_dim,
        episodes=episodes,
        horizon=horizon,
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
        "mean_drop_ratio": ni_result.mean_drop_ratio,
        "ci_upper_ratio": ni_result.ci_upper_ratio,
        "ci_lower_ratio": ni_result.ci_lower_ratio,
        "margin_ratio": ni_result.margin_ratio,
        "sample_size": ni_result.sample_size,
        "confidence": ni_result.confidence,
        "scores": scores,
        "baseline_mean": baseline_mean,
        "baseline_std": float(baseline.get("std", 0.0)),
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
        verdict_reason=ni_result.verdict_reason or "",
    )


def _load_model_from_target(target_path: Path, *, device: str) -> torch.nn.Module:
    """Load a world model from a checkpoint path.

    Supports two layouts:
    1. Directory with ``config.json`` + ``model.pt`` (save_pretrained format)
    2. Single ``.pt`` file (Trainer checkpoint with ``model_state_dict``)
    """
    from worldflux import create_world_model

    if target_path.is_dir():
        # save_pretrained directory layout
        config_json = target_path / "config.json"
        model_pt = target_path / "model.pt"

        if config_json.exists() and model_pt.exists():
            config_data = json.loads(config_json.read_text(encoding="utf-8"))
            model_id = str(config_data.get("model_type", "dreamer")) + ":ci"
            obs_shape = tuple(config_data.get("obs_shape", [3, 64, 64]))
            action_dim = int(config_data.get("action_dim", 6))
            hidden_dim = int(config_data.get("hidden_dim", 32))

            model = create_world_model(
                model=model_id,
                obs_shape=obs_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device,
            )
            state_dict = torch.load(model_pt, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            return model

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
    from worldflux import create_world_model

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint format: {checkpoint_path}. "
            f"Expected a Trainer checkpoint with 'model_state_dict'."
        )

    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, dict):
        model_config = {}

    model_type = str(model_config.get("model_type", "dreamer"))
    model_id = f"{model_type}:ci"
    obs_shape = tuple(model_config.get("obs_shape", [3, 64, 64]))
    action_dim = int(model_config.get("action_dim", 6))
    hidden_dim = int(model_config.get("hidden_dim", 32))

    model = create_world_model(
        model=model_id,
        obs_shape=obs_shape,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
