"""Evaluation metric functions for world models."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from worldflux.evals.result import EvalResult

if TYPE_CHECKING:
    from worldflux.core.model import WorldModel

logger = logging.getLogger(__name__)


def reconstruction_fidelity(
    model: WorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    suite: str = "manual",
    model_id: str = "unknown",
) -> EvalResult:
    """Measure observation reconstruction quality via MSE.

    Encodes observations then decodes; computes MSE between original and
    reconstructed observations. Returns ``passed=None`` (informational) if the
    model does not expose a decoder.
    """
    ts = time.time()
    with torch.no_grad():
        state = model.encode(obs)

        try:
            output = model.decode(state)
        except Exception:
            logger.debug("Model does not support decode; skipping reconstruction_fidelity")
            return EvalResult(
                suite=suite,
                metric="reconstruction_fidelity",
                value=float("nan"),
                threshold=None,
                passed=None,
                timestamp=ts,
                model_id=model_id,
                metadata={"note": "Model does not support decode"},
            )

        reconstructed = output.predictions.get("obs")
        if reconstructed is None:
            return EvalResult(
                suite=suite,
                metric="reconstruction_fidelity",
                value=float("nan"),
                threshold=None,
                passed=None,
                timestamp=ts,
                model_id=model_id,
                metadata={"note": "Decoder output missing 'obs' key"},
            )

        mse = float(torch.nn.functional.mse_loss(reconstructed, obs).item())

    return EvalResult(
        suite=suite,
        metric="reconstruction_fidelity",
        value=mse,
        threshold=None,
        passed=None,
        timestamp=ts,
        model_id=model_id,
    )


def latent_consistency(
    model: WorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    suite: str = "manual",
    model_id: str = "unknown",
) -> EvalResult:
    """Check deterministic encoding consistency.

    Encodes the same observation twice and measures L2 distance between the
    resulting latent tensors. Models with stochastic encoders may produce
    nonzero distance; deterministic encoders should yield near-zero.
    """
    ts = time.time()
    threshold = 1e-5
    with torch.no_grad():
        state1 = model.encode(obs, deterministic=True)
        state2 = model.encode(obs, deterministic=True)

        distances: list[float] = []
        for key in state1.tensors:
            t1 = state1.tensors[key].float()
            t2 = state2.tensors.get(key)
            if t2 is None:
                continue
            distances.append(float(torch.norm(t1 - t2.float()).item()))

        l2_distance = max(distances) if distances else 0.0

    return EvalResult(
        suite=suite,
        metric="latent_consistency",
        value=l2_distance,
        threshold=threshold,
        passed=l2_distance < threshold,
        timestamp=ts,
        model_id=model_id,
    )


def imagination_coherence(
    model: WorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    horizon: int = 10,
    suite: str = "manual",
    model_id: str = "unknown",
) -> EvalResult:
    """Check that imagination rollouts produce finite, bounded outputs.

    Encodes an initial observation, rolls out ``horizon`` steps, and verifies
    that all state tensors remain finite (no NaN/Inf).
    """
    ts = time.time()
    with torch.no_grad():
        state = model.encode(obs)
        rollout_actions = actions[:horizon]

        try:
            trajectory = model.rollout(state, rollout_actions)
        except Exception as exc:
            logger.debug("Rollout failed in imagination_coherence: %s", exc)
            return EvalResult(
                suite=suite,
                metric="imagination_coherence",
                value=0.0,
                threshold=None,
                passed=False,
                timestamp=ts,
                model_id=model_id,
                metadata={"error": str(exc)},
            )

        all_finite = True
        for s in trajectory.states:
            for tensor in s.tensors.values():
                if not torch.isfinite(tensor).all():
                    all_finite = False
                    break
            if not all_finite:
                break

        if all_finite and trajectory.rewards is not None:
            if not torch.isfinite(trajectory.rewards).all():
                all_finite = False

    return EvalResult(
        suite=suite,
        metric="imagination_coherence",
        value=1.0 if all_finite else 0.0,
        threshold=None,
        passed=all_finite,
        timestamp=ts,
        model_id=model_id,
    )


def reward_prediction_accuracy(
    model: WorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    *,
    suite: str = "manual",
    model_id: str = "unknown",
) -> EvalResult:
    """Measure reward prediction accuracy via MSE.

    Encodes an initial observation, rolls out using the provided actions, and
    compares predicted rewards against ground truth.
    """
    ts = time.time()
    with torch.no_grad():
        state = model.encode(obs)

        try:
            trajectory = model.rollout(state, actions)
        except Exception as exc:
            logger.debug("Rollout failed in reward_prediction_accuracy: %s", exc)
            return EvalResult(
                suite=suite,
                metric="reward_prediction_accuracy",
                value=float("nan"),
                threshold=None,
                passed=None,
                timestamp=ts,
                model_id=model_id,
                metadata={"error": str(exc)},
            )

        if trajectory.rewards is None:
            return EvalResult(
                suite=suite,
                metric="reward_prediction_accuracy",
                value=float("nan"),
                threshold=None,
                passed=None,
                timestamp=ts,
                model_id=model_id,
                metadata={"note": "Model does not predict rewards"},
            )

        pred_rewards = trajectory.rewards
        # Align shapes: rewards may be (horizon, batch) or (horizon, batch, 1)
        target = rewards[: pred_rewards.shape[0]]
        if target.shape != pred_rewards.shape:
            target = target.reshape(pred_rewards.shape)

        mse = float(torch.nn.functional.mse_loss(pred_rewards, target).item())

    return EvalResult(
        suite=suite,
        metric="reward_prediction_accuracy",
        value=mse,
        threshold=None,
        passed=None,
        timestamp=ts,
        model_id=model_id,
    )


def latent_utilization(
    model: WorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    variance_threshold: float = 1e-4,
    suite: str = "manual",
    model_id: str = "unknown",
) -> EvalResult:
    """Measure latent space utilization.

    Encodes multiple observations and computes the fraction of latent
    dimensions that have variance above ``variance_threshold``, indicating
    active use of the latent capacity.
    """
    ts = time.time()
    with torch.no_grad():
        state = model.encode(obs)

        # Concatenate all state tensors into a single feature vector per sample.
        flat_parts: list[torch.Tensor] = []
        for tensor in state.tensors.values():
            flat_parts.append(tensor.float().flatten(start_dim=1))

        if not flat_parts:
            return EvalResult(
                suite=suite,
                metric="latent_utilization",
                value=0.0,
                threshold=None,
                passed=None,
                timestamp=ts,
                model_id=model_id,
                metadata={"note": "No latent tensors found"},
            )

        features = torch.cat(flat_parts, dim=-1)  # (batch, total_dim)
        variances = features.var(dim=0)  # (total_dim,)
        total_dims = int(variances.numel())
        active_dims = int((variances > variance_threshold).sum().item())
        ratio = active_dims / total_dims if total_dims > 0 else 0.0

    return EvalResult(
        suite=suite,
        metric="latent_utilization",
        value=ratio,
        threshold=None,
        passed=None,
        timestamp=ts,
        model_id=model_id,
        metadata={"active_dims": active_dims, "total_dims": total_dims},
    )
