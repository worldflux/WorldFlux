"""Evaluation suite runner for world models."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from worldflux.evals.metrics import (
    imagination_coherence,
    latent_consistency,
    latent_utilization,
    reconstruction_fidelity,
    reward_prediction_accuracy,
)
from worldflux.evals.result import EvalReport, EvalResult

if TYPE_CHECKING:
    from worldflux.core.model import WorldModel

logger = logging.getLogger(__name__)

SUITE_CONFIGS: dict[str, list[str]] = {
    "quick": ["reconstruction_fidelity", "latent_consistency"],
    "standard": [
        "reconstruction_fidelity",
        "latent_consistency",
        "imagination_coherence",
        "latent_utilization",
    ],
    "comprehensive": [
        "reconstruction_fidelity",
        "latent_consistency",
        "imagination_coherence",
        "reward_prediction_accuracy",
        "latent_utilization",
    ],
}

_METRIC_FUNCTIONS: dict[str, Any] = {
    "reconstruction_fidelity": reconstruction_fidelity,
    "latent_consistency": latent_consistency,
    "imagination_coherence": imagination_coherence,
    "reward_prediction_accuracy": reward_prediction_accuracy,
    "latent_utilization": latent_utilization,
}


def _generate_synthetic_data(
    model: WorldModel,
    *,
    batch_size: int = 4,
    horizon: int = 10,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Generate synthetic test data appropriate for the model.

    Uses the model's config to determine observation shape and action
    dimensions, producing small random tensors for evaluation.
    """
    config = getattr(model, "config", None)
    obs_shape = tuple(getattr(config, "obs_shape", (3, 64, 64)))
    action_dim = int(getattr(config, "action_dim", 6))

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    obs = torch.randn(batch_size, *obs_shape, device=device)
    actions = torch.randn(horizon, batch_size, action_dim, device=device)
    rewards = torch.randn(horizon, batch_size, device=device)

    return {"obs": obs, "actions": actions, "rewards": rewards}


def run_eval_suite(
    model: WorldModel,
    *,
    suite: str = "quick",
    output_path: Path | str | None = None,
    device: torch.device | str | None = None,
    model_id: str = "unknown",
    batch_size: int = 4,
    horizon: int = 10,
) -> EvalReport:
    """Run an evaluation suite on a world model.

    Args:
        model: World model to evaluate.
        suite: Suite name from SUITE_CONFIGS (``quick``, ``standard``,
            ``comprehensive``).
        output_path: Optional path to save the JSON report.
        device: Device to run evaluation on.
        model_id: Identifier string for the model being evaluated.
        batch_size: Batch size for synthetic test data.
        horizon: Rollout horizon for imagination metrics.

    Returns:
        An :class:`EvalReport` with all metric results.
    """
    if suite not in SUITE_CONFIGS:
        raise ValueError(f"Unknown suite {suite!r}. Available: {sorted(SUITE_CONFIGS)}")

    metric_names = SUITE_CONFIGS[suite]
    start_time = time.time()

    if device is not None:
        device = torch.device(device) if isinstance(device, str) else device
        model = model.to(device)

    model.eval()

    data = _generate_synthetic_data(model, batch_size=batch_size, horizon=horizon, device=device)

    results: list[EvalResult] = []
    for name in metric_names:
        fn = _METRIC_FUNCTIONS.get(name)
        if fn is None:
            logger.warning("Unknown metric %r in suite %r, skipping", name, suite)
            continue

        if name == "reward_prediction_accuracy":
            result = fn(
                model,
                data["obs"],
                data["actions"],
                data["rewards"],
                suite=suite,
                model_id=model_id,
            )
        else:
            result = fn(
                model,
                data["obs"],
                data["actions"],
                suite=suite,
                model_id=model_id,
            )
        results.append(result)

    wall_time = time.time() - start_time

    pass_statuses = [r.passed for r in results if r.passed is not None]
    if pass_statuses:
        all_passed: bool | None = all(pass_statuses)
    else:
        all_passed = None
        logger.warning("All eval metrics returned inconclusive results")

    report = EvalReport(
        suite=suite,
        model_id=model_id,
        results=tuple(results),
        timestamp=start_time,
        wall_time_sec=wall_time,
        all_passed=all_passed,
    )

    if output_path is not None:
        report.save(Path(output_path))
        logger.info("Eval report saved to %s", output_path)

    return report
