# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Execution bridge for Dreamer sensitivity campaigns."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sensitivity import SensitivityAnalysis, SensitivityReport, SweepResult

DEFAULT_TASK_ID = "atari100k_pong"
DEFAULT_ENV_BACKEND = "stub"
DEFAULT_MODEL_PROFILE = "wf12m"
DEFAULT_RUN_ROOT = Path("outputs/sensitivity/dreamerv3")
_SUPPORTED_SWEEP_NAMES = {
    "kl_dynamics",
    "kl_representation",
    "free_nats",
    "learning_rate",
    "imagination_horizon",
}
_DEFAULT_DREAMER_LOSS_SCALES = {
    "reconstruction": 1.0,
    "kl_dynamics": 0.5,
    "kl_representation": 0.1,
    "reward": 1.0,
    "continue": 1.0,
}


@dataclass(frozen=True)
class SensitivityRunnerLaunch:
    """Concrete subprocess launch description for one sweep config."""

    command: list[str]
    metrics_out: Path
    param_name: str
    param_value: float
    seed: int
    steps: int


def build_dreamer_runner_override_payload(overrides: dict[str, float]) -> dict[str, Any]:
    """Map sensitivity sweep values onto Dreamer runner/model overrides."""
    unknown = sorted(set(overrides) - _SUPPORTED_SWEEP_NAMES)
    if unknown:
        raise ValueError(f"Unsupported Dreamer sensitivity overrides: {unknown}")

    payload: dict[str, Any] = {
        "learning_rate_override": float(overrides["learning_rate"]),
        "model_config_overrides": {
            "kl_free": float(overrides["free_nats"]),
            "imagination_horizon": int(float(overrides["imagination_horizon"])),
            "loss_scales": {
                **_DEFAULT_DREAMER_LOSS_SCALES,
                "kl_dynamics": float(overrides["kl_dynamics"]),
                "kl_representation": float(overrides["kl_representation"]),
            },
        },
    }
    return payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _wrapper_path() -> Path:
    return _repo_root() / "scripts" / "parity" / "wrappers" / "worldflux_native_online_runner.py"


def _slug_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "neg").replace(".", "p")


def build_runner_launch(
    run_config: dict[str, Any],
    *,
    lane: str,
    task_id: str,
    env_backend: str,
    device: str,
    model_profile: str,
    run_root: Path,
    python_executable: str | None = None,
) -> SensitivityRunnerLaunch:
    """Build the concrete runner command for one sweep configuration."""
    param_name = str(run_config["param_name"])
    param_value = float(run_config["param_value"])
    seed = int(run_config["seed"])
    steps = int(run_config["total_steps"])
    override_payload = build_dreamer_runner_override_payload(
        {str(key): float(value) for key, value in dict(run_config["overrides"]).items()}
    )

    run_dir = run_root / param_name / f"value_{_slug_value(param_value)}" / f"seed_{seed}"
    metrics_out = run_dir / "metrics.json"
    smoke_mode = str(lane).strip().lower() == "smoke"

    command = [
        python_executable or sys.executable,
        str(_wrapper_path()),
        "--family",
        "dreamerv3",
        "--task-id",
        task_id,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--eval-interval",
        str(max(1, steps) if smoke_mode else max(24, steps // 2)),
        "--eval-episodes",
        "1" if smoke_mode else "3",
        "--eval-window",
        "1" if smoke_mode else "3",
        "--env-backend",
        env_backend,
        "--device",
        device,
        "--buffer-capacity",
        "64" if smoke_mode else "256",
        "--warmup-steps",
        "1" if smoke_mode else "8",
        "--train-steps-per-eval",
        "1",
        "--sequence-length",
        "2" if smoke_mode else "4",
        "--batch-size",
        "2" if smoke_mode else "4",
        "--max-episode-steps",
        "4" if smoke_mode else "512",
        "--dreamer-train-ratio",
        "1",
        "--dreamer-replay-ratio",
        "1",
        "--dreamer-train-chunk-size",
        "1",
        "--dreamer-model-profile",
        model_profile,
        "--dreamer-config-overrides-json",
        json.dumps(override_payload, sort_keys=True),
        "--run-dir",
        str(run_dir),
        "--metrics-out",
        str(metrics_out),
    ]
    return SensitivityRunnerLaunch(
        command=command,
        metrics_out=metrics_out,
        param_name=param_name,
        param_value=param_value,
        seed=seed,
        steps=steps,
    )


def load_runner_metrics(metrics_path: Path) -> dict[str, Any]:
    """Load and validate one runner metrics payload."""
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Sensitivity runner payload must be a JSON object: {metrics_path}")
    if str(payload.get("schema_version", "")).strip() != "parity.v1":
        raise ValueError(f"Unexpected sensitivity runner schema: {metrics_path}")
    if not bool(payload.get("success", False)):
        raise ValueError(f"Sensitivity runner reported failure: {metrics_path}")
    if "final_return_mean" not in payload:
        raise ValueError(f"Sensitivity runner payload missing final_return_mean: {metrics_path}")
    return payload


def run_sweep_config(
    run_config: dict[str, Any],
    *,
    lane: str,
    task_id: str,
    env_backend: str,
    device: str,
    model_profile: str,
    run_root: Path,
    python_executable: str | None = None,
) -> SweepResult:
    """Execute a single sweep configuration and return its aggregate result."""
    launch = build_runner_launch(
        run_config,
        lane=lane,
        task_id=task_id,
        env_backend=env_backend,
        device=device,
        model_profile=model_profile,
        run_root=run_root,
        python_executable=python_executable,
    )
    completed = subprocess.run(
        launch.command,
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown runner failure"
        raise RuntimeError(
            f"Sensitivity run failed for {launch.param_name}={launch.param_value} seed={launch.seed}: {message}"
        )

    payload = load_runner_metrics(launch.metrics_out)
    return SweepResult(
        param_name=launch.param_name,
        param_value=launch.param_value,
        seed=launch.seed,
        final_reward=float(payload["final_return_mean"]),
        steps=launch.steps,
    )


def _validate_results(results: list[SweepResult], expected_total: int) -> None:
    if len(results) != expected_total:
        raise ValueError(
            f"Incomplete sensitivity run set: expected {expected_total}, got {len(results)}"
        )


def run_sensitivity_campaign(
    *,
    analysis: SensitivityAnalysis,
    lane: str,
    task_id: str,
    env_backend: str,
    device: str,
    model_profile: str,
    run_root: Path,
    python_executable: str | None = None,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> SensitivityReport:
    """Run the full Dreamer sensitivity campaign and return the aggregate report."""
    results: list[SweepResult] = []
    configs = analysis.generate_run_configs()
    total = len(configs)
    for index, config in enumerate(configs, start=1):
        if progress_callback is not None:
            progress_callback(index, total, config)
        results.append(
            run_sweep_config(
                config,
                lane=lane,
                task_id=task_id,
                env_backend=env_backend,
                device=device,
                model_profile=model_profile,
                run_root=run_root,
                python_executable=python_executable,
            )
        )

    _validate_results(results, analysis.total_runs)
    analysis.add_results(results)
    report = analysis.generate_report()
    report.task_id = str(task_id)
    report.env_backend = str(env_backend)
    report.model_profile = str(model_profile)
    return report
