# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Hyperparameter sensitivity analysis infrastructure.

Defines parameter sweep specifications, one-at-a-time sensitivity analysis,
result aggregation, and ranking of hyperparameter importance. The canonical
execution path uses the Dreamer native runner on a small Atari task with
multiple seeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterSweep:
    """Definition of a single hyperparameter sweep.

    Parameters
    ----------
    name:
        Config attribute name (e.g. ``"kl_dynamics"``).
    values:
        Grid of values to test.
    default:
        The production default value.
    description:
        Human-readable description for reports.
    """

    name: str
    values: list[float]
    default: float
    description: str = ""


@dataclass(frozen=True)
class SweepResult:
    """Result of a single (parameter, value, seed) evaluation."""

    param_name: str
    param_value: float
    seed: int
    final_reward: float
    steps: int


@dataclass
class ParameterSensitivity:
    """Aggregated sensitivity for one hyperparameter."""

    name: str
    default_value: float
    values: list[float]
    mean_rewards: list[float]
    std_rewards: list[float]
    sensitivity_score: float = 0.0
    default_rank_percentile: float = 0.0

    @property
    def default_in_safe_range(self) -> bool:
        """True when the default is in the middle 50% of tested values."""
        return 25.0 <= self.default_rank_percentile <= 75.0


@dataclass
class SensitivityReport:
    """Full sensitivity analysis report."""

    family: str
    environment: str
    seeds: list[int]
    total_steps: int
    parameters: list[ParameterSensitivity]
    generated_at_utc: str = ""
    task_id: str = ""
    env_backend: str = ""
    model_profile: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at_utc:
            self.generated_at_utc = (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            )

    def to_json_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "worldflux.sensitivity.v1",
            "generated_at_utc": self.generated_at_utc,
            "family": self.family,
            "environment": self.environment,
            "seeds": self.seeds,
            "total_steps": self.total_steps,
            "total_runs": sum(len(param.values) for param in self.parameters)
            * max(1, len(self.seeds)),
            "parameters": [],
        }
        if self.task_id:
            payload["task_id"] = self.task_id
        if self.env_backend:
            payload["env_backend"] = self.env_backend
        if self.model_profile:
            payload["model_profile"] = self.model_profile
        for param in self.parameters:
            payload["parameters"].append(
                {
                    "name": param.name,
                    "default_value": param.default_value,
                    "values": param.values,
                    "mean_rewards": param.mean_rewards,
                    "std_rewards": param.std_rewards,
                    "sensitivity_score": param.sensitivity_score,
                    "default_rank_percentile": param.default_rank_percentile,
                    "default_in_safe_range": param.default_in_safe_range,
                }
            )
        return payload

    @classmethod
    def from_json_payload(cls, raw: dict[str, Any]) -> SensitivityReport:
        params: list[ParameterSensitivity] = []
        for payload in raw.get("parameters", []):
            params.append(
                ParameterSensitivity(
                    name=payload["name"],
                    default_value=payload["default_value"],
                    values=payload["values"],
                    mean_rewards=payload["mean_rewards"],
                    std_rewards=payload["std_rewards"],
                    sensitivity_score=payload["sensitivity_score"],
                    default_rank_percentile=payload["default_rank_percentile"],
                )
            )
        return cls(
            family=raw.get("family", "dreamerv3"),
            environment=raw.get("environment", "unknown"),
            seeds=raw.get("seeds", []),
            total_steps=raw.get("total_steps", 0),
            parameters=params,
            generated_at_utc=raw.get("generated_at_utc", ""),
            task_id=raw.get("task_id", ""),
            env_backend=raw.get("env_backend", ""),
            model_profile=raw.get("model_profile", ""),
        )


# ---------------------------------------------------------------------------
# Default DreamerV3 sweep definitions (from PRD ML-04)
# ---------------------------------------------------------------------------

DREAMERV3_SWEEPS: list[ParameterSweep] = [
    ParameterSweep(
        name="kl_dynamics",
        values=[0.1, 0.3, 0.5, 1.0, 2.0],
        default=0.5,
        description="KL loss scale for dynamics prior",
    ),
    ParameterSweep(
        name="kl_representation",
        values=[0.01, 0.05, 0.1, 0.5, 1.0],
        default=0.1,
        description="KL loss scale for representation posterior",
    ),
    ParameterSweep(
        name="free_nats",
        values=[0.0, 0.5, 1.0, 2.0, 5.0],
        default=1.0,
        description="Free nats threshold for KL loss",
    ),
    ParameterSweep(
        name="learning_rate",
        values=[3e-5, 1e-4, 3e-4, 1e-3],
        default=1e-4,
        description="Optimizer learning rate",
    ),
    ParameterSweep(
        name="imagination_horizon",
        values=[5.0, 10.0, 15.0, 20.0, 30.0],
        default=15.0,
        description="Imagination rollout horizon for actor-critic",
    ),
]


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------


class SensitivityAnalysis:
    """One-at-a-time hyperparameter sensitivity analysis.

    For each parameter, all other parameters are held at their defaults while
    the target is varied across the specified grid. Each configuration is
    evaluated with multiple seeds.

    Parameters
    ----------
    sweeps:
        List of ``ParameterSweep`` definitions.
    family:
        Model family name (e.g. ``"dreamerv3"``).
    environment:
        Environment used for evaluation (e.g. ``"CartPole-v1"``).
    seeds:
        Random seeds for each configuration.
    total_steps:
        Environment steps per run.
    """

    def __init__(
        self,
        sweeps: list[ParameterSweep] | None = None,
        family: str = "dreamerv3",
        environment: str = "atari100k_pong",
        seeds: list[int] | None = None,
        total_steps: int = 100_000,
    ) -> None:
        self.sweeps = sweeps if sweeps is not None else DREAMERV3_SWEEPS
        self.family = family
        self.environment = environment
        self.seeds = seeds if seeds is not None else [0, 1, 2]
        self.total_steps = total_steps
        self._results: list[SweepResult] = []

    @property
    def total_runs(self) -> int:
        """Total number of (param, value, seed) combinations."""
        return sum(len(s.values) for s in self.sweeps) * len(self.seeds)

    def generate_run_configs(self) -> list[dict[str, Any]]:
        """Generate all run configurations for the sensitivity sweep.

        Returns a list of dicts, each containing the parameter overrides,
        seed, and metadata needed to launch one training run.
        """
        configs: list[dict[str, Any]] = []
        for sweep in self.sweeps:
            for value in sweep.values:
                for seed in self.seeds:
                    overrides = {s.name: s.default for s in self.sweeps}
                    overrides[sweep.name] = value
                    configs.append(
                        {
                            "param_name": sweep.name,
                            "param_value": value,
                            "seed": seed,
                            "family": self.family,
                            "environment": self.environment,
                            "total_steps": self.total_steps,
                            "overrides": overrides,
                        }
                    )
        return configs

    def add_result(self, result: SweepResult) -> None:
        """Record an individual sweep result."""
        self._results.append(result)

    def add_results(self, results: list[SweepResult]) -> None:
        """Record multiple sweep results."""
        self._results.extend(results)

    def aggregate(self) -> list[ParameterSensitivity]:
        """Aggregate results into per-parameter sensitivity scores.

        The sensitivity score is the coefficient of variation of mean
        rewards across parameter values - higher means the parameter has
        a larger effect on performance.
        """
        sensitivities: list[ParameterSensitivity] = []

        for sweep in self.sweeps:
            param_results = [r for r in self._results if r.param_name == sweep.name]

            value_means: list[float] = []
            value_stds: list[float] = []

            for val in sweep.values:
                val_scores = [
                    r.final_reward for r in param_results if abs(r.param_value - val) < 1e-12
                ]
                if val_scores:
                    value_means.append(float(np.mean(val_scores)))
                    value_stds.append(
                        float(np.std(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0
                    )
                else:
                    value_means.append(0.0)
                    value_stds.append(0.0)

            # Sensitivity: coefficient of variation of means
            arr_means = np.array(value_means)
            overall_mean = float(arr_means.mean()) if len(arr_means) > 0 else 0.0
            if overall_mean != 0:
                sensitivity_score = float(arr_means.std() / abs(overall_mean))
            else:
                sensitivity_score = 0.0

            # Compute percentile rank of the default value
            default_scores = [
                r.final_reward for r in param_results if abs(r.param_value - sweep.default) < 1e-12
            ]
            default_mean = float(np.mean(default_scores)) if default_scores else 0.0
            if value_means:
                rank = sum(1 for m in value_means if m <= default_mean)
                default_percentile = 100.0 * rank / len(value_means)
            else:
                default_percentile = 50.0

            sensitivities.append(
                ParameterSensitivity(
                    name=sweep.name,
                    default_value=sweep.default,
                    values=sweep.values,
                    mean_rewards=value_means,
                    std_rewards=value_stds,
                    sensitivity_score=sensitivity_score,
                    default_rank_percentile=default_percentile,
                )
            )

        # Sort by sensitivity (most sensitive first)
        sensitivities.sort(key=lambda s: s.sensitivity_score, reverse=True)
        return sensitivities

    def generate_report(self) -> SensitivityReport:
        """Generate the full sensitivity report."""
        return SensitivityReport(
            family=self.family,
            environment=self.environment,
            seeds=self.seeds,
            total_steps=self.total_steps,
            parameters=self.aggregate(),
        )

    def export_json(self, path: Path) -> None:
        """Serialize the report to JSON."""
        report = self.generate_report()
        payload = report.to_json_payload()
        payload["total_runs"] = self.total_runs
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_sensitivity_markdown(report: SensitivityReport) -> str:
    """Render a sensitivity report as Markdown."""
    lines = [
        "# Hyperparameter Sensitivity Analysis",
        "",
        f"Generated at: {report.generated_at_utc}",
        f"Family: {report.family}",
        f"Environment: {report.environment}",
        f"Task ID: {report.task_id or report.environment}",
        f"Env backend: {report.env_backend or 'unknown'}",
        f"Model profile: {report.model_profile or 'unknown'}",
        f"Seeds: {report.seeds}",
        f"Steps per run: {report.total_steps:,}",
        "",
        "This is an initial measured Dreamer sensitivity report. It is not a proof claim or a benchmark claim.",
        "",
        "## Sensitivity Ranking",
        "",
        "| Rank | Parameter | Sensitivity Score | Default | Default Safe |",
        "| ---: | --- | ---: | ---: | --- |",
    ]

    for rank, param in enumerate(report.parameters, 1):
        safe = "Yes" if param.default_in_safe_range else "No"
        lines.append(
            f"| {rank} | {param.name} | {param.sensitivity_score:.4f} "
            f"| {param.default_value} | {safe} |"
        )

    lines.extend(["", "## Per-Parameter Details", ""])

    for param in report.parameters:
        lines.append(f"### {param.name}")
        lines.append("")
        lines.append("| Value | Mean Reward | Std |")
        lines.append("| ---: | ---: | ---: |")
        for val, mean, std in zip(param.values, param.mean_rewards, param.std_rewards):
            marker = " **(default)**" if abs(val - param.default_value) < 1e-12 else ""
            lines.append(f"| {val}{marker} | {mean:.2f} | {std:.2f} |")
        lines.append("")

    lines.extend(
        [
            "## Reproducing",
            "",
            "```bash",
            "python scripts/run_sensitivity.py --dry-run",
            "",
            "# smoke / deterministic CI-sized execution",
            "python scripts/run_sensitivity.py \\",
            f"    --task-id {report.task_id or report.environment} \\",
            f"    --env-backend {report.env_backend or 'stub'} \\",
            f"    --model-profile {report.model_profile or 'wf12m'} \\",
            f"    --seeds {','.join(str(seed) for seed in report.seeds) or '0'} \\",
            f"    --steps {report.total_steps} \\",
            "    --output reports/parity/sensitivity/dreamerv3_sensitivity.json",
            "",
            "# render markdown from existing results",
            "python scripts/run_sensitivity.py \\",
            "    --report-from reports/parity/sensitivity/dreamerv3_sensitivity.json \\",
            "    --output-md docs/reference/hyperparameter-sensitivity.md",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"
