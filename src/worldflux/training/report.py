"""Structured training report generation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthSignal:
    """A single health signal from training diagnostics."""

    name: str
    status: str  # "healthy" | "warning" | "critical"
    value: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "message": self.message,
        }


@dataclass(frozen=True)
class LossCurveSummary:
    """Summary statistics for the training loss curve."""

    initial_loss: float
    final_loss: float
    best_loss: float
    best_step: int
    convergence_slope: float
    plateau_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "convergence_slope": self.convergence_slope,
            "plateau_detected": self.plateau_detected,
        }


@dataclass(frozen=True)
class TrainingReport:
    """Structured report summarizing a training run."""

    model_id: str
    total_steps: int
    wall_time_sec: float
    final_loss: float
    best_loss: float
    ttfi_sec: float
    throughput_steps_per_sec: float
    health_score: float  # 0.0-1.0
    health_signals: dict[str, HealthSignal] = field(default_factory=dict)
    loss_curve_summary: LossCurveSummary = field(
        default_factory=lambda: LossCurveSummary(
            initial_loss=0.0,
            final_loss=0.0,
            best_loss=0.0,
            best_step=0,
            convergence_slope=0.0,
            plateau_detected=False,
        )
    )
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "total_steps": self.total_steps,
            "wall_time_sec": self.wall_time_sec,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "ttfi_sec": self.ttfi_sec,
            "throughput_steps_per_sec": self.throughput_steps_per_sec,
            "health_score": self.health_score,
            "health_signals": {k: v.to_dict() for k, v in self.health_signals.items()},
            "loss_curve_summary": self.loss_curve_summary.to_dict(),
            "recommendations": self.recommendations,
        }

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
