# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Evaluation result and report containers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalResult:
    """Single evaluation metric result."""

    suite: str
    metric: str
    value: float
    threshold: float | None
    passed: bool | None  # None = informational
    timestamp: float
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EvalReport:
    """Aggregated evaluation report for a suite run."""

    suite: str
    model_id: str
    results: tuple[EvalResult, ...]  # tuple for frozen dataclass
    timestamp: float
    wall_time_sec: float
    all_passed: bool | None
    mode: str = "synthetic"
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "suite": self.suite,
            "model_id": self.model_id,
            "results": [r.to_dict() for r in self.results],
            "timestamp": self.timestamp,
            "wall_time_sec": self.wall_time_sec,
            "all_passed": self.all_passed,
            "mode": self.mode,
        }
        provenance = dict(self.provenance)
        if self.mode == "dataset_replay":
            payload["dataset_replay_provenance"] = provenance
            payload["real_provenance"] = provenance
        elif self.mode == "env_policy":
            payload["env_policy_provenance"] = provenance
            payload["real_provenance"] = provenance
        elif self.mode == "real":
            payload["real_provenance"] = provenance
        else:
            if not provenance:
                provenance = {"kind": "synthetic", "label": "synthetic evaluation input"}
            payload["synthetic_provenance"] = provenance
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
